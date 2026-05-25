#!/usr/bin/env python3
"""
真实使用实验 — 模拟企业场景的实际任务流

测试场景：
  1. 翻译批量任务（高频简单任务）
  2. 数据提取（中频中等任务）
  3. 文本分析（低频复杂任务）
  4. 代码相关（高复杂度任务）
  5. 混合负载（模拟真实工作流）

记录维度：
  - 每个任务的路由决策（local/cloud/escalated）
  - 策略选择（direct/cot/cod/structured）
  - 延迟、token 消耗
  - 输出质量（规则校验）
  - TQBC 置信度和不确定性
  - 累计统计
"""

import os
import sys
import time
import json
import random
import tempfile
from dataclasses import dataclass, field, asdict
from collections import defaultdict

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
sys.path.insert(0, SCRIPTS_DIR)

from config import get_config, set_config, RouterConfig
from routing import Task, estimate_complexity, detect_task_type
from models import call_ollama
from reasoning import select_strategy, enhance_prompt_with_strategy, STRATEGY_TOKEN_MULTIPLIER, get_strategy_tracker
from confidence import extract_confidence, extract_confidence_from_text
from tqbc import TQBCRouter, extract_quantile_features
from outcome_cache import OutcomeAwareCache


# ─── 数据结构 ──────────────────────────────────────────────

@dataclass
class TaskResult:
    task_id: int
    category: str
    action: str
    text_preview: str
    task_type: str
    complexity_score: float
    strategy: str
    route_decision: str          # local / escalated
    tqbc_should_escalate: bool
    tqbc_confidence: float
    tqbc_uncertainty: float
    model_confidence: float
    quality_ok: bool
    tokens_in: int
    tokens_out: int
    latency_ms: int
    output_preview: str
    error: str = ""


@dataclass
class ExperimentReport:
    start_time: str
    end_time: str
    model: str
    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    avg_latency_ms: float
    total_tokens_in: int
    total_tokens_out: int
    route_distribution: dict = field(default_factory=dict)
    strategy_distribution: dict = field(default_factory=dict)
    category_stats: dict = field(default_factory=dict)
    tqbc_stats: dict = field(default_factory=dict)
    tasks: list = field(default_factory=list)


# ─── 真实任务集（模拟企业场景）──────────────────────────────────

SCENARIOS = {
    "翻译批量": [
        ("翻译", "The quick brown fox jumps over the lazy dog."),
        ("翻译", "Machine learning is transforming industries worldwide."),
        ("翻译", "Please confirm receipt of this email at your earliest convenience."),
        ("翻译成中文", "Artificial intelligence has made significant progress in natural language processing."),
        ("翻译", "The meeting has been rescheduled to next Monday at 3 PM."),
    ],
    "数据提取": [
        ("提取JSON", "张三, 工号A001, 部门研发部, 入职日期2024-01-15"),
        ("提取关键词", "人工智能正在改变医疗行业，深度学习在医学影像诊断中表现出色。"),
        ("提取数字", "订单金额128.5元，数量3件，折扣0.8，运费15元"),
        ("提取邮箱", "联系 alice@example.com 或 bob@test.org，紧急联系 support@company.cn"),
        ("提取日期", "会议定于2024年3月15日下午2点，截止日期为4月1日"),
    ],
    "文本分析": [
        ("分析原因", "为什么 Python 在数据科学领域比 Java 更受欢迎？"),
        ("对比优缺点", "React 和 Vue 框架的优缺点对比"),
        ("概括", "本文讨论了人工智能在医疗领域的应用前景，包括影像诊断、药物研发、健康管理等方面。深度学习技术在CT影像分析中已经达到专家水平。"),
        ("评价", "评估这个商业计划书的可行性：目标用户1000万，首年营收1亿。"),
        ("推理", "如果所有的猫都怕水，Tom 是一只猫，那么 Tom 怕水吗？请用逻辑推理。"),
    ],
    "格式化": [
        ("格式化表格", "姓名:张三 年龄:25 部门:研发\n姓名:李四 年龄:30 部门:产品\n姓名:王五 年龄:28 部门:设计"),
        ("转换JSON", "name=张三,age=25,dept=研发,phone=13812345678"),
        ("排序数字", "5, 3, 8, 1, 9, 2, 7, 4, 6"),
        ("统计", "苹果,香蕉,橙子,苹果,香蕉,苹果,橙子,苹果"),
        ("去重", "apple, banana, apple, cherry, banana, date, cherry"),
    ],
    "代码相关": [
        ("写函数", "Write a Python function to check if a string is a palindrome."),
        ("写正则", "Write a regex to validate email addresses."),
        ("解释代码", "解释这段代码的作用：\n```python\ndef f(n): return n if n < 2 else f(n-1) + f(n-2)\n```"),
        ("Debug", "这段代码有bug，请修复：\n```python\ndef avg(lst): return sum(lst) / len(lst)\nprint(avg([]))\n```"),
        ("优化", "优化这个函数的性能：\n```python\ndef find_dupes(lst):\n    return [x for i, x in enumerate(lst) if x in lst[:i]]\n```"),
    ],
}


def run_task(task_id: int, category: str, action: str, text: str,
             tqbc: TQBCRouter, cache: OutcomeAwareCache) -> TaskResult:
    """执行单个任务并收集所有指标"""
    task = Task(action=action, text=text)
    task_type = detect_task_type(action, {})
    routing = estimate_complexity(task)
    complexity = routing["score"]

    # 策略选择
    strategy_decision = select_strategy(
        action=action, text=text,
        complexity_score=complexity, task_type=task_type,
    )
    strategy = strategy_decision.strategy

    # Prompt 构建
    prompt = enhance_prompt_with_strategy(f"{action}: {text}", strategy)

    # 模型调用
    t0 = time.time()
    try:
        result = call_ollama(prompt, with_logprobs=True, max_tokens=300)
    except Exception as e:
        return TaskResult(
            task_id=task_id, category=category, action=action,
            text_preview=text[:50], task_type=task_type,
            complexity_score=complexity, strategy=strategy,
            route_decision="error", tqbc_should_escalate=False,
            tqbc_confidence=0, tqbc_uncertainty=0, model_confidence=0,
            quality_ok=False, tokens_in=0, tokens_out=0,
            latency_ms=int((time.time() - t0) * 1000),
            output_preview="", error=str(e),
        )
    latency = int((time.time() - t0) * 1000)

    logprobs = result.get("logprobs", [])
    conf_data = extract_confidence(logprobs) if logprobs else extract_confidence_from_text(result["text"])
    model_conf = conf_data["confidence"]

    # TQBC 决策
    tqbc_decision = tqbc.decide(
        logprobs=logprobs, complexity_score=complexity, task_type=task_type,
    )

    # 质量校验
    output = result["text"]
    quality_ok = len(output.strip()) > 3

    # 记录反馈
    tqbc.record_outcome(
        decision=tqbc_decision, success=quality_ok,
        escalated=tqbc_decision.should_escalate, task_type=task_type,
    )

    return TaskResult(
        task_id=task_id, category=category, action=action,
        text_preview=text[:50], task_type=task_type,
        complexity_score=complexity, strategy=strategy,
        route_decision="escalated" if tqbc_decision.should_escalate else "local",
        tqbc_should_escalate=tqbc_decision.should_escalate,
        tqbc_confidence=round(tqbc_decision.calibrated_confidence, 4),
        tqbc_uncertainty=round(tqbc_decision.uncertainty, 4),
        model_confidence=round(model_conf, 4),
        quality_ok=quality_ok,
        tokens_in=result["tokens_input"],
        tokens_out=result["tokens_output"],
        latency_ms=latency,
        output_preview=output[:120],
    )


def generate_report(results: list[TaskResult], start_time: str, end_time: str) -> ExperimentReport:
    """生成实验报告"""
    successful = [r for r in results if not r.error]
    failed = [r for r in results if r.error]

    # 路由分布
    route_dist = defaultdict(int)
    for r in results:
        route_dist[r.route_decision] += 1

    # 策略分布
    strategy_dist = defaultdict(int)
    for r in results:
        strategy_dist[r.strategy] += 1

    # 分类统计
    cat_stats = defaultdict(lambda: {"count": 0, "quality_ok": 0, "avg_latency": 0, "avg_tokens": 0, "latencies": [], "tokenss": []})
    for r in results:
        cs = cat_stats[r.category]
        cs["count"] += 1
        if r.quality_ok:
            cs["quality_ok"] += 1
        cs["latencies"].append(r.latency_ms)
        cs["tokenss"].append(r.tokens_out)
    for cs in cat_stats.values():
        cs["avg_latency"] = round(sum(cs["latencies"]) / len(cs["latencies"])) if cs["latencies"] else 0
        cs["avg_tokens"] = round(sum(cs["tokenss"]) / len(cs["tokenss"])) if cs["tokenss"] else 0
        cs["quality_rate"] = round(cs["quality_ok"] / cs["count"], 3) if cs["count"] else 0
        del cs["latencies"]
        del cs["tokenss"]

    # TQBC 统计
    tqbc_confs = [r.tqbc_confidence for r in results if not r.error]
    model_confs = [r.model_confidence for r in results if not r.error]
    escalations = sum(1 for r in results if r.tqbc_should_escalate)

    return ExperimentReport(
        start_time=start_time,
        end_time=end_time,
        model=get_config().local_model,
        total_tasks=len(results),
        successful_tasks=len(successful),
        failed_tasks=len(failed),
        avg_latency_ms=round(sum(r.latency_ms for r in successful) / len(successful)) if successful else 0,
        total_tokens_in=sum(r.tokens_in for r in results),
        total_tokens_out=sum(r.tokens_out for r in results),
        route_distribution=dict(route_dist),
        strategy_distribution=dict(strategy_dist),
        category_stats=dict(cat_stats),
        tqbc_stats={
            "avg_tqbc_confidence": round(sum(tqbc_confs) / len(tqbc_confs), 4) if tqbc_confs else 0,
            "avg_model_confidence": round(sum(model_confs) / len(model_confs), 4) if model_confs else 0,
            "escalation_count": escalations,
            "escalation_rate": round(escalations / len(results), 3) if results else 0,
        },
        tasks=[asdict(r) for r in results],
    )


def print_report(report: ExperimentReport):
    """打印实验报告"""
    print("\n" + "=" * 70)
    print("  TaskRouter 真实使用实验报告")
    print("=" * 70)

    print(f"\n  时间: {report.start_time} → {report.end_time}")
    print(f"  模型: {report.model}")
    print(f"  总任务: {report.total_tasks}")
    print(f"  成功: {report.successful_tasks}  失败: {report.failed_tasks}")

    print(f"\n  ── 性能指标 ──")
    print(f"  平均延迟: {report.avg_latency_ms}ms")
    print(f"  总输入 token: {report.total_tokens_in}")
    print(f"  总输出 token: {report.total_tokens_out}")

    print(f"\n  ── 路由决策分布 ──")
    for route, count in sorted(report.route_distribution.items()):
        pct = count / report.total_tasks * 100
        bar = "█" * int(pct / 2)
        print(f"  {route:<12} {count:>3} ({pct:>5.1f}%) {bar}")

    print(f"\n  ── 策略使用分布 ──")
    for strategy, count in sorted(report.strategy_distribution.items(), key=lambda x: -x[1]):
        pct = count / report.total_tasks * 100
        bar = "█" * int(pct / 2)
        budget = STRATEGY_TOKEN_MULTIPLIER.get(strategy, 1.0)
        print(f"  {strategy:<12} {count:>3} ({pct:>5.1f}%) {bar}  (token预算: {budget}x)")

    print(f"\n  ── 分类统计 ──")
    print(f"  {'分类':<12} {'数量':<6} {'质量率':<8} {'平均延迟':<10} {'平均token'}")
    print(f"  {'-'*52}")
    for cat, stats in sorted(report.category_stats.items()):
        print(f"  {cat:<12} {stats['count']:<6} {stats['quality_rate']:<8.1%} "
              f"{stats['avg_latency']:>6}ms   {stats['avg_tokens']:>5}")

    print(f"\n  ── TQBC 路由引擎 ──")
    ts = report.tqbc_stats
    print(f"  平均 TQBC 置信度: {ts['avg_tqbc_confidence']:.4f}")
    print(f"  平均模型置信度:   {ts['avg_model_confidence']:.4f}")
    print(f"  升级次数: {ts['escalation_count']} ({ts['escalation_rate']:.1%})")

    print(f"\n  ── 任务详情 ──")
    print(f"  {'#':<4} {'分类':<10} {'操作':<10} {'策略':<10} {'路由':<10} {'TQBC':<8} {'模型':<8} {'质量':<6} {'延迟':<8} {'token'}")
    print(f"  {'-'*90}")
    for t in report.tasks:
        qual = "V" if t["quality_ok"] else "X"
        err = "ERR" if t["error"] else ""
        print(f"  {t['task_id']:<4} {t['category']:<10} {t['action']:<10} {t['strategy']:<10} "
              f"{t['route_decision']:<10} {t['tqbc_confidence']:<8.3f} {t['model_confidence']:<8.3f} "
              f"{qual:<6} {t['latency_ms']:>5}ms  {t['tokens_out']:>4}tok {err}")

    # 分析与建议
    print(f"\n  ── 分析与建议 ──")
    escalations = report.tqbc_stats["escalation_count"]
    total = report.total_tasks
    if escalations / total > 0.3:
        print(f"  ⚠ 升级率偏高 ({escalations/total:.0%})，建议降低 ESCALATION_THRESHOLD")
    if report.avg_latency_ms > 5000:
        print(f"  ⚠ 平均延迟偏高 ({report.avg_latency_ms}ms)，建议增加 CoD 使用率")
    cod_count = report.strategy_distribution.get("cod", 0)
    if cod_count / total < 0.3:
        print(f"  ⚠ CoD 使用率偏低 ({cod_count/total:.0%})，可进一步优化策略选择")
    quality_rate = report.successful_tasks / report.total_tasks
    if quality_rate >= 0.95:
        print(f"  ✓ 质量优秀 ({quality_rate:.0%})")
    elif quality_rate >= 0.8:
        print(f"  ✓ 质量良好 ({quality_rate:.0%})")
    else:
        print(f"  ⚠ 质量需改进 ({quality_rate:.0%})")


def main():
    start_time = time.strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print("  TaskRouter 真实使用实验")
    print(f"  开始时间: {start_time}")
    print(f"  模型: {get_config().local_model}")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        tqbc = TQBCRouter(os.path.join(tmpdir, "tqbc"))
        cache = OutcomeAwareCache(os.path.join(tmpdir, "cache"))

        results = []
        task_id = 0

        for category, tasks in SCENARIOS.items():
            print(f"\n  [{category}] {len(tasks)} 个任务...")
            for action, text in tasks:
                task_id += 1
                print(f"    {task_id:>2}. {action}: {text[:40]}...", end=" ", flush=True)
                result = run_task(task_id, category, action, text, tqbc, cache)
                results.append(result)
                status = "V" if result.quality_ok else "X"
                route = result.route_decision
                print(f"{status} {route} {result.strategy} {result.latency_ms}ms {result.tokens_out}tok")

        end_time = time.strftime("%Y-%m-%d %H:%M:%S")

        # 生成报告
        report = generate_report(results, start_time, end_time)
        print_report(report)

        # 保存
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment_report.json")
        with open(output_path, "w") as f:
            json.dump(asdict(report), f, indent=2, ensure_ascii=False)
        print(f"\n  报告已保存: {output_path}")


if __name__ == "__main__":
    main()
