#!/usr/bin/env python3
"""
高难度任务实验 — 测试 TaskRouter 的能力上限

难度维度：
  L1: 多步推理（需要链式思考）
  L2: 约束满足（多个条件同时满足）
  L3: 类比迁移（跨领域知识运用）
  L4: 反事实推理（假设-推导-验证）
  L5: 综合分析（多源信息整合）
"""

import os
import sys
import time
import json
import tempfile
from dataclasses import dataclass, asdict
from collections import defaultdict

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
sys.path.insert(0, SCRIPTS_DIR)

from config import get_config
from routing import Task, estimate_complexity, detect_task_type
from models import call_ollama
from reasoning import select_strategy, enhance_prompt_with_strategy, STRATEGY_TOKEN_MULTIPLIER
from confidence import extract_confidence, extract_confidence_from_text
from tqbc import TQBCRouter


@dataclass
class HardTaskResult:
    task_id: int
    difficulty: str
    category: str
    prompt_preview: str
    strategy: str
    route: str
    tqbc_conf: float
    model_conf: float
    tokens_out: int
    latency_ms: int
    output: str
    quality_score: float   # 0-1, 人工评估标准
    quality_notes: str


# ─── 高难度任务集 ──────────────────────────────────────────

HARD_TASKS = [
    # L1: 多步推理
    {
        "difficulty": "L1-多步推理",
        "category": "逻辑",
        "prompt": "小明比小红大3岁，小红比小华大2岁，小华今年10岁。请问小明今年几岁？如果3年后小明的年龄是小华年龄的几倍？",
        "check": lambda t: "15" in t and ("1.2" in t or "6/5" in t or "1.2倍" in t),
        "quality_note": "需要两步计算：小明=10+2+3=15，3年后 18/13≈1.38",
    },
    {
        "difficulty": "L1-多步推理",
        "category": "数学",
        "prompt": "一个水池有两个水管。A管每小时注入3吨水，B管每小时排出1吨水。水池初始有5吨水，容量为20吨。请问几小时后水池满？满后关闭A管，仅开B管，几小时后水池空？",
        "check": lambda t: "7.5" in t or "7又" in t or ("7" in t and "20" in t),
        "quality_note": "注入速率 3-1=2吨/h，(20-5)/2=7.5h满，20/1=20h空",
    },
    {
        "difficulty": "L1-多步推理",
        "category": "逻辑",
        "prompt": "甲说：乙在说谎。乙说：丙在说谎。丙说：甲和乙都在说谎。请问谁在说真话，谁在说谎？",
        "check": lambda t: ("乙" in t and "真" in t) or ("乙说真" in t) or ("乙是诚实" in t),
        "quality_note": "经典逻辑题：乙说真话，甲和丙说谎",
    },

    # L2: 约束满足
    {
        "difficulty": "L2-约束满足",
        "category": "规划",
        "prompt": "我有5个会议要安排在周一到周五，每天一个。已知条件：\n1. 会议A不能在周一\n2. 会议B必须在会议C之前\n3. 会议D必须在周三\n4. 会议E不能在周五\n请给出一种合理的安排。",
        "check": lambda t: "d" in t.lower() and "wednesday" in t.lower() or "周三" in t or ("D" in t and "三" in t),
        "quality_note": "D必须在周三，其他按约束排列",
    },
    {
        "difficulty": "L2-约束满足",
        "category": "编码",
        "prompt": "写一个Python函数，输入一个整数列表，返回满足以下所有条件的子列表：\n1. 长度至少为2\n2. 元素严格递增\n3. 相邻元素差恰好为1（连续整数）\n4. 返回最长的一个",
        "check": lambda t: "def " in t and ("range" in t or "append" in t or "len" in t),
        "quality_note": "需要实现最长连续递增子序列",
    },
    {
        "difficulty": "L2-约束满足",
        "category": "优化",
        "prompt": "一个背包容量为10kg，有以下物品：\nA(3kg,价值4), B(4kg,价值5), C(5kg,价值7), D(2kg,价值3), E(1kg,价值2)\n请问如何选择使总价值最大？列出选择和总价值。",
        "check": lambda t: ("13" in t or "14" in t) and any(x in t for x in ["C", "D", "E", "A"]),
        "quality_note": "最优解: C+D+E=8kg,价值12 或 A+C=8kg,价值11 或 C+D+A=10kg,价值14",
    },

    # L3: 类比迁移
    {
        "difficulty": "L3-类比迁移",
        "category": "类比",
        "prompt": "医生:医院 = 教师:? 请回答并解释推理过程。然后再给出3个类似的类比。",
        "check": lambda t: "学校" in t or "教室" in t or "学院" in t,
        "quality_note": "医生在医院工作，教师在学校工作",
    },
    {
        "difficulty": "L3-类比迁移",
        "category": "跨域",
        "prompt": "用物理学中的'熵增定律'来类比解释为什么企业组织会自然趋向混乱，以及如何对抗这种趋势。",
        "check": lambda t: len(t) > 100 and ("熵" in t or "管理" in t or "制度" in t or "流程" in t),
        "quality_note": "需要跨领域类比能力",
    },
    {
        "difficulty": "L3-类比迁移",
        "category": "模式",
        "prompt": "观察以下序列的规律，给出下一个数：\n2, 6, 12, 20, 30, ?",
        "check": lambda t: "42" in t,
        "quality_note": "n*(n+1): 1*2=2, 2*3=6, 3*4=12, 4*5=20, 5*6=30, 6*7=42",
    },

    # L4: 反事实推理
    {
        "difficulty": "L4-反事实",
        "category": "历史",
        "prompt": "如果互联网在1900年就被发明，世界历史会有什么不同？请从政治、经济、文化三个角度分析。",
        "check": lambda t: len(t) > 150 and any(k in t for k in ["政治", "经济", "文化", "战争", "革命"]),
        "quality_note": "需要反事实推理+多角度分析",
    },
    {
        "difficulty": "L4-反事实",
        "category": "科学",
        "prompt": "假设地球的自转速度突然变为现在的2倍，会产生哪些连锁反应？从天气、海洋、生态、人类生活四个方面分析。",
        "check": lambda t: len(t) > 100 and any(k in t for k in ["天", "昼", "风", "海", "离心"]),
        "quality_note": "白天缩短、风力增强、赤道膨胀等",
    },
    {
        "difficulty": "L4-反事实",
        "category": "技术",
        "prompt": "如果所有编程语言都突然消失，只剩下汇编语言，软件行业会如何应对？短期(1年)和长期(10年)分别会怎样？",
        "check": lambda t: len(t) > 100 and ("汇编" in t or "编译" in t or "效率" in t),
        "quality_note": "短期混乱，长期会重建高级语言",
    },

    # L5: 综合分析
    {
        "difficulty": "L5-综合分析",
        "category": "商业",
        "prompt": "分析以下商业场景并给出建议：\n一家SaaS公司，月收入100万，月增长率15%，月流失率8%，获客成本5000元/客户，月均ARPU 2000元。\n请计算：1)月净增长率 2)客户LTV 3)LTV/CAC比 4)是否健康？5)改进建议",
        "check": lambda t: ("7" in t or "净增长" in t) and ("LTV" in t or "生命周期" in t),
        "quality_note": "净增长=15%-8%=7%, LTV=2000/0.08=25000, LTV/CAC=5",
    },
    {
        "difficulty": "L5-综合分析",
        "category": "技术架构",
        "prompt": "设计一个支持千万级用户的实时聊天系统架构。需要考虑：消息存储、实时推送、群聊、消息搜索、已读回执。给出核心组件和技术选型。",
        "check": lambda t: len(t) > 200 and any(k in t.lower() for k in ["websocket", "kafka", "redis", "mysql", "mongodb", "推送", "消息队列"]),
        "quality_note": "需要系统设计能力",
    },
    {
        "difficulty": "L5-综合分析",
        "category": "论文分析",
        "prompt": "用300字以内总结Transformer架构的核心创新，解释Self-Attention机制为什么比RNN更适合处理长序列，并指出Transformer的一个主要缺点。",
        "check": lambda t: "attention" in t.lower() or "注意力" in t and ("并行" in t or "长距离" in t or "O(n" in t),
        "quality_note": "核心：自注意力机制、并行计算、长距离依赖；缺点：O(n²)复杂度",
    },
]


def run_hard_task(tqbc: TQBCRouter, task_def: dict, task_id: int) -> HardTaskResult:
    """执行单个高难度任务"""
    prompt = task_def["prompt"]
    t0 = time.time()

    try:
        result = call_ollama(prompt, with_logprobs=True, max_tokens=500)
    except Exception as e:
        return HardTaskResult(
            task_id=task_id, difficulty=task_def["difficulty"],
            category=task_def["category"], prompt_preview=prompt[:60],
            strategy="error", route="error", tqbc_conf=0, model_conf=0,
            tokens_out=0, latency_ms=int((time.time()-t0)*1000),
            output="", quality_score=0, quality_notes=str(e),
        )
    latency = int((time.time() - t0) * 1000)
    output = result["text"]
    logprobs = result.get("logprobs", [])

    # 策略选择（用 logprobs）
    task_type = detect_task_type(prompt[:20], {})
    routing = estimate_complexity(Task(action=prompt[:50], text=prompt))
    strategy_decision = select_strategy(
        action=prompt[:100], text=prompt,
        complexity_score=routing["score"], task_type=task_type,
        logprobs=logprobs,
    )

    # TQBC 决策
    tqbc_decision = tqbc.decide(
        logprobs=logprobs,
        complexity_score=routing["score"],
        task_type=task_type,
    )

    # 置信度
    conf_data = extract_confidence(logprobs) if logprobs else extract_confidence_from_text(output)

    # 质量评估
    quality_ok = task_def["check"](output)
    quality_score = 1.0 if quality_ok else 0.0
    if len(output.strip()) < 10:
        quality_score = 0.0

    # 记录反馈
    tqbc.record_outcome(
        decision=tqbc_decision, success=quality_ok,
        escalated=tqbc_decision.should_escalate, task_type=task_type,
    )

    return HardTaskResult(
        task_id=task_id, difficulty=task_def["difficulty"],
        category=task_def["category"], prompt_preview=prompt[:60],
        strategy=strategy_decision.strategy,
        route="escalated" if tqbc_decision.should_escalate else "local",
        tqbc_conf=round(tqbc_decision.calibrated_confidence, 4),
        model_conf=round(conf_data["confidence"], 4),
        tokens_out=result["tokens_output"],
        latency_ms=latency,
        output=output,
        quality_score=quality_score,
        quality_notes=task_def.get("quality_note", ""),
    )


def main():
    print("=" * 70)
    print("  TaskRouter 高难度任务实验")
    print(f"  模型: {get_config().local_model}")
    print(f"  任务数: {len(HARD_TASKS)}")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        tqbc = TQBCRouter(os.path.join(tmpdir, "tqbc"))
        results = []

        for i, task_def in enumerate(HARD_TASKS):
            task_id = i + 1
            diff = task_def["difficulty"]
            cat = task_def["category"]
            preview = task_def["prompt"][:50]
            print(f"\n  [{task_id:>2}] {diff} | {cat} | {preview}...")
            print(f"       ", end="", flush=True)

            r = run_hard_task(tqbc, task_def, task_id)
            results.append(r)

            qual = "V" if r.quality_score > 0 else "X"
            print(f"{qual} {r.strategy:<10} {r.route:<10} "
                  f"tqbc={r.tqbc_conf:.3f} model={r.model_conf:.3f} "
                  f"{r.latency_ms:>6}ms {r.tokens_out:>4}tok")
            if r.quality_score == 0:
                print(f"       期望: {r.quality_notes}")
                print(f"       实际: {r.output[:120]}...")

        # 汇总
        print("\n" + "=" * 70)
        print("  实验结果汇总")
        print("=" * 70)

        total = len(results)
        passed = sum(1 for r in results if r.quality_score > 0)
        avg_lat = sum(r.latency_ms for r in results) / total
        avg_tok = sum(r.tokens_out for r in results) / total

        print(f"\n  总任务: {total}")
        print(f"  通过: {passed}/{total} = {passed/total:.0%}")
        print(f"  平均延迟: {avg_lat:.0f}ms")
        print(f"  平均输出 token: {avg_tok:.0f}")

        # 按难度统计
        print(f"\n  ── 按难度统计 ──")
        by_diff = defaultdict(lambda: {"total": 0, "passed": 0, "latencies": [], "tokens": []})
        for r in results:
            d = by_diff[r.difficulty]
            d["total"] += 1
            if r.quality_score > 0:
                d["passed"] += 1
            d["latencies"].append(r.latency_ms)
            d["tokens"].append(r.tokens_out)

        print(f"  {'难度':<16} {'通过':<8} {'通过率':<8} {'平均延迟':<12} {'平均token'}")
        print(f"  {'-'*56}")
        for diff, stats in sorted(by_diff.items()):
            avg_l = sum(stats["latencies"]) / len(stats["latencies"])
            avg_t = sum(stats["tokens"]) / len(stats["tokens"])
            rate = stats["passed"] / stats["total"]
            print(f"  {diff:<16} {stats['passed']}/{stats['total']:<5} {rate:<8.0%} {avg_l:>8.0f}ms   {avg_t:>6.0f}")

        # 按策略统计
        print(f"\n  ── 按策略统计 ──")
        by_strat = defaultdict(lambda: {"total": 0, "passed": 0, "latencies": []})
        for r in results:
            s = by_strat[r.strategy]
            s["total"] += 1
            if r.quality_score > 0:
                s["passed"] += 1
            s["latencies"].append(r.latency_ms)

        print(f"  {'策略':<12} {'通过':<8} {'通过率':<8} {'平均延迟'}")
        print(f"  {'-'*40}")
        for strat, stats in sorted(by_strat.items(), key=lambda x: -x[1]["passed"]):
            avg_l = sum(stats["latencies"]) / len(stats["latencies"])
            rate = stats["passed"] / stats["total"]
            print(f"  {strat:<12} {stats['passed']}/{stats['total']:<5} {rate:<8.0%} {avg_l:>8.0f}ms")

        # TQBC 分析
        print(f"\n  ── TQBC 路由分析 ──")
        escalated = [r for r in results if r.route == "escalated"]
        local = [r for r in results if r.route == "local"]
        print(f"  本地处理: {len(local)} 个, 质量通过 {sum(1 for r in local if r.quality_score>0)}/{len(local)}")
        print(f"  升级信号: {len(escalated)} 个, 质量通过 {sum(1 for r in escalated if r.quality_score>0)}/{len(escalated)}")
        if local:
            print(f"  本地平均置信度: {sum(r.tqbc_conf for r in local)/len(local):.4f}")
        if escalated:
            print(f"  升级平均置信度: {sum(r.tqbc_conf for r in escalated)/len(escalated):.4f}")

        # 详细任务列表
        print(f"\n  ── 任务详情 ──")
        print(f"  {'#':<4} {'难度':<14} {'策略':<10} {'路由':<10} {'质量':<6} {'延迟':<8} {'token':<8} {'TQBC'}")
        print(f"  {'-'*72}")
        for r in results:
            qual = "V" if r.quality_score > 0 else "X"
            print(f"  {r.task_id:<4} {r.difficulty:<14} {r.strategy:<10} {r.route:<10} "
                  f"{qual:<6} {r.latency_ms:>5}ms {r.tokens_out:>5}tok {r.tqbc_conf:.3f}")

        # 保存
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment_hard_results.json")
        with open(output_path, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)
        print(f"\n  结果已保存: {output_path}")


if __name__ == "__main__":
    main()
