#!/usr/bin/env python3
"""
端到端系统实验 — 用真实模型测试 TaskRouter 全链路

实验设计：
  E1: 路由准确率 — 本地任务是否被正确路由到本地？
  E2: TQBC 学习收敛 — 路由准确率是否随反馈积累提升？
  E3: 策略效果对比 — direct/cot/cod/structured 哪个最好？
  E4: 缓存命中优化 — 质量感知缓存是否提升命中率？
  E5: Token 效率 — 压缩和 CoD 策略实际节省多少 token？
"""

import os
import sys
import time
import json
import random
import tempfile
import shutil
from collections import defaultdict

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
sys.path.insert(0, SCRIPTS_DIR)

from config import get_config, set_config, RouterConfig
from routing import Task, estimate_complexity, detect_task_type
from models import call_ollama
from reasoning import select_strategy, enhance_prompt_with_strategy, STRATEGY_TOKEN_MULTIPLIER, get_strategy_tracker
from adaptive_compression import compress_adaptive, estimate_importance
from confidence import extract_confidence, extract_confidence_from_text
from tqbc import TQBCRouter, extract_quantile_features
from outcome_cache import OutcomeAwareCache

# ─── 测试任务集（标注了期望路由和期望输出质量）────────────────────

TASKS = [
    # 简单本地任务（应路由到 local，direct 策略）
    {"action": "翻译", "text": "The quick brown fox jumps over the lazy dog.",
     "expected_route": "local", "type": "translation",
     "quality_check": lambda t: len(t) > 5},
    {"action": "分类情感", "text": "这部电影太棒了，强烈推荐！",
     "expected_route": "local", "type": "sentiment",
     "quality_check": lambda t: any(k in t.lower() for k in ["positive", "积极", "正面", "好"])},
    {"action": "提取关键词", "text": "人工智能正在改变医疗行业，深度学习在医学影像诊断中表现出色。",
     "expected_route": "local", "type": "extraction",
     "quality_check": lambda t: any(k in t for k in ["人工智能", "深度学习", "医疗"])},
    {"action": "排序", "text": "5, 3, 8, 1, 9, 2, 7",
     "expected_route": "local", "type": "sort_numbers",
     "quality_check": lambda t: "1" in t and "9" in t},
    {"action": "统计", "text": "苹果,香蕉,橙子,苹果,香蕉,苹果",
     "expected_route": "local", "type": "_count",
     "quality_check": lambda t: "3" in t or "苹果" in t},

    # 中等复杂度任务（需要推理策略）
    {"action": "分析原因", "text": "为什么 Python 在数据科学领域比 Java 更受欢迎？",
     "expected_route": "local", "type": "summarization",
     "quality_check": lambda t: len(t) > 20},
    {"action": "对比优缺点", "text": "React vs Vue 的优缺点对比",
     "expected_route": "local", "type": "summarization",
     "quality_check": lambda t: "react" in t.lower() or "vue" in t.lower() or len(t) > 30},
    {"action": "概括", "text": "本文讨论了人工智能在医疗领域的应用前景，包括影像诊断、药物研发、健康管理等方面。",
     "expected_route": "local", "type": "summarization",
     "quality_check": lambda t: len(t) > 10},

    # 高复杂度任务（可能需要云端）
    {"action": "设计微服务架构", "text": "为一个电商平台设计微服务架构，需要处理百万级并发。",
     "expected_route": "cloud", "type": "summarization",
     "quality_check": lambda t: len(t) > 50},
    {"action": "推理", "text": "如果所有的猫都怕水，Tom 是一只猫，那么 Tom 怕水吗？请用逻辑推理。",
     "expected_route": "local", "type": "summarization",
     "quality_check": lambda t: "怕水" in t or "怕" in t or "yes" in t.lower()},
]


def run_experiment_e1_routing_accuracy(tqbc: TQBCRouter) -> dict:
    """E1: 路由准确率 — 本地任务是否被正确路由到本地？"""
    print("\n" + "=" * 60)
    print("  E1: 路由准确率测试")
    print("=" * 60)

    correct = 0
    total = 0
    details = []

    for task_def in TASKS:
        task = Task(action=task_def["action"], text=task_def["text"])
        task_type = detect_task_type(task_def["action"], {})
        routing = estimate_complexity(task)
        score = routing["score"]

        # 策略选择（无 logprobs 的预选）
        strategy_decision = select_strategy(
            action=task_def["action"],
            text=task_def["text"],
            complexity_score=score,
            task_type=task_type,
        )

        # 调用模型
        prompt = enhance_prompt_with_strategy(
            f"{task_def['action']}: {task_def['text']}",
            strategy_decision.strategy,
        )
        result = call_ollama(prompt, with_logprobs=True, max_tokens=200)
        logprobs = result.get("logprobs", [])
        conf_data = extract_confidence(logprobs) if logprobs else extract_confidence_from_text(result["text"])

        # TQBC 路由决策
        tqbc_decision = tqbc.decide(
            logprobs=logprobs,
            complexity_score=score,
            task_type=task_type,
        )

        # 判断路由是否正确
        actual_route = "cloud" if tqbc_decision.should_escalate else "local"
        expected = task_def["expected_route"]
        is_correct = actual_route == expected
        if is_correct:
            correct += 1
        total += 1

        # 质量检查
        quality_ok = task_def["quality_check"](result["text"])

        # 记录 TQBC 结果
        tqbc.record_outcome(
            decision=tqbc_decision,
            success=is_correct and quality_ok,
            escalated=tqbc_decision.should_escalate,
            task_type=task_type,
        )

        status = "V" if is_correct else "X"
        qual = "V" if quality_ok else "X"
        details.append({
            "task": task_def["action"][:10],
            "expected": expected,
            "actual": actual_route,
            "correct": is_correct,
            "quality": quality_ok,
            "confidence": conf_data["confidence"],
            "tqbc_conf": tqbc_decision.calibrated_confidence,
            "should_esc": tqbc_decision.should_escalate,
            "strategy": strategy_decision.strategy,
            "tokens_out": result["tokens_output"],
            "time_ms": result["time_ms"],
        })

        print(f"  [{status}路由 {qual}质量] {task_def['action']:<12} "
              f"期望={expected:<6} 实际={actual_route:<6} "
              f"conf={conf_data['confidence']:.3f} tqbc={tqbc_decision.calibrated_confidence:.3f} "
              f"策略={strategy_decision.strategy} {result['time_ms']}ms {result['tokens_output']}tok")

    accuracy = correct / total if total > 0 else 0
    print(f"\n  路由准确率: {correct}/{total} = {accuracy:.1%}")
    return {"accuracy": accuracy, "correct": correct, "total": total, "details": details}


def run_experiment_e2_learning_curve() -> dict:
    """E2: TQBC 学习收敛 — 路由准确率是否随反馈积累提升？"""
    print("\n" + "=" * 60)
    print("  E2: TQBC 学习收敛曲线")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tqbc = TQBCRouter(os.path.join(tmpdir, "tqbc"))

        checkpoints = [0, 10, 25, 50, 100]
        results = []

        random.seed(42)
        tasks_with_weight = []
        for t in TASKS:
            tasks_with_weight.extend([t] * 3)

        total_decisions = 0
        correct_decisions = 0

        for round_i in range(max(checkpoints) // len(TASKS) + 2):
            for task_def in tasks_with_weight:
                total_decisions += 1
                task = Task(action=task_def["action"], text=task_def["text"])
                routing = estimate_complexity(task)
                logprobs = []  # 模拟简化：不实际调用模型

                tqbc_decision = tqbc.decide(
                    logprobs=logprobs,
                    complexity_score=routing["score"],
                    task_type=detect_task_type(task_def["action"], {}),
                )

                actual = "cloud" if tqbc_decision.should_escalate else "local"
                expected = task_def["expected_route"]
                is_correct = actual == expected
                if is_correct:
                    correct_decisions += 1

                # 模拟反馈
                tqbc.record_outcome(
                    decision=tqbc_decision,
                    success=is_correct and random.random() < 0.9,
                    escalated=tqbc_decision.should_escalate,
                    task_type=detect_task_type(task_def["action"], {}),
                )

                if total_decisions in checkpoints or (round_i > 0 and total_decisions % 50 == 0):
                    if len(results) == 0 or results[-1]["total"] != total_decisions:
                        results.append({
                            "total": total_decisions,
                            "accuracy": correct_decisions / total_decisions,
                        })

        print(f"\n  {'反馈数':<10} {'准确率':<10}")
        print(f"  {'-'*20}")
        for r in results:
            print(f"  {r['total']:<10} {r['accuracy']:.1%}")

        if len(results) >= 2:
            improvement = results[-1]["accuracy"] - results[0]["accuracy"]
            print(f"\n  改进: {improvement:+.1%}")

    return {"checkpoints": results}


def run_experiment_e3_strategy_comparison() -> dict:
    """E3: 策略效果对比 — direct/cot/cod 哪个最好？"""
    print("\n" + "=" * 60)
    print("  E3: 推理策略效果对比")
    print("=" * 60)

    strategies = ["direct", "cot", "cod"]
    test_prompts = [
        ("翻译", "Translate to Chinese: Machine learning is transforming industries."),
        ("提取", "Extract JSON from: 张三, 工号A001, 部门研发部"),
        ("分析", "为什么 Python 在数据科学领域更受欢迎？"),
    ]

    results = {}
    for strategy in strategies:
        strat_results = []
        for name, prompt in test_prompts:
            enhanced = enhance_prompt_with_strategy(prompt, strategy)
            t0 = time.time()
            r = call_ollama(enhanced, with_logprobs=True, max_tokens=200)
            elapsed = (time.time() - t0) * 1000
            logprobs = r.get("logprobs", [])
            conf = extract_confidence(logprobs) if logprobs else {"confidence": 0.0}
            strat_results.append({
                "task": name,
                "time_ms": elapsed,
                "tokens_out": r["tokens_output"],
                "confidence": conf["confidence"],
                "output_len": len(r["text"]),
            })
        avg_time = sum(s["time_ms"] for s in strat_results) / len(strat_results)
        avg_tok = sum(s["tokens_out"] for s in strat_results) / len(strat_results)
        avg_conf = sum(s["confidence"] for s in strat_results) / len(strat_results)
        results[strategy] = {"avg_time": avg_time, "avg_tokens": avg_tok, "avg_confidence": avg_conf, "details": strat_results}

    print(f"\n  {'策略':<10} {'平均延迟':<12} {'平均token':<12} {'平均置信度':<12}")
    print(f"  {'-'*46}")
    for s in strategies:
        r = results[s]
        print(f"  {s:<10} {r['avg_time']:>8.0f}ms   {r['avg_tokens']:>8.0f}    {r['avg_confidence']:>8.4f}")

    # 找最优
    best_by_time = min(strategies, key=lambda s: results[s]["avg_time"])
    best_by_token = min(strategies, key=lambda s: results[s]["avg_tokens"])
    best_by_conf = max(strategies, key=lambda s: results[s]["avg_confidence"])
    print(f"\n  最快: {best_by_time}, 最省token: {best_by_token}, 最高置信度: {best_by_conf}")

    return results


def run_experiment_e4_cache_optimization() -> dict:
    """E4: 缓存命中优化 — 质量感知缓存是否提升命中率？"""
    print("\n" + "=" * 60)
    print("  E4: 缓存质量感知优化")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = OutcomeAwareCache(tmpdir)

        # 模拟缓存命中场景
        keys = [f"task_{i}" for i in range(20)]
        for key in keys:
            # 模拟不同质量的缓存条目
            success_rate = random.uniform(0.3, 1.0)
            for _ in range(10):
                cache.record_outcome(key, success=random.random() < success_rate)

        # 测试质量感知阈值
        fixed_threshold = 0.85
        quality_adjusted = []
        for key in keys:
            q = cache.get_quality(key)
            adjusted = cache.get_effective_threshold(key, fixed_threshold)
            quality_adjusted.append((key, q, adjusted))

        print(f"\n  {'缓存键':<12} {'质量分':<10} {'固定阈值':<10} {'质量阈值':<10} {'效果'}")
        print(f"  {'-'*52}")
        for key, q, adj in sorted(quality_adjusted, key=lambda x: -x[1])[:8]:
            effect = "更容易命中" if adj < fixed_threshold else "更难命中"
            print(f"  {key:<12} {q:<10.3f} {fixed_threshold:<10.2f} {adj:<10.2f} {effect}")

        stats = cache.get_stats()
        print(f"\n  统计: {json.dumps(stats, indent=2)}")

    return {"stats": stats}


def run_experiment_e5_token_efficiency() -> dict:
    """E5: Token 效率 — 压缩和策略实际节省多少 token？"""
    print("\n" + "=" * 60)
    print("  E5: Token 效率分析")
    print("=" * 60)

    test_prompts = [
        ("短prompt", "翻译: Hello world"),
        ("中等prompt", "请分析以下文本的情感倾向，并给出具体的判断依据。文本：这部电影的剧情非常精彩，演员的表演也很到位，但是特效部分稍显不足。"),
        ("长prompt", "\n".join([f"第{i}行：这是测试内容，包含一些重要的信息和数据{i}。" for i in range(30)])),
    ]

    results = {}
    for name, prompt in test_prompts:
        # 不同压缩级别的 token 节省
        compressions = []
        for conf in [0.3, 0.5, 0.7, 0.9]:
            cr = compress_adaptive(prompt, confidence=conf)
            savings = len(prompt) - len(cr.compressed_prompt)
            compressions.append({
                "confidence": conf,
                "level": cr.level,
                "original": cr.original_length,
                "compressed": cr.compressed_length,
                "ratio": cr.compression_ratio,
                "savings_chars": savings,
            })

        results[name] = compressions
        print(f"\n  [{name}] 原始长度: {len(prompt)} 字符")
        print(f"  {'置信度':<10} {'级别':<12} {'压缩后':<10} {'比例':<10} {'节省'}")
        print(f"  {'-'*50}")
        for c in compressions:
            print(f"  {c['confidence']:<10.1f} {c['level']:<12} {c['compressed']:<10} {c['ratio']:<10.2f} {c['savings_chars']} 字符")

    # 策略 token 预算对比
    print(f"\n  策略 Token 预算倍数:")
    for strategy, mult in STRATEGY_TOKEN_MULTIPLIER.items():
        print(f"    {strategy:<12} {mult}x")

    return results


def main():
    print("=" * 60)
    print("  TaskRouter 端到端系统实验")
    print("  模型: qwen-tool (qwen3.5:0.8b)")
    print("=" * 60)

    all_results = {}

    # E3: 策略对比（最快，先跑）
    all_results["e3_strategies"] = run_experiment_e3_strategy_comparison()

    # E5: Token 效率（不需要模型调用）
    all_results["e5_token_efficiency"] = run_experiment_e5_token_efficiency()

    # E4: 缓存优化（不需要模型调用）
    all_results["e4_cache"] = run_experiment_e4_cache_optimization()

    # E1: 路由准确率（需要模型调用）
    with tempfile.TemporaryDirectory() as tmpdir:
        tqbc = TQBCRouter(os.path.join(tmpdir, "tqbc"))
        all_results["e1_routing"] = run_experiment_e1_routing_accuracy(tqbc)

    # E2: 学习收敛（最后跑，需要多轮）
    all_results["e2_learning"] = run_experiment_e2_learning_curve()

    # 保存结果
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  结果已保存: {output_path}")


if __name__ == "__main__":
    main()
