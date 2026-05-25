#!/usr/bin/env python3
"""
学习曲线基准测试 — 展示 TQBC 系统随反馈积累的改进

评估维度：
1. 路由准确率随训练轮数的增长
2. 置信度间隔（Gatekeeper Gap）的演化
3. Thompson Sampling 的探索-利用平衡
4. 校准 ECE 的改善
"""

import os
import sys
import random
import tempfile
import math
from collections import defaultdict

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
sys.path.insert(0, SCRIPTS_DIR)


def make_confident_logprobs(n: int = 10) -> list[dict]:
    return [{"logprob": -0.05, "top_logprobs": {"correct": -0.05, "wrong": -5.0}} for _ in range(n)]


def make_uncertain_logprobs(n: int = 5) -> list[dict]:
    return [{"logprob": -3.0, "top_logprobs": {"a": -2.5, "b": -2.6, "c": -2.8}} for _ in range(n)]


TASKS = [
    # 边界场景：复杂度高但 logprobs 置信 → 本地
    {"type": "translation", "complexity": 5.5, "logprobs_fn": make_confident_logprobs,
     "expected_route": "local", "n_tokens": 18, "weight": 3},
    {"type": "code_gen", "complexity": 5.5, "logprobs_fn": make_confident_logprobs,
     "expected_route": "local", "n_tokens": 15, "weight": 2},
    # 边界场景：复杂度中等但 logprobs 不确定 → 云端
    {"type": "extraction", "complexity": 3.5, "logprobs_fn": make_uncertain_logprobs,
     "expected_route": "cloud", "n_tokens": 4, "weight": 3},
    {"type": "summarization", "complexity": 4.0, "logprobs_fn": make_uncertain_logprobs,
     "expected_route": "cloud", "n_tokens": 3, "weight": 3},
    # 明确场景
    {"type": "translation", "complexity": 1.5, "logprobs_fn": make_confident_logprobs,
     "expected_route": "local", "n_tokens": 15, "weight": 2},
    {"type": "reasoning", "complexity": 8.0, "logprobs_fn": make_uncertain_logprobs,
     "expected_route": "cloud", "n_tokens": 4, "weight": 2},
]


def get_tasks():
    tasks = []
    for t in TASKS:
        tasks.extend([t] * t.get("weight", 1))
    return tasks


def compute_ece(predictions: list[tuple[float, bool]], n_bins: int = 10) -> float:
    bins = defaultdict(lambda: {"conf_sum": 0, "acc_sum": 0, "n": 0})
    for conf, correct in predictions:
        idx = min(int(conf * n_bins), n_bins - 1)
        bins[idx]["conf_sum"] += conf
        bins[idx]["acc_sum"] += float(correct)
        bins[idx]["n"] += 1
    total = len(predictions)
    if total == 0:
        return 0.0
    ece = 0.0
    for b in bins.values():
        if b["n"] > 0:
            ece += (b["n"] / total) * abs(b["conf_sum"] / b["n"] - b["acc_sum"] / b["n"])
    return ece


def run_learning_curve():
    print("=" * 60)
    print("TQBC 学习曲线 — 系统随反馈积累的改进")
    print("=" * 60)

    random.seed(42)
    tasks = get_tasks()

    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "tqbc"), exist_ok=True)
        from tqbc import TQBCRouter

        tqbc = TQBCRouter(os.path.join(tmpdir, "tqbc"))

        # 学习曲线数据点
        checkpoints = [10, 25, 50, 100, 200, 500, 1000, 2000]
        results = []

        total_decisions = 0
        correct_decisions = 0
        predictions = []
        recorded_checkpoints = set()

        for round_i in range(max(checkpoints) + 1):
            for task in tasks:
                total_decisions += 1
                logprobs = task["logprobs_fn"](task.get("n_tokens", 10))
                decision = tqbc.decide(
                    logprobs=logprobs,
                    complexity_score=task["complexity"],
                    task_type=task["type"],
                )

                expected = task["expected_route"]
                is_correct = decision.route == expected
                if is_correct:
                    correct_decisions += 1

                # 模拟反馈（85% 准确率的噪声）
                success = is_correct and random.random() < 0.9
                tqbc.record_outcome(
                    decision=decision,
                    success=success,
                    escalated=decision.should_escalate,
                    task_type=task["type"],
                )

                predictions.append((decision.calibrated_confidence, is_correct))

            # 检查点（每轮结束时检查一次）
            if round_i in checkpoints and round_i not in recorded_checkpoints:
                recorded_checkpoints.add(round_i)
                stats = tqbc.get_stats()
                gap_stats = stats.get("confidence_gap", {})
                recent_preds = predictions[-200:] if len(predictions) > 200 else predictions
                ece = compute_ece(recent_preds)

                results.append({
                    "round": round_i,
                    "accuracy": correct_decisions / total_decisions,
                    "ece": ece,
                    "gap": gap_stats.get("gap", 0),
                    "n_correct": gap_stats.get("n_correct", 0),
                    "n_incorrect": gap_stats.get("n_incorrect", 0),
                    "total": stats.get("total_decisions", 0),
                })

        # 输出结果
        print("\n轮数 | 准确率 | ECE    | 间隔   | 正确数 | 错误数")
        print("-" * 55)
        for r in results:
            print(f"{r['round']:4d} | {r['accuracy']:.1%} | {r['ece']:.4f} | "
                  f"{r['gap']:.4f} | {r['n_correct']:5d} | {r['n_incorrect']:5d}")

        # 分析改进趋势
        if len(results) >= 2:
            first = results[0]
            last = results[-1]
            print("\n" + "=" * 60)
            print("改进分析")
            print("=" * 60)
            print(f"  准确率: {first['accuracy']:.1%} → {last['accuracy']:.1%} "
                  f"({last['accuracy'] - first['accuracy']:+.1%})")
            print(f"  ECE:    {first['ece']:.4f} → {last['ece']:.4f} "
                  f"({last['ece'] - first['ece']:+.4f})")
            print(f"  间隔:   {first['gap']:.4f} → {last['gap']:.4f} "
                  f"({last['gap'] - first['gap']:+.4f})")


if __name__ == "__main__":
    run_learning_curve()
