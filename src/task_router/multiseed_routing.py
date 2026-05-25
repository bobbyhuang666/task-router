#!/usr/bin/env python3
"""
实验 1: 多种子 + 置信区间

改造 benchmark_routing.py，支持 --seeds N 参数。
每个 seed 跑一遍 60 个任务场景，收集：
- 路由准确率
- Conformal 覆盖率
- 成本节省率

最终输出 mean ± 95% CI。
JSON 结果存 results/multiseed_routing.json。
"""

import argparse
import json
import math
import os
import random
import sys
import time

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

# 复用 benchmark_routing.py 的任务定义和评估逻辑
from task_router.benchmark_routing import (
    TASK_SUITE, evaluate_routing, _heuristic_route,
)


def mean_ci_95(values: list[float]) -> dict:
    """计算 mean ± 95% CI"""
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std": 0.0}
    mean = sum(values) / n
    if n < 2:
        return {"mean": mean, "ci_lower": mean, "ci_upper": mean, "std": 0.0}
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(variance)
    # t-value for 95% CI with n-1 degrees of freedom (approximation for n >= 2)
    t_values = {2: 12.706, 5: 2.571, 10: 2.262, 15: 2.145, 20: 2.093, 30: 2.045, 50: 2.010, 100: 1.984}
    t = 2.0  # default
    for key in sorted(t_values.keys()):
        if n - 1 <= key:
            t = t_values[key]
            break
    else:
        t = 1.96  # z for large n
    margin = t * std / math.sqrt(n)
    return {
        "mean": round(mean, 4),
        "ci_lower": round(mean - margin, 4),
        "ci_upper": round(mean + margin, 4),
        "std": round(std, 4),
        "n": n,
    }


def run_multiseed(seeds: int = 10, verbose: bool = False) -> dict:
    """运行多种子评估"""
    print(f"\n{'='*70}")
    print(f"实验 1: 多种子路由评估 (seeds={seeds})")
    print(f"{'='*70}")

    accuracies = []
    coverages = []
    savings = []
    all_results = []

    start_time = time.monotonic()

    for seed_idx in range(seeds):
        seed = seed_idx * 1000 + 42
        random.seed(seed)

        result = evaluate_routing(TASK_SUITE, verbose=False, route_fn=_heuristic_route)

        acc = result.correct_routes / result.total_tasks * 100
        cov = result.conformal_coverage_hits / result.total_tasks * 100
        sav = (1 - result.total_cost_actual / result.total_cost_cloud) * 100 if result.total_cost_cloud > 0 else 0

        accuracies.append(acc)
        coverages.append(cov)
        savings.append(sav)

        all_results.append({
            "seed": seed,
            "accuracy": round(acc, 2),
            "coverage": round(cov, 2),
            "savings": round(sav, 2),
            "cost_cloud": round(result.total_cost_cloud, 6),
            "cost_actual": round(result.total_cost_actual, 6),
            "elapsed_ms": round(result.elapsed_ms, 1),
        })

        if verbose:
            print(f"  Seed {seed:5d}: acc={acc:.1f}% cov={cov:.1f}% sav={sav:.1f}%")

    elapsed = (time.monotonic() - start_time) * 1000

    # 汇总统计
    acc_stats = mean_ci_95(accuracies)
    cov_stats = mean_ci_95(coverages)
    sav_stats = mean_ci_95(savings)

    print(f"\n{'指标':30} {'Mean':>8} {'95% CI':>20} {'Std':>8}")
    print("-" * 70)
    print(f"{'路由准确率 (%)':30} {acc_stats['mean']:7.2f}% "
          f"[{acc_stats['ci_lower']:.2f}, {acc_stats['ci_upper']:.2f}] "
          f"{acc_stats['std']:7.2f}%")
    print(f"{'Conformal 覆盖率 (%)':30} {cov_stats['mean']:7.2f}% "
          f"[{cov_stats['ci_lower']:.2f}, {cov_stats['ci_upper']:.2f}] "
          f"{cov_stats['std']:7.2f}%")
    print(f"{'成本节省率 (%)':30} {sav_stats['mean']:7.2f}% "
          f"[{sav_stats['ci_lower']:.2f}, {sav_stats['ci_upper']:.2f}] "
          f"{sav_stats['std']:7.2f}%")
    print(f"\n总耗时: {elapsed:.0f}ms ({seeds} seeds × {len(TASK_SUITE)} tasks)")

    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_seeds": seeds,
        "n_tasks": len(TASK_SUITE),
        "accuracy": acc_stats,
        "coverage": cov_stats,
        "savings": sav_stats,
        "per_seed": all_results,
        "elapsed_ms": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="多种子路由评估 (mean ± 95% CI)")
    parser.add_argument("--seeds", type=int, default=10, help="种子数量 (默认 10)")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--output", type=str, default="results/multiseed_routing.json")
    args = parser.parse_args()

    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    result = run_multiseed(seeds=args.seeds, verbose=args.verbose)

    with open(output_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到 {output_path}")


if __name__ == "__main__":
    main()
