#!/usr/bin/env python3
"""
实验 4: 路由决策延迟微基准

- 测量单次 decide() 调用的 p50/p95/p99 延迟（微秒）
- 跑 10000 次调用
- 分别测：有锁 vs 无锁路径
- 结果存 results/latency.json
"""

import json
import math
import os
import random
import sys
import threading
import time

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from task_router.tqbc import (
    TQBCRouter, extract_quantile_features,
    ThompsonSamplingRouter, BayesianConfidenceCalibrator,
)


def make_logprobs(n=10, confident=True):
    if confident:
        return [{"logprob": -0.05, "top_logprobs": {"correct": -0.05, "wrong": -5.0}} for _ in range(n)]
    else:
        return [{"logprob": -3.0, "top_logprobs": {"a": -2.5, "b": -2.6, "c": -2.8}} for _ in range(n)]


def percentile(data, p):
    """计算百分位数"""
    k = (len(data) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return data[int(k)]
    return data[f] * (c - k) + data[c] * (k - f)


def compute_stats(latencies_us: list[float]) -> dict:
    """计算延迟统计"""
    sorted_lats = sorted(latencies_us)
    n = len(sorted_lats)
    mean = sum(sorted_lats) / n
    var = sum((x - mean) ** 2 for x in sorted_lats) / n

    return {
        "p50_us": round(percentile(sorted_lats, 50), 1),
        "p95_us": round(percentile(sorted_lats, 95), 1),
        "p99_us": round(percentile(sorted_lats, 99), 1),
        "mean_us": round(mean, 1),
        "min_us": round(sorted_lats[0], 1),
        "max_us": round(sorted_lats[-1], 1),
        "std_us": round(math.sqrt(var), 1),
        "n": n,
    }


def bench_decide_serial(router: TQBCRouter, n_iter: int = 10000) -> dict:
    """串行测试 decide() 延迟（有锁路径）"""
    logprobs_confident = make_logprobs(10, confident=True)
    logprobs_uncertain = make_logprobs(5, confident=False)

    latencies = []
    for i in range(n_iter):
        logprobs = logprobs_confident if i % 2 == 0 else logprobs_uncertain
        complexity = random.uniform(1.0, 9.0)
        task_type = random.choice(["translation", "code_gen", "summarization", "classification"])

        start = time.perf_counter_ns()
        router.decide(
            logprobs=logprobs,
            complexity_score=complexity,
            task_type=task_type,
            text_length=random.randint(10, 500),
            capability_success_rate=random.uniform(0.3, 0.9),
        )
        elapsed_us = (time.perf_counter_ns() - start) / 1000.0
        latencies.append(elapsed_us)

    return compute_stats(latencies)


def bench_decide_threaded(router: TQBCRouter, n_threads: int = 4, n_per_thread: int = 2500) -> dict:
    """多线程测试 decide() 延迟（有锁竞争）"""
    logprobs = make_logprobs(10, confident=True)
    latencies = []
    lock = threading.Lock()

    def worker():
        local_lats = []
        for _ in range(n_per_thread):
            start = time.perf_counter_ns()
            router.decide(
                logprobs=logprobs,
                complexity_score=random.uniform(1.0, 9.0),
                task_type="translation",
            )
            elapsed_us = (time.perf_counter_ns() - start) / 1000.0
            local_lats.append(elapsed_us)
        with lock:
            latencies.extend(local_lats)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    start_total = time.perf_counter_ns()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total_ms = (time.perf_counter_ns() - start_total) / 1_000_000.0

    stats = compute_stats(latencies)
    stats["total_ms"] = round(total_ms, 1)
    stats["throughput_ops_per_sec"] = round(len(latencies) / (total_ms / 1000.0), 0)
    stats["n_threads"] = n_threads
    return stats


def bench_quantile_extract(n_iter: int = 10000) -> dict:
    """单独测试 quantile 特征提取延迟"""
    logprobs = make_logprobs(20, confident=True)
    latencies = []
    for _ in range(n_iter):
        start = time.perf_counter_ns()
        extract_quantile_features(logprobs)
        elapsed_us = (time.perf_counter_ns() - start) / 1000.0
        latencies.append(elapsed_us)
    return compute_stats(latencies)


def bench_thompson_select(n_iter: int = 10000) -> dict:
    """单独测试 Thompson Sampling 选择延迟"""
    ts = ThompsonSamplingRouter(n_features=11)
    features = [0.5] * 11
    latencies = []
    for _ in range(n_iter):
        start = time.perf_counter_ns()
        ts.select_arm(features)
        elapsed_us = (time.perf_counter_ns() - start) / 1000.0
        latencies.append(elapsed_us)
    return compute_stats(latencies)


def bench_calibration(n_iter: int = 10000) -> dict:
    """单独测试贝叶斯校准延迟"""
    with tempfile.TemporaryDirectory() as tmpdir:
        cal = BayesianConfidenceCalibrator(tmpdir)
        latencies = []
        for i in range(n_iter):
            conf = random.uniform(0.1, 0.9)
            start = time.perf_counter_ns()
            cal.calibrate(conf, task_type="translation")
            elapsed_us = (time.perf_counter_ns() - start) / 1000.0
            latencies.append(elapsed_us)
    return compute_stats(latencies)


import tempfile


def main():
    print(f"\n{'='*70}")
    print(f"实验 4: 路由决策延迟微基准")
    print(f"{'='*70}")

    with tempfile.TemporaryDirectory() as tmpdir:
        router = TQBCRouter(tmpdir)

        # 预热
        print("预热中...")
        for _ in range(100):
            router.decide(
                logprobs=make_logprobs(10),
                complexity_score=5.0,
                task_type="translation",
            )

        results = {}

        # 1. 串行 decide()
        print("\n[1/6] 串行 decide() (10000 次)...")
        serial_stats = bench_decide_serial(router, 10000)
        results["decide_serial"] = serial_stats
        print(f"  p50={serial_stats['p50_us']:.0f}μs  p95={serial_stats['p95_us']:.0f}μs  "
              f"p99={serial_stats['p99_us']:.0f}μs  mean={serial_stats['mean_us']:.0f}μs")

        # 2. 多线程 decide()
        print("\n[2/6] 多线程 decide() (4 线程 × 2500)...")
        threaded_stats = bench_decide_threaded(router, n_threads=4, n_per_thread=2500)
        results["decide_threaded"] = threaded_stats
        print(f"  p50={threaded_stats['p50_us']:.0f}μs  p95={threaded_stats['p95_us']:.0f}μs  "
              f"p99={threaded_stats['p99_us']:.0f}μs  throughput={threaded_stats['throughput_ops_per_sec']:.0f} ops/s")

        # 3. Quantile 特征提取
        print("\n[3/6] Quantile 特征提取 (10000 次)...")
        quantile_stats = bench_quantile_extract(10000)
        results["quantile_extract"] = quantile_stats
        print(f"  p50={quantile_stats['p50_us']:.0f}μs  p95={quantile_stats['p95_us']:.0f}μs")

        # 4. Thompson Sampling 选择
        print("\n[4/6] Thompson Sampling 选择 (10000 次)...")
        ts_stats = bench_thompson_select(10000)
        results["thompson_select"] = ts_stats
        print(f"  p50={ts_stats['p50_us']:.0f}μs  p95={ts_stats['p95_us']:.0f}μs")

        # 5. 贝叶斯校准
        print("\n[5/6] 贝叶斯校准 (10000 次)...")
        cal_stats = bench_calibration(10000)
        results["bayesian_calibration"] = cal_stats
        print(f"  p50={cal_stats['p50_us']:.0f}μs  p95={cal_stats['p95_us']:.0f}μs")

        # 6. 端到端延迟分解
        print("\n[6/6] 延迟分解...")
        results["latency_breakdown"] = {
            "quantile_extract_pct": round(quantile_stats["mean_us"] / serial_stats["mean_us"] * 100, 1),
            "thompson_select_pct": round(ts_stats["mean_us"] / serial_stats["mean_us"] * 100, 1),
            "bayesian_calibration_pct": round(cal_stats["mean_us"] / serial_stats["mean_us"] * 100, 1),
            "overhead_pct": round(
                max(0, serial_stats["mean_us"] - quantile_stats["mean_us"]
                    - ts_stats["mean_us"] - cal_stats["mean_us"]) / serial_stats["mean_us"] * 100, 1),
        }

    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_iterations": 10000,
        "results": results,
    }

    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "latency.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到 {output_path}")

    # 打印总结
    print(f"\n{'='*70}")
    print("延迟总结")
    print(f"{'='*70}")
    print(f"  端到端 decide(): {serial_stats['mean_us']:.0f}μs (p50), {serial_stats['p99_us']:.0f}μs (p99)")
    print(f"  吞吐量 (串行):   {1_000_000 / serial_stats['mean_us']:.0f} ops/sec")
    print(f"  吞吐量 (4线程):  {threaded_stats['throughput_ops_per_sec']:.0f} ops/sec")


if __name__ == "__main__":
    main()
