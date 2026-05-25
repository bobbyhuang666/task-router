#!/usr/bin/env python3
"""
补充实验 3: 不同本地模型对比

分别用 qwen2.5:1.5b 和 qwen2.5:3b 运行 30 个真实任务，记录：
- 每个任务的输出质量（启发式 + 可选 LLM judge）
- 延迟
- 路由准确率
- 成本

结果存 results/model_comparison.json。
"""

import json
import os
import subprocess
import sys
import time

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from benchmark_real import REAL_TASKS, check_ollama, run_local, evaluate_quality, simulate_router_decision, _call_cloud_api

MODELS_TO_TEST = [
    "qwen2.5:1.5b",
    "qwen2.5:3b",
]


def run_model_benchmark():
    print(f"\n{'='*70}")
    print(f"补充实验 3: 不同本地模型对比")
    print(f"{'='*70}")

    ollama_ok = check_ollama()
    if not ollama_ok:
        print("错误: Ollama 未运行")
        return

    # 检查可用模型
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    available = result.stdout
    for model in MODELS_TO_TEST:
        if model in available:
            print(f"  ✓ {model}")
        else:
            print(f"  ✗ {model} (不可用，跳过)")

    models_to_run = [m for m in MODELS_TO_TEST if m in available]
    if not models_to_run:
        print("没有可用模型")
        return

    # 收集云端输出作为对比基准
    api_url = os.environ.get("CLOUD_API_URL", "")
    api_key = os.environ.get("CLOUD_API_KEY", "")
    cloud_available = bool(api_url and api_key)

    all_model_results = {}

    for model_name in models_to_run:
        print(f"\n{'='*50}")
        print(f"测试模型: {model_name}")
        print(f"{'='*50}")

        # 需要修改 benchmark_real 中的 LOCAL_MODEL
        import benchmark_real
        benchmark_real.LOCAL_MODEL = model_name

        model_results = []
        total_local_ms = 0

        for task in REAL_TASKS:
            print(f"  [{task['id']}] {task['category']}: {task['action'][:30]}...", end=" ", flush=True)

            # 本地执行
            local_result = run_local(task, True)
            local_quality = evaluate_quality(local_result["output"], task)
            total_local_ms += local_result["latency_ms"]

            # 路由决策
            route_info = simulate_router_decision(task)

            # 云端（只收集一次，复用）
            cloud_quality = None
            if cloud_available and task["id"] not in all_model_results.get("_cloud_cache", {}):
                cloud_result = _call_cloud_api(task, api_url, api_key)
                cloud_quality = evaluate_quality(cloud_result["output"], task)
                cloud_latency = cloud_result["latency_ms"]
                if "_cloud_cache" not in all_model_results:
                    all_model_results["_cloud_cache"] = {}
                    all_model_results["_cloud_latency"] = {}
                all_model_results["_cloud_cache"][task["id"]] = cloud_quality
                all_model_results["_cloud_latency"][task["id"]] = cloud_latency
                time.sleep(0.3)

            cached_cloud_q = all_model_results.get("_cloud_cache", {}).get(task["id"], {"overall": 0.95})
            cached_cloud_lat = all_model_results.get("_cloud_latency", {}).get(task["id"], 500)

            entry = {
                "id": task["id"],
                "category": task["category"],
                "difficulty": task["difficulty"],
                "expected_route": task["expected_route"],
                "chosen_route": route_info["route"],
                "router_correct": route_info.get("correct", True) if task["expected_route"] == "either" else (route_info["route"] == task["expected_route"]),
                "local_output_len": len(local_result["output"]),
                "local_quality": local_quality,
                "local_latency_ms": local_result["latency_ms"],
                "cloud_quality": cached_cloud_q,
                "cloud_latency_ms": cached_cloud_lat,
                "quality_gap": round(local_quality["overall"] - cached_cloud_q["overall"], 3),
            }
            model_results.append(entry)
            print(f"q={local_quality['overall']:.2f} lat={local_result['latency_ms']:.0f}ms")

        # 汇总
        n = len(model_results)
        avg_quality = sum(r["local_quality"]["overall"] for r in model_results) / n
        avg_latency = total_local_ms / n
        avg_cloud_quality = sum(r["cloud_quality"]["overall"] for r in model_results) / n
        correct = sum(1 for r in model_results if r["router_correct"])

        by_cat = {}
        for r in model_results:
            cat = r["category"]
            if cat not in by_cat:
                by_cat[cat] = {"q_sum": 0, "lat_sum": 0, "n": 0}
            by_cat[cat]["q_sum"] += r["local_quality"]["overall"]
            by_cat[cat]["lat_sum"] += r["local_latency_ms"]
            by_cat[cat]["n"] += 1

        for cat in by_cat:
            s = by_cat[cat]
            s["avg_quality"] = round(s["q_sum"] / s["n"], 3)
            s["avg_latency_ms"] = round(s["lat_sum"] / s["n"], 1)
            del s["q_sum"]
            del s["lat_sum"]

        summary = {
            "model": model_name,
            "avg_local_quality": round(avg_quality, 3),
            "avg_cloud_quality": round(avg_cloud_quality, 3),
            "avg_latency_ms": round(avg_latency, 1),
            "routing_accuracy_pct": round(correct / n * 100, 1),
            "total_tasks": n,
            "by_category": by_cat,
        }

        print(f"\n  汇总: quality={avg_quality:.3f} latency={avg_latency:.0f}ms "
              f"accuracy={correct}/{n} ({correct/n*100:.0f}%)")
        print(f"  vs cloud: local={avg_quality:.3f} cloud={avg_cloud_quality:.3f} "
              f"gap={avg_quality - avg_cloud_quality:+.3f}")

        all_model_results[model_name] = {
            "summary": summary,
            "tasks": model_results,
        }

    # 对比表
    print(f"\n{'='*70}")
    print(f"模型对比汇总")
    print(f"{'='*70}")
    print(f"\n{'模型':20} {'质量分':8} {'延迟(ms)':10} {'准确率':8} {'vs 云端质量'}")
    print("-" * 60)
    for model_name in models_to_run:
        s = all_model_results[model_name]["summary"]
        gap = s["avg_local_quality"] - s["avg_cloud_quality"]
        print(f"{model_name:20} {s['avg_local_quality']:7.3f} {s['avg_latency_ms']:9.0f} "
              f"{s['routing_accuracy_pct']:6.1f}% {gap:+8.3f}")

    # 保存
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "models_tested": models_to_run,
        "n_tasks": len(REAL_TASKS),
    }
    for model_name in models_to_run:
        output[model_name] = all_model_results[model_name]

    # 清理缓存字段
    output.pop("_cloud_cache", None)
    output.pop("_cloud_latency", None)

    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "model_comparison.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到 {out_path}")


if __name__ == "__main__":
    run_model_benchmark()
