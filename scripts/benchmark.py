#!/usr/bin/env python3
"""
TaskRouter 基准测试 — 路由准确率 & 节省比例

3 个典型场景: 翻译、分类、代码生成
用模拟数据对比 纯云端 vs TaskRouter 混合路由
"""

import json

from task_router import (
    Task, run_task, estimate_complexity, detect_task_type, calc_cost,
    CONFIG, cache,
)
from prompts import PROMPT_TEMPLATES

# ─── 测试用例 ──────────────────────────────────────────────────

SCENARIOS = [
    {
        "name": "翻译任务",
        "category": "translation",
        "cases": [
            {"action": "翻译成中文", "text": "Hello, how are you today? I hope you are doing well."},
            {"action": "翻译这段英文", "text": "The quick brown fox jumps over the lazy dog."},
            {"action": "将以下内容翻译为中文", "text": "Artificial intelligence is transforming industries worldwide."},
            {"action": "帮我翻译", "text": "Please confirm receipt of this email at your earliest convenience."},
            {"action": "翻译", "text": "Machine learning models require large datasets for training."},
        ],
    },
    {
        "name": "分类任务",
        "category": "classification",
        "cases": [
            {"action": "分类", "text": "iPhone 15 Pro Max 256GB 原色钛金属"},
            {"action": "判断情感", "text": "这个产品太好用了，强烈推荐给大家！"},
            {"action": "分类以下文本", "text": "今日A股三大指数集体收涨，沪指涨0.5%"},
            {"action": "判断类别", "text": "患者主诉头痛三天，伴有恶心呕吐症状"},
            {"action": "情感分析", "text": "服务态度极差，等了一个小时都没人理"},
        ],
    },
    {
        "name": "代码生成任务",
        "category": "code_generation",
        "cases": [
            {"action": "写一个Python函数，实现快速排序算法，要求支持自定义比较函数", "text": ""},
            {"action": "用JavaScript写一个防抖函数，支持立即执行和取消功能", "text": ""},
            {"action": "实现一个LRU缓存，要求get和put操作都是O(1)时间复杂度", "text": ""},
            {"action": "写一个SQL查询，找出每个部门工资最高的员工", "text": ""},
            {"action": "用Go实现一个并发安全的计数器，支持原子递增和读取", "text": ""},
        ],
    },
]

# DeepSeek API 定价 (per 1M tokens)
CLOUD_INPUT_PRICE = 0.14   # $0.14 / 1M input tokens
CLOUD_OUTPUT_PRICE = 0.28  # $0.28 / 1M output tokens


def estimate_cloud_tokens(action: str, text: str) -> tuple[int, int]:
    """估算云端调用的 token 数量"""
    input_tokens = max(50, len(action + text) // 2)
    output_tokens = max(80, len(action))
    return input_tokens, output_tokens


def run_benchmark() -> dict:
    """运行基准测试"""
    results = {"scenarios": [], "summary": {}}
    total_cloud_cost = 0.0
    total_router_cost = 0.0
    total_local = 0
    total_cloud = 0
    total_cases = 0
    correct_routes = 0

    for scenario in SCENARIOS:
        scenario_result = {
            "name": scenario["name"],
            "category": scenario["category"],
            "cases": [],
            "local_count": 0,
            "cloud_count": 0,
            "cloud_cost": 0.0,
            "router_cost": 0.0,
        }

        for case in scenario["cases"]:
            action = case["action"]
            text = case.get("text", "")

            # 路由决策
            task = Task(action=action, text=text)
            decision = estimate_complexity(task, base_threshold=CONFIG.base_threshold)
            task_type = detect_task_type(action, PROMPT_TEMPLATES)
            suggested = decision["route"]

            # 模拟云端成本
            in_tok, out_tok = estimate_cloud_tokens(action, text)
            cloud_cost = calc_cost(in_tok, out_tok)

            # 执行任务
            task = run_task(task)
            router_cost = task.cost_saved if task.route == "local" else cloud_cost

            # 路由准确率判断
            is_correct = True
            if task_type in ("code_generation", "code_review") and suggested == "local":
                is_correct = False
            if task_type in ("translation", "classification", "sentiment") and suggested == "cloud":
                if decision["score"] < 5:  # 低分任务不应走云端
                    is_correct = False

            if is_correct:
                correct_routes += 1

            case_result = {
                "action": action[:50],
                "task_type": task_type,
                "score": round(decision["score"], 1),
                "route": task.route,
                "suggested": suggested,
                "model": task.model_used,
                "cloud_cost": round(cloud_cost, 6),
                "router_cost": round(router_cost, 6),
                "correct": is_correct,
            }
            scenario_result["cases"].append(case_result)

            if "local" in task.route or "cache" in task.route:
                scenario_result["local_count"] += 1
                total_local += 1
            else:
                scenario_result["cloud_count"] += 1
                total_cloud += 1

            scenario_result["cloud_cost"] += cloud_cost
            scenario_result["router_cost"] += router_cost
            total_cloud_cost += cloud_cost
            total_router_cost += router_cost
            total_cases += 1

        results["scenarios"].append(scenario_result)

    # 汇总
    savings_pct = ((total_cloud_cost - total_router_cost) / total_cloud_cost * 100) if total_cloud_cost > 0 else 0
    accuracy = (correct_routes / total_cases * 100) if total_cases > 0 else 0

    results["summary"] = {
        "total_cases": total_cases,
        "local_calls": total_local,
        "cloud_calls": total_cloud,
        "accuracy_pct": round(accuracy, 1),
        "cloud_cost_if_all_cloud": round(total_cloud_cost, 4),
        "actual_router_cost": round(total_router_cost, 4),
        "savings_pct": round(savings_pct, 1),
        "savings_amount": round(total_cloud_cost - total_router_cost, 4),
    }

    return results


def main():
    print("=" * 60)
    print("TaskRouter 基准测试")
    print("=" * 60)

    # 清除缓存确保公平测试
    cache._entries.clear()

    results = run_benchmark()
    s = results["summary"]

    for scenario in results["scenarios"]:
        print(f"\n--- {scenario['name']} ---")
        for c in scenario["cases"]:
            icon = "🟢" if "local" in c["route"] or "cache" in c["route"] else "🔵"
            print(f"  {icon} [{c['task_type']:15}] score={c['score']:4.1f} → {c['route']:15} | ${c['cloud_cost']:.4f} → ${c['router_cost']:.4f}")
        saved = scenario["cloud_cost"] - scenario["router_cost"]
        pct = (saved / scenario["cloud_cost"] * 100) if scenario["cloud_cost"] > 0 else 0
        print(f"  节省: ${saved:.4f} ({pct:.0f}%)")

    print(f"\n{'=' * 60}")
    print("汇总结果")
    print(f"{'=' * 60}")
    print(f"测试用例: {s['total_cases']} 个")
    print(f"路由准确率: {s['accuracy_pct']}%")
    print(f"本地调用: {s['local_calls']} 次 | 云端调用: {s['cloud_calls']} 次")
    print(f"全云端成本: ${s['cloud_cost_if_all_cloud']:.4f}")
    print(f"实际成本:   ${s['actual_router_cost']:.4f}")
    print(f"节省金额:   ${s['savings_amount']:.4f}")
    print(f"节省比例:   {s['savings_pct']}%")

    # 输出 JSON
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n详细结果已保存到 benchmark_results.json")


if __name__ == "__main__":
    main()
