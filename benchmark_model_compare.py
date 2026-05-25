#!/usr/bin/env python3
"""
模型对比基准测试 — qwen-tool vs qwen3.5:0.8b

评估维度：
1. 响应延迟（含首次加载）
2. Token 效率（输入/输出 token 数）
3. logprobs 支持与置信度
4. 实际输出质量（规则验证）
"""

import os
import sys
import time
import json

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
sys.path.insert(0, SCRIPTS_DIR)

from config import get_config, set_config, RouterConfig
from models import call_ollama
from confidence import extract_confidence


# ─── 测试任务 ──────────────────────────────────────────────

TASKS = [
    {
        "name": "翻译(英→中)",
        "prompt": "Translate to Chinese: The quick brown fox jumps over the lazy dog.",
        "check": lambda t: len(t) > 5,
    },
    {
        "name": "JSON提取",
        "prompt": 'Extract fields from: "张三, 工号A001, 部门研发部". Output JSON with name, id, dept.',
        "check": lambda t: "name" in t.lower() or "张三" in t,
    },
    {
        "name": "分类",
        "prompt": 'Classify sentiment as positive/negative: "这部电影太棒了，强烈推荐！"',
        "check": lambda t: "positive" in t.lower() or "积极" in t or "正面" in t,
    },
    {
        "name": "摘要",
        "prompt": "Summarize in one sentence: Python is a high-level programming language known for its readability and versatility. It supports multiple paradigms and has a large ecosystem.",
        "check": lambda t: "python" in t.lower() or "编程" in t or "语言" in t,
    },
    {
        "name": "数学",
        "prompt": "Calculate: 17 * 23 + 45 / 9. Show the answer only.",
        "check": lambda t: "396" in t or "395" in t,
    },
    {
        "name": "格式化",
        "prompt": 'Format as markdown table: Apple $1.2, Banana $0.5, Cherry $2.0',
        "check": lambda t: "|" in t and ("apple" in t.lower() or "苹果" in t),
    },
    {
        "name": "关键词",
        "prompt": "提取关键词: 人工智能正在改变医疗行业，深度学习在医学影像诊断中表现出色。",
        "check": lambda t: any(k in t for k in ["人工智能", "深度学习", "医疗", "影像"]),
    },
    {
        "name": "代码生成",
        "prompt": "Write a Python function to reverse a string. One line only.",
        "check": lambda t: "return" in t or "::-1" in t or "reverse" in t.lower(),
    },
]


def run_benchmark(model_name: str, warmup: bool = True) -> dict:
    """对指定模型运行所有测试任务"""
    print(f"\n{'='*60}")
    print(f"  模型: {model_name}")
    print(f"{'='*60}")

    # 预热（第一次加载模型较慢）
    if warmup:
        print("  预热中...")
        try:
            call_ollama("hi", model=model_name, max_tokens=5)
        except Exception as e:
            print(f"  预热失败: {e}")
            return {"error": str(e)}

    results = []
    total_time = 0
    total_in = 0
    total_out = 0
    pass_count = 0

    for i, task in enumerate(TASKS):
        print(f"\n  [{i+1}/{len(TASKS)}] {task['name']}...", end=" ", flush=True)
        try:
            r = call_ollama(task["prompt"], model=model_name, with_logprobs=True, max_tokens=256)
            text = r["text"]
            elapsed = r["time_ms"]
            tok_in = r["tokens_input"]
            tok_out = r["tokens_output"]
            logprobs = r.get("logprobs", [])

            # 提取置信度
            conf_data = extract_confidence(logprobs) if logprobs else {"confidence": 0.0, "margin": 0.0}

            # 质量检查
            passed = task["check"](text)
            if passed:
                pass_count += 1

            total_time += elapsed
            total_in += tok_in
            total_out += tok_out

            status = "✓" if passed else "✗"
            print(f"{status} {elapsed}ms | {tok_in}+{tok_out} tok | conf={conf_data['confidence']:.3f}")
            if not passed:
                print(f"    输出: {text[:100]}...")

            results.append({
                "task": task["name"],
                "passed": passed,
                "time_ms": elapsed,
                "tokens_in": tok_in,
                "tokens_out": tok_out,
                "confidence": conf_data["confidence"],
                "margin": conf_data.get("margin", 0),
                "logprobs_count": len(logprobs),
                "output_preview": text[:200],
            })

        except Exception as e:
            print(f"✗ 错误: {e}")
            results.append({"task": task["name"], "passed": False, "error": str(e)})

    summary = {
        "model": model_name,
        "tasks_total": len(TASKS),
        "tasks_passed": pass_count,
        "pass_rate": pass_count / len(TASKS) if TASKS else 0,
        "total_time_ms": total_time,
        "avg_time_ms": total_time / len(results) if results else 0,
        "total_tokens_in": total_in,
        "total_tokens_out": total_out,
        "avg_confidence": sum(r.get("confidence", 0) for r in results) / len(results) if results else 0,
        "logprobs_supported": any(r.get("logprobs_count", 0) > 0 for r in results),
        "details": results,
    }
    return summary


def print_comparison(old: dict, new: dict):
    """对比两个模型的结果"""
    print(f"\n{'='*60}")
    print(f"  对比结果")
    print(f"{'='*60}")

    def delta(old_v, new_v, lower_better=False):
        d = new_v - old_v
        if lower_better:
            symbol = "↓" if d < 0 else "↑"
            good = d < 0
        else:
            symbol = "↑" if d > 0 else "↓"
            good = d > 0
        return f"{new_v:.1f} ({symbol} {abs(d):.1f}, {'✓' if good else '✗'})"

    print(f"\n  {'指标':<20} {'qwen-tool':<18} {'qwen3.5:0.8b':<22} {'变化'}")
    print(f"  {'-'*75}")
    print(f"  {'通过率':<20} {old['pass_rate']:.0%}{'':<14} {new['pass_rate']:.0%}{'':<14} {new['pass_rate']-old['pass_rate']:+.0%}")
    print(f"  {'平均延迟(ms)':<20} {old['avg_time_ms']:<18.0f} {delta(old['avg_time_ms'], new['avg_time_ms'], lower_better=True)}")
    print(f"  {'总token入':<20} {old['total_tokens_in']:<18d} {delta(old['total_tokens_in'], new['total_tokens_in'], lower_better=True)}")
    print(f"  {'总token出':<20} {old['total_tokens_out']:<18d} {delta(old['total_tokens_out'], new['total_tokens_out'], lower_better=True)}")
    print(f"  {'平均置信度':<20} {old['avg_confidence']:<18.4f} {delta(old['avg_confidence'], new['avg_confidence'], lower_better=False)}")
    print(f"  {'logprobs支持':<20} {str(old['logprobs_supported']):<18} {str(new['logprobs_supported'])}")

    # 逐任务对比
    print(f"\n  逐任务对比:")
    print(f"  {'任务':<14} {'旧模型':<12} {'新模型':<12} {'延迟差(ms)'}")
    print(f"  {'-'*55}")
    for old_d, new_d in zip(old["details"], new["details"]):
        old_s = "✓" if old_d.get("passed") else "✗"
        new_s = "✓" if new_d.get("passed") else "✗"
        time_diff = new_d.get("time_ms", 0) - old_d.get("time_ms", 0)
        print(f"  {old_d['task']:<14} {old_s:<12} {new_s:<12} {time_diff:+d}")


def main():
    print("=" * 60)
    print("  模型性能对比测试")
    print("  qwen-tool vs qwen3.5:0.8b")
    print("=" * 60)

    # 测试旧模型
    old_result = run_benchmark("qwen-tool", warmup=True)

    # 测试新模型
    new_result = run_benchmark("qwen3.5:0.8b", warmup=True)

    # 对比
    if "error" not in old_result and "error" not in new_result:
        print_comparison(old_result, new_result)

    # 保存结果
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_model_compare.json")
    with open(output_path, "w") as f:
        json.dump({"old": old_result, "new": new_result}, f, indent=2, ensure_ascii=False)
    print(f"\n  结果已保存: {output_path}")


if __name__ == "__main__":
    main()
