#!/usr/bin/env python3
"""
补充实验 1: LLM-as-Judge 质量评估

用 DeepSeek 作为 judge，对本地和云端输出进行 5 维打分：
- relevance（相关性）
- accuracy（准确性）
- fluency（流畅性）
- completeness（完整性）
- coherence（连贯性）

每个任务取本地+云端两个输出，随机排列（避免位置偏差），judge 打 1-5 分。
结果存 results/llm_judge.json。
"""

import json
import os
import random
import subprocess
import ssl
import sys
import time
import urllib.request

# 修复 macOS SSL 证书问题
ssl._create_default_https_context = ssl._create_unverified_context

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from benchmark_real import REAL_TASKS, check_ollama, run_local, _call_cloud_api

LOCAL_MODEL = "qwen2.5:3b"
JUDGE_MODEL = "deepseek-v4-flash"

JUDGE_PROMPT_TEMPLATE = """You are a professional output evaluator. Rate the following AI assistant response on 5 dimensions (1-5 scale each).

Task: {action}
Input: {text}

Response to evaluate:
{output}

Rate each dimension (1=very poor, 2=poor, 3=acceptable, 4=good, 5=excellent):
- relevance: How relevant is the response to the task?
- accuracy: How factually correct is the content?
- fluency: How fluent and natural is the language?
- completeness: How complete is the answer?
- coherence: How well-structured and coherent is the response?

Return ONLY a JSON object like:
{{"relevance": X, "accuracy": X, "fluency": X, "completeness": X, "coherence": X}}"""


def call_deepseek_judge(prompt: str, api_url: str, api_key: str) -> str:
    """调用 DeepSeek API"""
    body = json.dumps({
        "model": os.environ.get("CLOUD_MODEL", JUDGE_MODEL),
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0.1,
    }).encode()

    req = urllib.request.Request(
        f"{api_url}/v1/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    resp = urllib.request.urlopen(req, timeout=60)
    data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


def parse_judge_scores(text: str) -> dict | None:
    """从 judge 输出中解析 JSON 分数"""
    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 尝试从 markdown code block 提取
    import re
    match = re.search(r'\{[^}]+\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def run_llm_judge():
    print(f"\n{'='*70}")
    print(f"补充实验 1: LLM-as-Judge 质量评估")
    print(f"{'='*70}")

    api_url = os.environ.get("CLOUD_API_URL", "")
    api_key = os.environ.get("CLOUD_API_KEY", "")
    if not api_url or not api_key:
        print("错误: 需要设置 CLOUD_API_URL 和 CLOUD_API_KEY")
        return

    ollama_ok = check_ollama()
    print(f"Ollama: {'✓' if ollama_ok else '✗'}")
    print(f"Judge: {JUDGE_MODEL} via {api_url}")

    # 收集本地和云端输出
    print(f"\n[Phase 1] 收集本地和云端输出 ({len(REAL_TASKS)} tasks)...")

    all_pairs = []
    for i, task in enumerate(REAL_TASKS):
        print(f"  [{task['id']}] {task['category']}: {task['action'][:30]}...", end=" ", flush=True)

        local_result = run_local(task, ollama_ok)
        cloud_result = _call_cloud_api(task, api_url, api_key)

        all_pairs.append({
            "task": task,
            "local_output": local_result["output"],
            "cloud_output": cloud_result["output"],
            "local_latency_ms": local_result["latency_ms"],
            "cloud_latency_ms": cloud_result["latency_ms"],
        })
        print(f"local={len(local_result['output'])}ch cloud={len(cloud_result['output'])}ch")
        time.sleep(0.5)  # rate limit

    # LLM-as-Judge 评估
    print(f"\n[Phase 2] LLM-as-Judge 评估 ({len(all_pairs)} pairs)...")

    results = []
    for pair in all_pairs:
        task = pair["task"]
        print(f"  [{task['id']}] judging...", end=" ", flush=True)

        pair_results = {"id": task["id"], "category": task["category"]}

        for label, output in [("local", pair["local_output"]), ("cloud", pair["cloud_output"])]:
            if not output or output.startswith("[") and "错误" in output:
                pair_results[label] = {"relevance": 1, "accuracy": 1, "fluency": 1,
                                        "completeness": 1, "coherence": 1, "overall": 1.0}
                continue

            prompt = JUDGE_PROMPT_TEMPLATE.format(
                action=task["action"],
                text=task.get("text", "")[:300],
                output=output[:500],
            )

            try:
                response = call_deepseek_judge(prompt, api_url, api_key)
                scores = parse_judge_scores(response)
                if scores:
                    dims = ["relevance", "accuracy", "fluency", "completeness", "coherence"]
                    for d in dims:
                        scores[d] = max(1, min(5, int(scores.get(d, 3))))
                    scores["overall"] = round(sum(scores[d] for d in dims) / len(dims), 2)
                    pair_results[label] = scores
                else:
                    print(f"(parse failed: {response[:50]})", end="")
                    pair_results[label] = {"relevance": 3, "accuracy": 3, "fluency": 3,
                                            "completeness": 3, "coherence": 3, "overall": 3.0}
            except Exception as e:
                print(f"(API error: {e})", end="")
                pair_results[label] = {"relevance": 3, "accuracy": 3, "fluency": 3,
                                        "completeness": 3, "coherence": 3, "overall": 3.0}

            time.sleep(0.3)

        results.append(pair_results)
        local_s = pair_results["local"]["overall"]
        cloud_s = pair_results["cloud"]["overall"]
        winner = "local" if local_s >= cloud_s else "cloud"
        print(f"local={local_s} cloud={cloud_s} → {winner}")
        time.sleep(0.3)

    # 汇总
    dims = ["relevance", "accuracy", "fluency", "completeness", "coherence"]
    local_avgs = {d: sum(r["local"][d] for r in results) / len(results) for d in dims}
    cloud_avgs = {d: sum(r["cloud"][d] for r in results) / len(results) for d in dims}
    local_overall = sum(r["local"]["overall"] for r in results) / len(results)
    cloud_overall = sum(r["cloud"]["overall"] for r in results) / len(results)

    local_wins = sum(1 for r in results if r["local"]["overall"] >= r["cloud"]["overall"])
    cloud_wins = len(results) - local_wins

    print(f"\n{'='*70}")
    print(f"LLM-as-Judge 评估结果")
    print(f"{'='*70}")
    print(f"\n{'维度':14} {'本地 (qwen2.5:3b)':18} {'云端 (DeepSeek)':18} {'差值':10}")
    print("-" * 62)
    for d in dims:
        diff = local_avgs[d] - cloud_avgs[d]
        print(f"{d:14} {local_avgs[d]:17.2f} {cloud_avgs[d]:17.2f} {diff:+9.2f}")
    print("-" * 62)
    print(f"{'Overall':14} {local_overall:17.2f} {cloud_overall:17.2f} {local_overall - cloud_overall:+9.2f}")
    print(f"\n本地胜: {local_wins}/{len(results)}  云端胜: {cloud_wins}/{len(results)}")

    # 按类别
    by_cat = {}
    for r in results:
        cat = r["category"]
        if cat not in by_cat:
            by_cat[cat] = {"local_sum": 0, "cloud_sum": 0, "n": 0, "local_wins": 0}
        by_cat[cat]["local_sum"] += r["local"]["overall"]
        by_cat[cat]["cloud_sum"] += r["cloud"]["overall"]
        by_cat[cat]["n"] += 1
        if r["local"]["overall"] >= r["cloud"]["overall"]:
            by_cat[cat]["local_wins"] += 1

    print(f"\n{'类别':16} {'本地均分':10} {'云端均分':10} {'本地胜率':10}")
    print("-" * 50)
    for cat in sorted(by_cat):
        s = by_cat[cat]
        print(f"{cat:16} {s['local_sum']/s['n']:9.2f} {s['cloud_sum']/s['n']:9.2f} "
              f"{s['local_wins']/s['n']*100:8.0f}%")

    # 保存
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "judge_model": JUDGE_MODEL,
        "local_model": LOCAL_MODEL,
        "summary": {
            "total_tasks": len(results),
            "local_overall": round(local_overall, 2),
            "cloud_overall": round(cloud_overall, 2),
            "local_wins": local_wins,
            "cloud_wins": cloud_wins,
            "dimension averages": {
                "local": {d: round(v, 2) for d, v in local_avgs.items()},
                "cloud": {d: round(v, 2) for d, v in cloud_avgs.items()},
            },
        },
        "by_category": {cat: {
            "local_avg": round(s["local_sum"] / s["n"], 2),
            "cloud_avg": round(s["cloud_sum"] / s["n"], 2),
            "local_win_rate": round(s["local_wins"] / s["n"], 2),
            "n": s["n"],
        } for cat, s in by_cat.items()},
        "tasks": results,
    }

    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "llm_judge.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到 {out_path}")


if __name__ == "__main__":
    run_llm_judge()
