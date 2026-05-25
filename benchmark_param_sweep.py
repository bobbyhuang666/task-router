#!/usr/bin/env python3
"""参数扫描：找出 qwen-tool 最优 Ollama 配置"""

import requests
import time
import subprocess
import json
import sys

TASKS = [
    ("translate", "Translate to Chinese: The quick brown fox jumps over the lazy dog.",
     lambda t: len(t) > 5 and ("快" in t or "brown" not in t)),
    ("json_extract", 'Extract fields from: "张三, 工号A001, 部门研发部". Output JSON with name, id, dept.',
     lambda t: "name" in t.lower() or "张三" in t),
    ("classify", 'Classify sentiment as positive/negative: "这部电影太棒了，强烈推荐！"',
     lambda t: "positive" in t.lower() or "积极" in t or "正面" in t),
    ("summarize", "Summarize in one sentence: Python is a high-level programming language known for its readability and versatility. It supports multiple paradigms and has a large ecosystem.",
     lambda t: "python" in t.lower() or "编程" in t or "语言" in t),
    ("math", "Calculate: 17 * 23 + 45 / 9. Show the answer only.",
     lambda t: "396" in t),
    ("format", "Format as markdown table: Apple $1.2, Banana $0.5, Cherry $2.0",
     lambda t: "|" in t and ("apple" in t.lower() or "1.2" in t)),
    ("keywords", "提取关键词: 人工智能正在改变医疗行业，深度学习在医学影像诊断中表现出色。",
     lambda t: any(k in t for k in ["人工智能", "深度学习", "医疗", "影像"])),
    ("code", "Write a Python function to reverse a string. One line only.",
     lambda t: "return" in t or "::-1" in t),
    ("numbers", "从文本中提取所有数字：订单金额128.5元，数量3件，折扣0.8",
     lambda t: "128" in t and "3" in t and "0.8" in t),
    ("bool", '判断："地球是平的"这个说法是否正确？回答正确或错误。',
     lambda t: "错误" in t or "不正确" in t or "wrong" in t.lower() or "false" in t.lower()),
]


def create_model(name, params, system_prompt=None):
    mf = f"FROM qwen3.5:0.8b\n"
    for k, v in params.items():
        mf += f"PARAMETER {k} {v}\n"
    if system_prompt:
        mf += f'SYSTEM "{system_prompt}"\n'
    with open("/tmp/sweep_modelfile", "w") as f:
        f.write(mf)
    r = subprocess.run(
        ["ollama", "create", name, "-f", "/tmp/sweep_modelfile"],
        capture_output=True, text=True, timeout=30,
    )
    return r.returncode == 0


def run_test(model_name):
    times = []
    passed = 0
    icons = []
    for _, prompt, check in TASKS:
        body = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {"num_predict": 200},
        }
        t0 = time.time()
        r = requests.post("http://localhost:11434/api/generate", json=body, timeout=60)
        elapsed = (time.time() - t0) * 1000
        data = r.json()
        text = data.get("response", "")
        ok = check(text)
        if ok:
            passed += 1
        times.append(elapsed)
        icons.append("V" if ok else "X")
    avg_ms = sum(times) / len(times)
    return passed, len(TASKS), avg_ms, "".join(icons)


# --- Parameter sweep ---

configs = [
    # (name, params, system_prompt)
    ("baseline", {"num_ctx": 2048, "temperature": 0.7, "top_p": 0.9, "repeat_penalty": 1.1}, None),
    ("precise", {"num_ctx": 2048, "temperature": 0.1, "top_p": 0.7, "repeat_penalty": 1.0}, None),
    ("precise+rp", {"num_ctx": 2048, "temperature": 0.1, "top_p": 0.7, "repeat_penalty": 1.1}, None),
    ("balanced", {"num_ctx": 2048, "temperature": 0.5, "top_p": 0.85, "repeat_penalty": 1.05}, None),
    ("lowtemp", {"num_ctx": 2048, "temperature": 0.3, "top_p": 0.8, "repeat_penalty": 1.1}, None),
    ("ctx4096", {"num_ctx": 4096, "temperature": 0.7, "top_p": 0.9, "repeat_penalty": 1.1}, None),
    ("ctx1024", {"num_ctx": 1024, "temperature": 0.7, "top_p": 0.9, "repeat_penalty": 1.1}, None),
    ("norepeat", {"num_ctx": 2048, "temperature": 0.7, "top_p": 0.9, "repeat_penalty": 1.0}, None),
    ("hirepeat", {"num_ctx": 2048, "temperature": 0.7, "top_p": 0.9, "repeat_penalty": 1.3}, None),
    ("topp95", {"num_ctx": 2048, "temperature": 0.7, "top_p": 0.95, "repeat_penalty": 1.1}, None),
    ("topp80", {"num_ctx": 2048, "temperature": 0.7, "top_p": 0.80, "repeat_penalty": 1.1}, None),
    # System prompt variants
    ("sys_concise", {"num_ctx": 2048, "temperature": 0.5, "top_p": 0.85, "repeat_penalty": 1.05},
     "Answer concisely. No explanation unless asked."),
    ("sys_direct", {"num_ctx": 2048, "temperature": 0.1, "top_p": 0.7, "repeat_penalty": 1.0},
     "Output only the requested result. No reasoning, no explanation."),
    ("sys_zh", {"num_ctx": 2048, "temperature": 0.5, "top_p": 0.85, "repeat_penalty": 1.05},
     "简洁高效地回答问题，不要多余解释。"),
]

print("=" * 65)
print("  qwen-tool 参数扫描 (10 tasks x {} configs)".format(len(configs)))
print("=" * 65)

results = []
for name, params, sys_prompt in configs:
    print(f"  {name:<16}", end="", flush=True)
    ok = create_model("qwen-tool-test", params, sys_prompt)
    if not ok:
        print("创建失败")
        continue
    passed, total, avg_ms, icons = run_test("qwen-tool-test")
    results.append((name, passed, total, avg_ms, icons, params, sys_prompt))
    print(f"  {passed}/{total}  {avg_ms:>7.0f}ms  {icons}")

# Sort: highest pass rate, then lowest latency
results.sort(key=lambda x: (-x[1], x[3]))

print()
print("=" * 65)
print("  排名")
print("=" * 65)
print(f"  {'排名':<4} {'配置':<16} {'通过':<6} {'延迟':<10} {'详情'}")
print("  " + "-" * 60)
for i, (name, passed, total, avg_ms, icons, params, sys_prompt) in enumerate(results):
    marker = " <-- 最优" if i == 0 else ""
    print(f"  {i+1:<4} {name:<16} {passed}/{total}   {avg_ms:>7.0f}ms  {icons}{marker}")

# Print best config details
best = results[0]
print()
print(f"  最优配置: {best[0]}")
print(f"  通过率: {best[1]}/{best[2]} ({best[1]/best[2]*100:.0f}%)")
print(f"  平均延迟: {best[3]:.0f}ms")
print(f"  参数: {json.dumps(best[5], indent=4)}")
if best[6]:
    print(f"  系统提示: {best[6]}")
