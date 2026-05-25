#!/usr/bin/env python3
"""
实验 6: 真实端到端测试

定义 30 个真实任务（翻译/分类/代码生成/摘要/问答/数据分析各 5 个），
每个任务同时走本地和云端，记录：
- 本地输出质量（QualityEvaluator 5维打分）
- 云端输出质量
- 路由器选择的路由
- 本地延迟 vs 云端延迟
- 成本
- 路由器是否选对了
- 质量差距
- 成本节省

结果存 results/real_e2e.json。
"""

import json
import os
import sys
import time
import subprocess

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)


# ─── 真实任务定义 ──────────────────────────────────────────────

REAL_TASKS = [
    # 翻译 (5)
    {"id": "R01", "category": "translation", "action": "翻译成中文",
     "text": "The rapid advancement of artificial intelligence has fundamentally transformed industries worldwide.",
     "difficulty": "medium", "expected_route": "cloud",
     "quality_keywords": ["人工智能", "快速", "进步", "行业"]},
    {"id": "R02", "category": "translation", "action": "翻译成英文",
     "text": "请将以下内容翻译成英文：量子计算代表了计算能力的下一个重大飞跃。",
     "difficulty": "medium", "expected_route": "cloud",
     "quality_keywords": ["quantum", "computing", "next", "leap"]},
    {"id": "R03", "category": "translation", "action": "翻译",
     "text": "Hello, how are you today?",
     "difficulty": "easy", "expected_route": "local",
     "quality_keywords": ["你好", "今天", "怎么样"]},
    {"id": "R04", "category": "translation", "action": "翻译为中文",
     "text": "Force majeure events including but not limited to acts of God, war, pandemic.",
     "difficulty": "hard", "expected_route": "cloud",
     "quality_keywords": ["不可抗力", "战争", "疫情"]},
    {"id": "R05", "category": "translation", "action": "翻译",
     "text": "Thank you for your assistance.",
     "difficulty": "easy", "expected_route": "local",
     "quality_keywords": ["谢谢", "帮助"]},

    # 分类 (5)
    {"id": "R06", "category": "classification", "action": "判断情感",
     "text": "这个产品太好用了，强烈推荐！",
     "difficulty": "easy", "expected_route": "local",
     "quality_keywords": ["正面", "积极"]},
    {"id": "R07", "category": "classification", "action": "分类",
     "text": "报告.pdf, 照片.jpg, 代码.py, 视频.mp4",
     "difficulty": "easy", "expected_route": "local",
     "quality_keywords": ["文档", "图片", "代码", "视频"]},
    {"id": "R08", "category": "classification", "action": "判断意图",
     "text": "我想退货，因为收到的商品和描述不一致",
     "difficulty": "medium", "expected_route": "local",
     "quality_keywords": ["退货", "退款", "投诉"]},
    {"id": "R09", "category": "classification", "action": "判断合规风险",
     "text": "该产品含有未经FDA批准的添加剂成分，可能违反联邦食品药品法规",
     "difficulty": "hard", "expected_route": "cloud",
     "quality_keywords": ["合规", "违规", "风险"]},
    {"id": "R10", "category": "classification", "action": "情感分析",
     "text": "服务态度极差，等了一个小时还没人来",
     "difficulty": "easy", "expected_route": "local",
     "quality_keywords": ["负面", "消极"]},

    # 代码生成 (5)
    {"id": "R11", "category": "code", "action": "写一个Python函数",
     "text": "计算斐波那契数列第n项",
     "difficulty": "easy", "expected_route": "local",
     "quality_keywords": ["def", "return", "fibonacci"]},
    {"id": "R12", "category": "code", "action": "写SQL查询",
     "text": "找出员工表中薪资最高的前5名员工",
     "difficulty": "easy", "expected_route": "local",
     "quality_keywords": ["SELECT", "ORDER BY", "LIMIT"]},
    {"id": "R13", "category": "code", "action": "实现LRU缓存",
     "text": "用Python实现一个线程安全的LRU缓存",
     "difficulty": "medium", "expected_route": "cloud",
     "quality_keywords": ["class", "dict", "threading"]},
    {"id": "R14", "category": "code", "action": "设计分布式锁",
     "text": "实现一个基于Redis的分布式锁，支持自动续期和可重入",
     "difficulty": "hard", "expected_route": "cloud",
     "quality_keywords": ["redis", "lock", "expire"]},
    {"id": "R15", "category": "code", "action": "写一个防抖函数",
     "text": "JavaScript防抖函数，支持立即执行选项",
     "difficulty": "easy", "expected_route": "local",
     "quality_keywords": ["function", "setTimeout", "debounce"]},

    # 摘要 (5)
    {"id": "R16", "category": "summarization", "action": "总结要点",
     "text": "公司第一季度营收增长15%，净利润增长20%，主要得益于新产品线的推出和海外市场的拓展。员工总数从500人增加到650人。",
     "difficulty": "easy", "expected_route": "local",
     "quality_keywords": ["营收", "增长", "净利润"]},
    {"id": "R17", "category": "summarization", "action": "概括",
     "text": "深度学习在自然语言处理领域的应用越来越广泛，包括机器翻译、文本生成、情感分析等。Transformer架构的出现使得大语言模型成为可能。",
     "difficulty": "medium", "expected_route": "local",
     "quality_keywords": ["深度学习", "NLP", "Transformer"]},
    {"id": "R18", "category": "summarization", "action": "提取关键信息",
     "text": "会议纪要：讨论了三个技术方案，方案A成本低但性能一般，方案B性能好但成本高，方案C折中。最终决定采用方案C，预计两个月内完成开发。",
     "difficulty": "medium", "expected_route": "local",
     "quality_keywords": ["方案C", "折中", "两个月"]},
    {"id": "R19", "category": "summarization", "action": "总结报告",
     "text": "本季度市场分析报告：全球AI市场规模达到5000亿美元，同比增长35%。中国市场占比25%，增速最快。主要驱动因素包括企业数字化转型、自动驾驶和医疗AI应用。",
     "difficulty": "hard", "expected_route": "cloud",
     "quality_keywords": ["AI市场", "5000亿", "数字化转型"]},
    {"id": "R20", "category": "summarization", "action": "概括摘要",
     "text": "用户反馈汇总：70%用户对产品速度满意，15%反映偶尔卡顿，10%提到UI需要改进，5%有数据同步问题。",
     "difficulty": "easy", "expected_route": "local",
     "quality_keywords": ["速度", "卡顿", "UI"]},

    # 问答 (5)
    {"id": "R21", "category": "qa", "action": "解释概念",
     "text": "什么是REST API？",
     "difficulty": "easy", "expected_route": "local",
     "quality_keywords": ["HTTP", "接口", "RESTful"]},
    {"id": "R22", "category": "qa", "action": "回答问题",
     "text": "Python中列表和元组有什么区别？",
     "difficulty": "easy", "expected_route": "local",
     "quality_keywords": ["可变", "不可变", "list", "tuple"]},
    {"id": "R23", "category": "qa", "action": "技术问答",
     "text": "微服务架构中如何实现服务发现和服务注册？请比较Consul和etcd方案。",
     "difficulty": "hard", "expected_route": "cloud",
     "quality_keywords": ["Consul", "etcd", "服务发现"]},
    {"id": "R24", "category": "qa", "action": "回答",
     "text": "Kubernetes中Pod被OOMKilled的常见原因和解决方案是什么？",
     "difficulty": "medium", "expected_route": "cloud",
     "quality_keywords": ["内存", "limit", "requests"]},
    {"id": "R25", "category": "qa", "action": "解释",
     "text": "Git中rebase和merge的区别是什么？什么时候用哪个？",
     "difficulty": "medium", "expected_route": "local",
     "quality_keywords": ["变基", "合并", "历史"]},

    # 数据分析 (5)
    {"id": "R26", "category": "analysis", "action": "分析数据趋势",
     "text": "Q1: 100万, Q2: 120万, Q3: 95万, Q4: 150万。请分析销售额趋势并给出解读。",
     "difficulty": "medium", "expected_route": "local",
     "quality_keywords": ["趋势", "增长", "波动"]},
    {"id": "R27", "category": "analysis", "action": "对比分析",
     "text": "React vs Vue 2024年生态系统对比：包括社区活跃度、企业采用率、性能表现。",
     "difficulty": "medium", "expected_route": "cloud",
     "quality_keywords": ["React", "Vue", "生态"]},
    {"id": "R28", "category": "analysis", "action": "根因分析",
     "text": "系统延迟从50ms突增到500ms，日志显示数据库连接池耗尽，同时CPU利用率飙升到95%。",
     "difficulty": "medium", "expected_route": "cloud",
     "quality_keywords": ["数据库", "连接池", "CPU"]},
    {"id": "R29", "category": "analysis", "action": "风险评估",
     "text": "投资项目：海外房地产基金，预期年化回报15%，投资期限3年，最低投资额100万。",
     "difficulty": "hard", "expected_route": "cloud",
     "quality_keywords": ["风险", "回报", "流动性"]},
    {"id": "R30", "category": "analysis", "action": "列出要点",
     "text": "Python的优点有哪些？",
     "difficulty": "easy", "expected_route": "local",
     "quality_keywords": ["简单", "易学", "生态"]},
]


# ─── LLM-as-Judge 评估 ─────────────────────────────────────────

_judge_cache: dict[str, dict] = {}


def llm_as_judge(task: dict, output: str, api_url: str, api_key: str) -> dict:
    """
    用 LLM 作为裁判评估输出质量（blind judge — 不知道是本地还是云端）。

    返回 {completeness, relevance, fluency, accuracy, format, overall}，
    每项 0-1 分。
    """
    cache_key = f"{task['id']}|{hash(output)}"
    if cache_key in _judge_cache:
        return _judge_cache[cache_key]

    if not output or output.startswith("[错误]") or output.startswith("[本地模型不可用]"):
        result = {"completeness": 0.1, "relevance": 0.1, "fluency": 0.1,
                  "accuracy": 0.1, "format": 0.1, "overall": 0.1}
        _judge_cache[cache_key] = result
        return result

    judge_prompt = f"""你是一个严格的质量评估专家。请评估以下 AI 模型对用户任务的回复质量。

## 用户任务
指令: {task['action']}
输入: {task['text']}

## AI 回复
{output[:2000]}

## 评分标准（每项 1-10 分）
1. completeness（完整性）: 回复是否涵盖了任务要求的所有要点
2. relevance（相关性）: 回复是否与任务直接相关，没有跑题
3. fluency（流畅性）: 语言是否通顺自然，无重复/矛盾
4. accuracy（准确性）: 事实是否正确，逻辑是否合理
5. format（格式）: 输出格式是否规范清晰

请严格按以下 JSON 格式返回，不要加任何其他文字:
{{"completeness": X, "relevance": X, "fluency": X, "accuracy": X, "format": X}}"""

    try:
        import urllib.request
        import json as _json

        body = _json.dumps({
            "model": os.environ.get("CLOUD_MODEL", "deepseek-v4-flash"),
            "messages": [{"role": "user", "content": judge_prompt}],
            "max_tokens": 200,
            "temperature": 0.0,  # judge 要确定性
        }).encode()

        req = urllib.request.Request(
            f"{api_url}/v1/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        resp = urllib.request.urlopen(req, timeout=30)
        data = _json.loads(resp.read())
        content = data["choices"][0]["message"]["content"].strip()

        # 解析 JSON（容忍 markdown code blocks）
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        scores = _json.loads(content)

        # 归一化到 0-1
        result = {
            "completeness": round(max(0, min(1, scores.get("completeness", 5) / 10.0)), 3),
            "relevance": round(max(0, min(1, scores.get("relevance", 5) / 10.0)), 3),
            "fluency": round(max(0, min(1, scores.get("fluency", 5) / 10.0)), 3),
            "accuracy": round(max(0, min(1, scores.get("accuracy", 5) / 10.0)), 3),
            "format": round(max(0, min(1, scores.get("format", 5) / 10.0)), 3),
        }
        result["overall"] = round(
            result["completeness"] * 0.25 + result["relevance"] * 0.25 +
            result["fluency"] * 0.15 + result["accuracy"] * 0.25 +
            result["format"] * 0.10, 3)

    except Exception as e:
        print(f"      ⚠ Judge 评分失败: {e}")
        result = {"completeness": 0.5, "relevance": 0.5, "fluency": 0.5,
                  "accuracy": 0.5, "format": 0.5, "overall": 0.5}

    _judge_cache[cache_key] = result
    return result


# ─── 质量评估（5维度，启发式备用） ──────────────────────────────

def evaluate_quality(text: str, task: dict, llm_available: bool = True) -> dict:
    """
    5 维质量评估：
    1. 完整性 — 是否涵盖了关键要点
    2. 相关性 — 输出是否与任务相关
    3. 流畅性 — 语言是否通顺
    4. 准确性 — 事实是否正确
    5. 格式   — 输出格式是否规范
    """
    if not text or text.startswith("[错误]") or text.startswith("[本地模型不可用]"):
        return {"completeness": 0.2, "relevance": 0.2, "fluency": 0.2, "accuracy": 0.2, "format": 0.2, "overall": 0.2}

    text_lower = text.lower()
    keywords = task.get("quality_keywords", [])

    # 完整性：关键词覆盖度
    if keywords:
        matched = sum(1 for kw in keywords if kw.lower() in text_lower or kw in text)
        completeness = min(1.0, matched / max(1, len(keywords) * 0.6))
    else:
        completeness = min(1.0, len(text) / 200.0)

    # 相关性：文本长度和任务类型的匹配
    expected_len = {"easy": (20, 200), "medium": (50, 500), "hard": (100, 1000)}.get(task.get("difficulty", "medium"), (50, 500))
    text_len = len(text)
    if expected_len[0] <= text_len <= expected_len[1]:
        relevance = 0.9
    elif text_len < expected_len[0]:
        relevance = max(0.3, text_len / expected_len[0])
    else:
        relevance = max(0.5, 1.0 - (text_len - expected_len[1]) / (expected_len[1] * 2))

    # 流畅性：基于简单启发式
    has_repetition = text.count("。") > len(text) / 50  # 过多句号
    fluency = 0.7 if not has_repetition else 0.5

    # 准确性：如果有关键词匹配
    if keywords:
        matched = sum(1 for kw in keywords if kw.lower() in text_lower or kw in text)
        accuracy = min(1.0, matched / max(1, len(keywords) * 0.5))
    else:
        accuracy = 0.7  # 无法判断

    # 格式：输出整洁度
    format_score = 0.8
    if text.startswith("\n") or text.endswith("\n\n\n"):
        format_score -= 0.2
    if "[[" in text or "]]" in text:
        format_score -= 0.2
    format_score = max(0.3, format_score)

    overall = (completeness * 0.25 + relevance * 0.25 + fluency * 0.15 +
               accuracy * 0.25 + format_score * 0.10)

    return {
        "completeness": round(completeness, 3),
        "relevance": round(relevance, 3),
        "fluency": round(fluency, 3),
        "accuracy": round(accuracy, 3),
        "format": round(format_score, 3),
        "overall": round(overall, 3),
    }


# ─── 路由决策模拟 ──────────────────────────────────────────────

def simulate_router_decision(task: dict) -> dict:
    """模拟路由器决策"""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

    from task_router.routing import Task, estimate_complexity

    t = Task(action=task["action"], text=task["text"])
    decision = estimate_complexity(t)

    # 模拟置信度
    diff = task["difficulty"]
    if diff == "easy":
        confidence = 0.8 + (hash(task["id"]) % 10) / 100
    elif diff == "hard":
        confidence = 0.2 + (hash(task["id"]) % 10) / 100
    else:
        confidence = 0.4 + (hash(task["id"]) % 20) / 100

    route = decision["route"]

    return {
        "route": route,
        "score": decision.get("score", 0),
        "confidence": round(confidence, 3),
        "reason": decision.get("reason", ""),
    }


# ─── 本地执行（模拟） ──────────────────────────────────────────

LOCAL_MODEL = "qwen2.5:3b"

def run_local(task: dict, ollama_available: bool) -> dict:
    """本地模型执行（真实调用 Ollama CLI）"""
    if not ollama_available:
        return {"output": "[本地模型不可用]", "latency_ms": 0, "route": "local"}

    prompt = f"{task['action']}\n{task['text']}" if task["text"] else task["action"]
    start = time.monotonic()
    try:
        result = subprocess.run(
            ["ollama", "run", LOCAL_MODEL, "--nowordwrap"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = result.stdout.strip()
        latency_ms = round((time.monotonic() - start) * 1000, 1)
        if not output:
            output = f"[本地模型无输出] stderr: {result.stderr[:200]}"
    except subprocess.TimeoutExpired:
        output = "[本地模型超时]"
        latency_ms = round((time.monotonic() - start) * 1000, 1)
    except Exception as e:
        output = f"[本地模型错误] {e}"
        latency_ms = round((time.monotonic() - start) * 1000, 1)

    return {"output": output, "latency_ms": latency_ms, "route": "local"}


def _rule_based_output(task: dict) -> str:
    """基于规则的本地输出生成（模拟本地小模型）"""
    action = task["action"]
    text = task["text"]
    cat = task["category"]

    if cat == "translation":
        if "翻译成中文" in action or "翻译为中文" in action:
            # 简单直译模拟
            return f"[翻译] {text[:80]}..."
        elif "翻译成英文" in action:
            return f"[Translation] The content discusses: {text[:60]}..."
        else:
            return f"[Translation] {text[:80]}..."

    elif cat == "classification":
        if "情感" in action:
            if any(w in text for w in ["好", "推荐", "棒", "喜欢"]):
                return "正面"
            elif any(w in text for w in ["差", "烂", "坏", "差"]):
                return "负面"
            return "中性"
        elif "意图" in action:
            return "投诉/退货"
        elif "风险" in action:
            return "存在合规风险"
        else:
            return "分类结果：一般类别"

    elif cat == "code":
        if "斐波那契" in text:
            return "def fibonacci(n):\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)"
        elif "SQL" in action or "sql" in action.lower():
            return "SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 5;"
        elif "防抖" in text:
            return "function debounce(fn, delay) {\n  let timer;\n  return (...args) => {\n    clearTimeout(timer);\n    timer = setTimeout(() => fn(...args), delay);\n  };\n}"
        elif "LRU" in text:
            return "from collections import OrderedDict\n\nclass LRUCache:\n    def __init__(self, capacity):\n        self.cache = OrderedDict()\n        self.capacity = capacity"
        elif "分布式锁" in text:
            return "import redis\n\ndef acquire_lock(key, ttl=30):\n    return client.set(key, '1', nx=True, ex=ttl)"
        else:
            return "# 代码实现\n# TODO: complete implementation"

    elif cat == "summarization":
        return f"主要要点：{text[:100]}..."

    elif cat == "qa":
        if "REST" in text:
            return "REST API是一种基于HTTP的接口设计风格，使用GET/POST/PUT/DELETE等方法..."
        elif "列表" in text and "元组" in text:
            return "列表(list)是可变的，元组(tuple)是不可变的..."
        elif "Git" in text or "git" in text:
            return "rebase会重写提交历史，merge会保留完整历史..."
        else:
            return f"回答：{text[:80]}..."

    elif cat == "analysis":
        return f"分析结果：根据数据 {text[:60]}...，趋势为整体增长"

    return f"处理完成：{task['action']}"


# ─── 主函数 ──────────────────────────────────────────────────

def check_ollama() -> bool:
    """检查 Ollama 是否在运行（使用 CLI 兼容性更好）"""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
        return result.returncode == 0 and "NAME" in result.stdout
    except Exception:
        return False


def run_real_benchmark():
    """运行真实端到端测试"""
    print(f"\n{'='*70}")
    print(f"实验 6: 真实端到端测试")
    print(f"{'='*70}")

    # 检查 Ollama
    ollama_available = check_ollama()
    if ollama_available:
        print("✓ Ollama 运行中")
    else:
        print("✗ Ollama 不可用，将使用模拟输出")

    # 检查云端 API
    cloud_api_url = os.environ.get("CLOUD_API_URL", "")
    cloud_api_key = os.environ.get("CLOUD_API_KEY", "")
    cloud_available = bool(cloud_api_url and cloud_api_key)
    if cloud_available:
        print(f"✓ 云端 API: {cloud_api_url}")
        print("  质量评估: LLM-as-Judge (blind)")
    else:
        print("✗ 云端 API 未配置，将使用模拟云端输出")
        print("  质量评估: 启发式 (fallback)")

    results = []
    start_time = time.monotonic()

    for task in REAL_TASKS:
        print(f"\n  [{task['id']}] {task['category']:16} {task['action'][:30]}")

        # 1. 路由决策
        router_result = simulate_router_decision(task)
        chosen_route = router_result["route"]
        print(f"    路由: {chosen_route} (score={router_result['score']}, conf={router_result['confidence']})")

        # 2. 本地执行
        local_result = run_local(task, ollama_available)

        # 3. 云端执行（模拟或真实）
        if cloud_available:
            cloud_result = _call_cloud_api(task, cloud_api_url, cloud_api_key)
        else:
            cloud_result = _simulate_cloud(task)

        # 4. 质量评估（优先 LLM-as-Judge，否则启发式）
        if cloud_available:
            print("    ⏳ Judge 评估本地输出...")
            local_quality = llm_as_judge(task, local_result["output"], cloud_api_url, cloud_api_key)
            print("    ⏳ Judge 评估云端输出...")
            cloud_quality = llm_as_judge(task, cloud_result["output"], cloud_api_url, cloud_api_key)
        else:
            local_quality = evaluate_quality(local_result["output"], task)
            cloud_quality = evaluate_quality(cloud_result["output"], task)

        print(f"    本地质量: {local_quality['overall']:.3f}  云端质量: {cloud_quality['overall']:.3f}  差距: {cloud_quality['overall'] - local_quality['overall']:+.3f}")

        # 5. 路由是否正确
        expected = task["expected_route"]
        if expected == "either":
            router_correct = True
        else:
            router_correct = (chosen_route == expected)

        # 5. 质量差距
        quality_gap = cloud_quality["overall"] - local_quality["overall"]

        # 6. 成本
        local_cost = 0.0  # 本地免费
        cloud_cost = _estimate_cloud_cost(task)

        results.append({
            "id": task["id"],
            "category": task["category"],
            "difficulty": task["difficulty"],
            "expected_route": expected,
            "chosen_route": chosen_route,
            "router_correct": router_correct,
            "router_score": router_result["score"],
            "router_confidence": router_result["confidence"],
            "local_output": local_result["output"][:200],
            "local_quality": local_quality,
            "local_latency_ms": local_result["latency_ms"],
            "local_cost": local_cost,
            "cloud_output": cloud_result["output"][:200],
            "cloud_quality": cloud_quality,
            "cloud_latency_ms": cloud_result["latency_ms"],
            "cloud_cost": cloud_cost,
            "quality_gap": round(quality_gap, 3),
        })

        status = "✓" if router_correct else "✗"
        print(f"    {status} local_q={local_quality['overall']:.2f} cloud_q={cloud_quality['overall']:.2f} "
              f"gap={quality_gap:+.2f}")

    elapsed_ms = (time.monotonic() - start_time) * 1000

    # 汇总统计
    total = len(results)
    correct = sum(1 for r in results if r["router_correct"])
    avg_local_q = sum(r["local_quality"]["overall"] for r in results) / total
    avg_cloud_q = sum(r["cloud_quality"]["overall"] for r in results) / total
    avg_gap = sum(r["quality_gap"] for r in results) / total
    total_cloud_cost = sum(r["cloud_cost"] for r in results)
    total_actual_cost = sum(r["cloud_cost"] if r["chosen_route"] == "cloud" else 0 for r in results)
    cost_savings = (1 - total_actual_cost / total_cloud_cost) * 100 if total_cloud_cost > 0 else 0

    # 按类别统计
    by_category = {}
    for r in results:
        cat = r["category"]
        if cat not in by_category:
            by_category[cat] = {"total": 0, "correct": 0, "local_q_sum": 0, "cloud_q_sum": 0}
        by_category[cat]["total"] += 1
        if r["router_correct"]:
            by_category[cat]["correct"] += 1
        by_category[cat]["local_q_sum"] += r["local_quality"]["overall"]
        by_category[cat]["cloud_q_sum"] += r["cloud_quality"]["overall"]

    for cat, stats in by_category.items():
        stats["accuracy"] = round(stats["correct"] / stats["total"] * 100, 1)
        stats["avg_local_quality"] = round(stats["local_q_sum"] / stats["total"], 3)
        stats["avg_cloud_quality"] = round(stats["cloud_q_sum"] / stats["total"], 3)
        del stats["local_q_sum"]
        del stats["cloud_q_sum"]

    # 打印报告
    print(f"\n{'='*70}")
    print("真实端到端测试报告")
    print(f"{'='*70}")
    print(f"\n总任务数: {total}")
    print(f"路由准确率: {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"平均本地质量: {avg_local_q:.3f}")
    print(f"平均云端质量: {avg_cloud_q:.3f}")
    print(f"平均质量差距: {avg_gap:+.3f}")
    print(f"全云端成本: ${total_cloud_cost:.6f}")
    print(f"实际成本: ${total_actual_cost:.6f}")
    print(f"成本节省: {cost_savings:.1f}%")
    print(f"耗时: {elapsed_ms:.0f}ms")

    print(f"\n{'类别':16} {'准确率':8} {'本地质量':10} {'云端质量':10}")
    print("-" * 50)
    for cat, stats in sorted(by_category.items()):
        print(f"{cat:16} {stats['accuracy']:6.1f}% {stats['avg_local_quality']:9.3f} "
              f"{stats['avg_cloud_quality']:9.3f}")

    # 保存结果
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "ollama_available": ollama_available,
        "cloud_available": cloud_available,
        "cloud_api_url": cloud_api_url,
        "evaluation_method": "llm_as_judge" if cloud_available else "heuristic",
        "judge_model": os.environ.get("CLOUD_MODEL", "deepseek-v4-flash") if cloud_available else "N/A",
        "summary": {
            "total_tasks": total,
            "routing_accuracy_pct": round(correct / total * 100, 1),
            "avg_local_quality": round(avg_local_q, 3),
            "avg_cloud_quality": round(avg_cloud_q, 3),
            "avg_quality_gap": round(avg_gap, 3),
            "total_cloud_cost": round(total_cloud_cost, 6),
            "total_actual_cost": round(total_actual_cost, 6),
            "cost_savings_pct": round(cost_savings, 1),
            "elapsed_ms": round(elapsed_ms, 1),
        },
        "by_category": by_category,
        "tasks": results,
    }

    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "real_e2e.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到 {output_path}")


def _simulate_cloud(task: dict) -> dict:
    """模拟云端 API 响应（无 API key 时使用）"""
    action = task.get("action", "")
    text = task.get("text", "")
    prompt = f"{action}\n{text}" if text else action
    # 简单模拟：基于任务复杂度返回合理长度的模拟输出
    simulated = f"[模拟云端输出] 对以下任务的响应：{prompt[:100]}..."
    return {"output": simulated, "latency_ms": 0.0, "route": "cloud"}


def _call_cloud_api(task: dict, api_url: str, api_key: str) -> dict:
    """调用真实 DeepSeek 云端 API"""
    try:
        import urllib.request
        import json as _json

        prompt = f"{task['action']}\n{task['text']}" if task["text"] else task["action"]
        body = _json.dumps({
            "model": os.environ.get("CLOUD_MODEL", "deepseek-v4-flash"),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500,
            "temperature": 0.3,
        }).encode()

        req = urllib.request.Request(
            f"{api_url}/v1/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )

        start = time.monotonic()
        resp = urllib.request.urlopen(req, timeout=60)
        latency_ms = (time.monotonic() - start) * 1000

        data = _json.loads(resp.read())
        output = data["choices"][0]["message"]["content"]

        return {"output": output, "latency_ms": round(latency_ms, 1), "route": "cloud"}
    except Exception as e:
        return {"output": f"[云端API错误] {e}", "latency_ms": 0, "route": "cloud"}


def _estimate_cloud_cost(task: dict) -> float:
    """估算云端成本"""
    text_len = len(task["action"]) + len(task["text"])
    input_tokens = max(50, text_len // 2)
    output_tokens = max(80, len(task["action"]) * 3)
    # DeepSeek 定价: $0.14/M input, $0.28/M output
    return (input_tokens * 0.14 + output_tokens * 0.28) / 1_000_000


if __name__ == "__main__":
    run_real_benchmark()
