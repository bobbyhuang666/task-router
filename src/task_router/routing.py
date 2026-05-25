"""
路由决策 — A3M 多信号复杂度评估 + 任务类型检测
"""

import re
import hashlib
import time
from dataclasses import dataclass, field
from typing import Optional


# ─── 任务数据结构 ──────────────────────────────────────────────

@dataclass
class Task:
    """任务描述"""
    id: str = ""
    action: str = ""
    text: str = ""
    files: list[str] = field(default_factory=list)
    output: str = ""
    model_used: str = ""
    tokens_input: int = 0
    tokens_output: int = 0
    cost_saved: float = 0.0
    time_ms: int = 0
    route: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            raw = f"{self.action}{self.text[:100]}{time.time()}"
            self.id = hashlib.md5(raw.encode()).hexdigest()[:12]


# ─── 路由常量 ──────────────────────────────────────────────────

LOCAL_TASK_PATTERNS: list[str] = [
    "分类", "归类", "整理", "排序", "重命名", "移动文件", "复制",
    "提取", "摘录", "格式化", "转换", "翻译", "概括",
    "替换", "补全", "填充", "合并", "拼接", "分割", "拆分",
    "过滤", "去重", "统计", "计数",
    "列出", "列举", "查询", "搜索",
    "属于哪", "是什么类型", "判定", "检查", "校验", "验证",
    "标记", "打标签",
    "批量", "所有文件", "全部", "遍历",
]

MANDATORY_LOCAL_PATTERNS: list[str] = ["本地", "离线", "不上传"]

CLOUD_PATTERNS: list[str] = [
    "代码", "编程", "debug", "调试", "重构", "架构",
    "bug", "错误", "异常", "项目", "仓库", "git", "复杂", "高级",
    "分布式", "高并发", "微服务", "系统设计", "算法",
]

# 动词/操作词强度映射（正数=复杂，负数=简单）
VERB_INTENSITY: dict[str, float] = {
    "设计": 0.25, "架构": 0.25, "规划": 0.20, "策略": 0.20,
    "分析": 0.15, "推理": 0.25, "对比": 0.15, "比较": 0.15,
    "评价": 0.15, "优化": 0.15, "重构": 0.25, "调试": 0.20,
    "生成": 0.10, "创建": 0.10, "实现": 0.15, "开发": 0.15,
    "编写": 0.10, "预测": 0.20, "推荐": 0.15, "建议": 0.10,
    "解释": 0.10, "说明": 0.10, "描述": 0.05, "介绍": 0.05,
    "分类": -0.15, "归类": -0.15, "整理": -0.10, "列出": -0.15,
    "列举": -0.15, "提取": -0.10, "摘录": -0.10, "翻译": -0.05,
    "格式化": -0.15, "转换": -0.10, "概括": -0.05, "总结": -0.05,
    "排序": -0.15, "去重": -0.15, "过滤": -0.10, "统计": -0.05,
    "计数": -0.15, "检查": -0.05, "验证": -0.05, "打标签": -0.10,
    "标记": -0.10, "重命名": -0.10, "复制": -0.10, "移动": -0.05,
}

MULTI_STEP_CONNECTORS: list[str] = [
    "并且", "然后", "接着", "再", "再然后", "之后", "随后",
    "同时", "以及", "和", "与", "并",
    "先", "首先", "其次", "最后", "第一步", "第二步",
    "1.", "2.", "3.", "①", "②", "③",
]

HIGH_COMPLEXITY_DOMAINS: list[str] = [
    "金融", "法律", "医疗", "医药", "投资", "税务", "合同",
    "机器学习", "深度学习", "算法", "神经网络", "量化",
    "安全", "加密", "密码", "网络协议", "编译",
]


# ─── 复杂度评估 ──────────────────────────────────────────────

def estimate_complexity(task: Task, base_threshold: float = 3.0, weights=None) -> dict:
    """
    A3M 风格多信号复杂度评估（支持可学习权重）。

    参数:
        weights: A3MWeights 实例，为 None 时使用默认权重

    返回:
        {"route": "local"|"cloud", "score": float, "reason": str, "will_save": bool}
    """
    from task_router.weights import A3MWeights
    w = weights or A3MWeights()

    action_lower = task.action.lower()
    text_len = len(task.text or "")
    file_count = len(task.files)

    # 硬规则
    for p in MANDATORY_LOCAL_PATTERNS:
        if p in action_lower:
            return {"route": "local", "reason": f"匹配强制本地模式: {p}", "score": 0, "will_save": True}

    for p in CLOUD_PATTERNS:
        if p in action_lower:
            return {"route": "cloud", "reason": f"匹配需云端模式: {p}", "score": 10, "will_save": False}

    # 信号 1：动词强度
    verb_score = 0.0
    verb_signals: list[str] = []
    for verb, weight in VERB_INTENSITY.items():
        if verb in action_lower:
            verb_score += weight
            verb_signals.append(f"{verb}({weight:+.2f})")

    # 信号 2：多步检测
    multi_step_count = sum(1 for c in MULTI_STEP_CONNECTORS if c in action_lower)
    multi_step_penalty = min(multi_step_count, 3) * w.multi_step_weight

    # 信号 3：领域复杂度
    domain_score = sum(w.domain_weight for d in HIGH_COMPLEXITY_DOMAINS if d in action_lower)

    # 信号 4：文本长度
    if text_len > w.text_long_threshold:
        text_score = w.text_long_score
    elif text_len > w.text_mid_threshold:
        text_score = w.text_mid_score
    elif text_len > w.text_short_threshold:
        text_score = w.text_short_score
    else:
        text_score = 0
    if 0 < text_len < w.text_tiny_max_len:
        text_score = max(0, text_score + w.text_tiny_penalty)

    # 信号 5：文件数量
    if file_count > w.file_many_threshold:
        file_score = w.file_many_score
    elif file_count > w.file_some_threshold:
        file_score = w.file_some_score
    else:
        file_score = 0

    # 信号 6：本地模式匹配
    local_match = sum(1 for p in LOCAL_TASK_PATTERNS if p in action_lower)
    local_pattern_bonus = w.local_pattern_weight * min(local_match, w.local_pattern_max)

    # 信号 7：action 长度
    action_len = len(action_lower)
    if action_len > w.action_long_threshold:
        action_score = w.action_long_score
    elif action_len < w.action_short_threshold:
        action_score = w.action_short_score
    else:
        action_score = 0

    # 综合评分
    score = (
        verb_score * w.verb_multiplier
        + multi_step_penalty
        + domain_score
        + text_score
        + file_score
        + local_pattern_bonus
        + action_score
    )

    # 构建原因
    signal_parts: list[str] = []
    if verb_signals:
        signal_parts.append(f"动词: {''.join(verb_signals)}={verb_score:.2f}")
    if multi_step_count:
        signal_parts.append(f"多步: {multi_step_count}个连接词")
    if domain_score:
        signal_parts.append(f"领域: +{domain_score}")

    route = "local" if score <= base_threshold else "cloud"
    reason = f"评分 {score:.1f} {'≤' if route == 'local' else '>'} {base_threshold}"
    if signal_parts:
        reason += f" ({', '.join(signal_parts)})"

    return {
        "route": route,
        "score": round(score, 2),
        "reason": reason,
        "will_save": route == "local",
    }


def detect_task_type(action: str, templates: dict) -> str:
    """检测细粒度任务类型"""
    action_lower = action.lower()
    for ttype, tpl in templates.items():
        if any(d in action_lower for d in tpl.get("detect", [])):
            return ttype
    return ""


def decompose_complex_task(action: str, text: str) -> list[dict]:
    """将复杂任务拆解为多个简单子任务"""
    action_lower = action.lower()
    subtasks: list[dict] = []

    has_classify = any(w in action_lower for w in ["分类", "归类"])
    has_rename = any(w in action_lower for w in ["重命名", "改名"])
    has_sort = any(w in action_lower for w in ["排序", "排列"])
    has_dedup = any(w in action_lower for w in ["去重", "去重复"])
    has_count = any(w in action_lower for w in ["统计", "计数", "多少个"])

    if has_classify:
        ext_list = ["pdf", "jpg", "png", "py", "js", "css", "html", "csv", "xlsx", "docx", "txt", "gif"]
        has_extensions = any(f".{ext}" in (text or "").lower() for ext in ext_list)
        classify_type = "file_classify" if has_extensions else "general_classify"
        subtasks.append({"type": classify_type, "action": "按文件类型分类" if has_extensions else "按类别分组", "text": text})

    if has_rename and text:
        subtasks.append({"type": "rename_suggest", "action": "建议新文件名", "text": text})
    if has_sort and text:
        subtasks.append({"type": "sort_numbers", "action": "按数值排序", "text": text})
    if has_dedup and text:
        subtasks.append({"type": "dedup", "action": "去除重复项", "text": text})
    if has_count and text:
        subtasks.append({"type": "_count", "action": "统计数量", "text": text})

    return subtasks


def _make_subtask(action: str, text: str, depth: int) -> dict:
    """构建子任务（含类型推断）"""
    from task_router.prompts import PROMPT_TEMPLATES
    task_type = detect_task_type(action, PROMPT_TEMPLATES)
    return {"action": action, "text": text, "depth": depth, "type": task_type}


def _recursive_decompose(action: str, text: str, depth: int = 0, max_depth: int = 3) -> Optional[list[dict]]:
    """递归拆解复合任务（真正递归：3+ 步任务完整拆解）"""
    if depth >= max_depth:
        return None

    # 长连接词优先（避免 "再" 单字误切 "再次确认" 等词）
    connectors = ["再然后", "然后", "接着", "同时", "并且", "且", "并"]
    for conn in connectors:
        if conn in action:
            parts = action.split(conn, 1)
            if len(parts) == 2 and len(parts[0].strip()) > 2 and len(parts[1].strip()) > 2:
                left = parts[0].strip()
                right = parts[1].strip()
                result = []

                # 递归拆解左侧子任务
                left_sub = _recursive_decompose(left, text, depth + 1, max_depth)
                if left_sub:
                    result.extend(left_sub)
                else:
                    result.append(_make_subtask(left, text, depth))

                # 递归拆解右侧子任务
                right_sub = _recursive_decompose(right, text, depth + 1, max_depth)
                if right_sub:
                    result.extend(right_sub)
                else:
                    result.append(_make_subtask(right, text, depth))

                return result

    # 逗号分隔的多个动作
    if "，" in action or "," in action:
        parts = re.split(r"[，,]", action)
        if len(parts) >= 2 and all(len(p.strip()) > 2 for p in parts):
            result = []
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                sub = _recursive_decompose(p, text, depth + 1, max_depth)
                if sub:
                    result.extend(sub)
                else:
                    result.append(_make_subtask(p, text, depth))
            return result if result else None

    return None
