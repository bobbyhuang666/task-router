"""
输出验证 — 检查本地模型输出质量，决定是否需要云端降级
"""

from typing import Any


FAILURE_SIGNALS: list[str] = [
    "抱歉", "对不起", "无法", "不能", "不懂", "不明白", "不知道",
    "作为AI", "作为语言模型", "作为一个AI",
    "我无法", "我不能", "我不确定", "我不清楚",
    "超出", "不在我的", "没有足够信息",
    "error", "Error", "ERROR",
    "undefined", "null", "None",
]

MIN_OUTPUT_LENGTH: dict[str, int] = {
    "general_classify": 10,
    "file_classify": 20,
    "sentiment": 2,
    "translate_en2zh": 2,
    "translate_zh2en": 2,
    "extract_keywords": 3,
    "extract_info": 3,
    "summarize_short": 10,
    "format_json": 5,
    "sort_numbers": 3,
    "sort_alpha": 3,
    "dedup": 3,
    "clean_data": 5,
    "rename_suggest": 10,
    "tag": 2,
    "qa_short": 5,
}


def validate_local_output(output: str, task_type: str = "") -> dict[str, Any]:
    """
    验证本地模型输出质量。

    返回:
        {"valid": bool, "reason": str, "signals": list[str]}
    """
    if not output or not output.strip():
        return {"valid": False, "reason": "空输出", "signals": ["empty"]}

    signals: list[str] = []

    # 1. 最小长度检查
    min_len = MIN_OUTPUT_LENGTH.get(task_type, 3)
    if len(output.strip()) < min_len:
        signals.append(f"输出过短({len(output.strip())}<{min_len})")

    # 2. 失败信号词检测
    for signal in FAILURE_SIGNALS:
        if signal in output:
            signals.append(f"含失败信号词: {signal[:10]}")
            break

    # 3. 任务特定检查
    if task_type == "sentiment":
        if output not in ("正面", "负面"):
            signals.append(f"情感分析输出应为'正面'或'负面'，实际: {output[:10]}")
    elif task_type == "tag":
        valid_tags = {"技术", "生活", "工作", "学习", "娱乐", "其他"}
        if output not in valid_tags:
            signals.append(f"标签超出可选范围: {output[:10]}")

    # 4. 输出质量问题 — 支持中文字符级重复检测
    # 英文：单词级重复
    words = output.split()
    if len(words) >= 3:
        repeat_count = sum(1 for i in range(len(words)-2)
                          if words[i] == words[i+1] == words[i+2])
        if repeat_count > 0:
            signals.append(f"输出含重复内容({repeat_count}处)")

    # 中文：字符级重复（连续相同字符）
    if len(output) >= 6:
        char_repeat = sum(1 for i in range(len(output)-2)
                         if output[i] == output[i+1] == output[i+2] and output[i].strip())
        if char_repeat > 0:
            signals.append(f"中文字符重复({char_repeat}处)")

    # 填充词检测：阈值按输出长度动态调整（避免长文本误判）
    filler_threshold = max(3, len(output) // 100)
    filler_count = sum(output.count(w) for w in ["嗯", "这个", "那个", "就是", "然后", "其实"])
    if filler_count > filler_threshold:
        signals.append(f"填充词过多({filler_count}个)")

    is_valid = len(signals) == 0
    reason = "通过验证" if is_valid else "; ".join(signals[:3])
    return {"valid": is_valid, "reason": reason, "signals": signals}
