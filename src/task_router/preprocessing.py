"""
文本预处理与后处理 — 纯函数，零依赖
"""



def preprocess_text(text: str, max_chars: int = 800) -> str:
    """预处理输入文本：规范化换行、逗号列表转行、截断。

    参数:
        text: 输入文本
        max_chars: 最大字符数

    返回:
        预处理后的文本
    """
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if "," in text and "\n" not in text:
        items = [x.strip() for x in text.split(",") if x.strip()]
        if len(items) > 3:
            text = "\n".join(items)
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n... (截断，共 {len(text)} 字符)"
    return text


def postprocess_output(output: str, task_type: str = "") -> str:
    """后处理模型输出：移除前缀、特殊任务处理。

    参数:
        output: 模型原始输出
        task_type: 任务类型

    返回:
        清理后的输出
    """
    if not output:
        return ""
    output = output.strip()
    # 移除模型可能添加的前缀
    for prefix in ["输出：", "结果：", "答案：", "Output:", "Result:", "Answer:"]:
        if output.startswith(prefix):
            output = output[len(prefix):].strip()
    # 情感分析特殊处理
    if task_type == "sentiment":
        if "正面" in output and "负面" not in output:
            return "正面"
        elif "负面" in output and "正面" not in output:
            return "负面"
    return output
