"""
规则引擎 — 用 Python 规则替代模型调用，100% 准确
"""

import re


def rule_execute(task_type: str, text: str) -> str:
    """规则执行：对某些任务类型用 Python 规则替代模型"""
    items = [x.strip() for x in text.replace("\n", ",").split(",") if x.strip()]

    if task_type == "sort_numbers":
        def extract_num(s: str) -> float:
            nums = re.findall(r'[：:]\s*(\d+(?:\.\d+)?)', s)
            if nums:
                return float(nums[0])
            nums = re.findall(r'(\d+(?:\.\d+)?)', s)
            if nums:
                return float(nums[-1])
            return 0
        items.sort(key=extract_num, reverse=True)
        return "\n".join(items)

    if task_type == "sort_alpha":
        items.sort()
        return "\n".join(items)

    if task_type == "dedup":
        seen: set[str] = set()
        result: list[str] = []
        for item in items:
            if item.lower() not in seen:
                seen.add(item.lower())
                result.append(item)
        return "\n".join(result)

    if task_type == "_count":
        return f"共 {len(items)} 项"

    return ""
