"""
数据隐私层 — PII 检测与脱敏

在发送到模型之前自动检测和匿名化敏感信息：
- 手机号码（中国大陆）
- 身份证号码
- 邮箱地址
- 银行卡号
- IP 地址
- 姓名（可选，需配置）

使用方式：
    from task_router.privacy import PrivacyFilter
    filter = PrivacyFilter()
    safe_text = filter.anonymize("我的手机号是 13812345678")
    # → "我的手机号是 [PHONE_0]"
    original = filter.deanonymize(safe_text)
    # → "我的手机号是 13812345678"
"""

import re
import json
import os
from dataclasses import dataclass, field
from typing import Optional


# ─── PII 检测模式 ───────────────────────────────────────────────

PII_PATTERNS = {
    "phone": {
        "pattern": r'(?<!\d)1[3-9]\d{9}(?!\d)',
        "label": "[手机号]",
        "priority": 1,
    },
    "id_card": {
        "pattern": r'(?<!\d)[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)',
        "label": "[身份证]",
        "priority": 1,
    },
    "email": {
        "pattern": r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}',
        "label": "[邮箱]",
        "priority": 2,
    },
    "bank_card": {
        "pattern": r'(?<!\d)(?:6[0-9]{15,18}|4[0-9]{12,15}|5[1-5][0-9]{14}|3[47][0-9]{13})(?!\d)',
        "label": "[银行卡]",
        "priority": 1,
    },
    "ip_address": {
        "pattern": r'(?<!\d)(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)(?!\d)',
        "label": "[IP地址]",
        "priority": 3,
    },
    "passport": {
        "pattern": r'(?<![A-Z0-9])[A-Z][0-9]{8}(?![A-Z0-9])',
        "label": "[护照]",
        "priority": 2,
    },
}


@dataclass
class PIIMatch:
    """PII 匹配结果"""
    pii_type: str           # phone, id_card, email, etc.
    original: str           # 原始值
    start: int              # 起始位置
    end: int                # 结束位置
    placeholder: str        # 替换后的占位符


@dataclass
class AnonymizationResult:
    """脱敏结果"""
    text: str                           # 脱敏后的文本
    matches: list[PIIMatch] = field(default_factory=list)
    original_length: int = 0
    anonymized_count: int = 0


class PrivacyFilter:
    """
    隐私过滤器。

    使用方法:
        filter = PrivacyFilter()

        # 脱敏
        result = filter.anonymize("请联系 13812345678 或 test@example.com")
        print(result.text)
        # → "请联系 [PHONE_0] 或 [EMAIL_1]"

        # 还原
        original = filter.deanonymize(result.text)
        # → "请联系 13812345678 或 test@example.com"

        # 检测（不脱敏）
        matches = filter.detect("我的身份证号是 110101199001011234")
    """

    # 映射表最大条目数（防止内存无界增长）
    MAX_MAP_ENTRIES = 10000

    def __init__(self, enabled_types: list[str] = None, custom_patterns: dict = None):
        """
        参数:
            enabled_types: 启用的 PII 类型列表，None 表示全部启用
            custom_patterns: 自定义正则模式
        """
        self.enabled_types = enabled_types
        self.patterns = dict(PII_PATTERNS)
        if custom_patterns:
            self.patterns.update(custom_patterns)

        # 编译正则
        self._compiled = {}
        for pii_type, config in self.patterns.items():
            if enabled_types and pii_type not in enabled_types:
                continue
            try:
                self._compiled[pii_type] = re.compile(config["pattern"])
            except re.error:
                pass

        # 脱敏映射（用于还原），使用有序字典支持 LRU 淘汰
        self._placeholder_map: dict[str, str] = {}
        self._reverse_map: dict[str, str] = {}

    def detect(self, text: str) -> list[PIIMatch]:
        """
        检测文本中的 PII。

        返回:
            PII 匹配列表
        """
        matches = []
        for pii_type, regex in self._compiled.items():
            for match in regex.finditer(text):
                matches.append(PIIMatch(
                    pii_type=pii_type,
                    original=match.group(),
                    start=match.start(),
                    end=match.end(),
                    placeholder="",
                ))

        # 按位置排序，处理重叠
        matches.sort(key=lambda m: (m.start, -(m.end - m.start)))
        return self._resolve_overlaps(matches)

    def anonymize(self, text: str) -> AnonymizationResult:
        """
        脱敏文本。

        返回:
            AnonymizationResult 包含脱敏文本和映射关系
        """
        matches = self.detect(text)
        if not matches:
            return AnonymizationResult(
                text=text,
                original_length=len(text),
                anonymized_count=0,
            )

        # 为每个匹配生成占位符
        counters: dict[str, int] = {}
        for m in matches:
            if m.pii_type not in counters:
                counters[m.pii_type] = 0
            label = self.patterns[m.pii_type]["label"]
            m.placeholder = f"{label}_{counters[m.pii_type]}"
            counters[m.pii_type] += 1

        # 从后往前替换（避免位置偏移）
        result_text = text
        for m in reversed(matches):
            result_text = result_text[:m.start] + m.placeholder + result_text[m.end:]

            # 记录映射
            self._placeholder_map[m.placeholder] = m.original
            self._reverse_map[m.original] = m.placeholder

        # 防止映射表无限增长：超出上限时清除最旧的一半
        if len(self._placeholder_map) > self.MAX_MAP_ENTRIES:
            self._evict_old_entries()

        return AnonymizationResult(
            text=result_text,
            matches=matches,
            original_length=len(text),
            anonymized_count=len(matches),
        )

    def deanonymize(self, text: str) -> str:
        """
        还原脱敏文本。

        参数:
            text: 脱敏后的文本

        返回:
            还原后的文本
        """
        result = text
        for placeholder, original in self._placeholder_map.items():
            result = result.replace(placeholder, original)
        return result

    def _resolve_overlaps(self, matches: list[PIIMatch]) -> list[PIIMatch]:
        """处理重叠的 PII 匹配（保留优先级更高的）"""
        if not matches:
            return matches

        resolved = [matches[0]]
        for m in matches[1:]:
            prev = resolved[-1]
            if m.start < prev.end:
                # 重叠：保留更长的匹配
                if (m.end - m.start) > (prev.end - prev.start):
                    resolved[-1] = m
            else:
                resolved.append(m)
        return resolved

    def _evict_old_entries(self) -> None:
        """清除最旧的一半映射条目，防止内存无界增长。

        Python 3.7+ dict 保持插入顺序，
        所以 popitem(last=False) 移除最旧的条目。
        """
        target = self.MAX_MAP_ENTRIES // 2
        while len(self._placeholder_map) > target:
            try:
                placeholder, original = self._placeholder_map.popitem(last=False)
                self._reverse_map.pop(original, None)
            except KeyError:
                break

    def get_stats(self) -> dict:
        """获取脱敏统计"""
        return {
            "mapped_count": len(self._placeholder_map),
            "enabled_types": list(self._compiled.keys()),
        }

    def clear(self):
        """清除映射（释放内存）"""
        self._placeholder_map.clear()
        self._reverse_map.clear()


# ─── 便捷函数 ──────────────────────────────────────────────────

def quick_anonymize(text: str) -> str:
    """快速脱敏（只返回脱敏文本）"""
    return get_privacy_filter().anonymize(text).text

def quick_deanonymize(text: str) -> str:
    """快速还原"""
    return get_privacy_filter().deanonymize(text)


# ─── 隐私配置 ──────────────────────────────────────────────────

@dataclass
class PrivacyConfig:
    """隐私配置"""
    # 是否启用脱敏
    enabled: bool = True
    # 启用的 PII 类型
    enabled_types: list[str] = field(default_factory=lambda: [
        "phone", "id_card", "email", "bank_card"
    ])
    # 是否对本地模型也脱敏（默认只对云端脱敏）
    anonymize_local: bool = False
    # 是否在日志中记录脱敏事件
    log_anonymization: bool = True

    @classmethod
    def from_file(cls, path: str) -> "PrivacyConfig":
        """从配置文件加载"""
        if not os.path.exists(path):
            return cls()
        try:
            import dataclasses as _dc
            valid_fields = {f.name for f in _dc.fields(cls)}
            with open(path) as f:
                data = json.load(f)
            return cls(**{k: v for k, v in data.items() if k in valid_fields})
        except (json.JSONDecodeError, TypeError):
            return cls()

    def save(self, path: str):
        """保存配置"""
        import dataclasses
        with open(path, "w") as f:
            json.dump(dataclasses.asdict(self), f, ensure_ascii=False, indent=2)


# ─── 模块级单例（线程安全）──────────────────────────────────────

import threading

_privacy_filter_instance: Optional[PrivacyFilter] = None
_privacy_filter_lock = threading.Lock()


def get_privacy_filter() -> Optional[PrivacyFilter]:
    """获取全局 PrivacyFilter 单例（线程安全）"""
    global _privacy_filter_instance
    if _privacy_filter_instance is not None:
        return _privacy_filter_instance
    with _privacy_filter_lock:
        if _privacy_filter_instance is not None:
            return _privacy_filter_instance
        _privacy_filter_instance = PrivacyFilter()
    return _privacy_filter_instance


if __name__ == "__main__":
    # 测试
    filter = PrivacyFilter()

    test_cases = [
        "请联系 13812345678 或 15900001111",
        "我的邮箱是 test@example.com，身份证号 110101199001011234",
        "银行卡号：6222021234567890123",
        "服务器 IP: 192.168.1.100",
        "没有敏感信息的普通文本",
        "张三的手机号是13812345678，邮箱是zhangsan@company.com",
    ]

    for text in test_cases:
        result = filter.anonymize(text)
        print(f"原文: {text}")
        print(f"脱敏: {result.text}")
        print(f"还原: {filter.deanonymize(result.text)}")
        print(f"匹配: {len(result.matches)} 个 PII")
        print("---")
