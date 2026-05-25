"""
自适应 Prompt 压缩器 — 置信度驱动的 Token 预算优化

核心创新：根据模型置信度动态调整 prompt 压缩强度。
高置信度 → 激进压缩（模型已"理解"任务，不需要太多上下文）
低置信度 → 轻度压缩（保留更多上下文帮助模型理解）

灵感来源：
- LLM-DCP (arXiv 2504.11004): MDP 压缩，6.9x 压缩率
- Selection-p (EMNLP 2024): 自监督 token 选择，10x 压缩
- Adaptive QuerySelect (arXiv 2407.15504): 查询感知压缩

与现有 compress_prompt_tokens 的区别：
- 现有：固定规则截断中间部分
- 新增：基于置信度的分级压缩 + 语义重要性保留
- 学习：根据任务结果反馈调整重要性权重
"""

import os
import re
import time
from dataclasses import dataclass

from io_utils import read_jsonl, append_jsonl


@dataclass
class CompressionResult:
    """压缩结果"""
    compressed_prompt: str
    original_length: int
    compressed_length: int
    compression_ratio: float  # compressed / original
    level: str  # none / light / moderate / aggressive


# 压缩级别定义
COMPRESSION_LEVELS = {
    "none": {"min_ratio": 1.0, "description": "不压缩"},
    "light": {"min_ratio": 0.8, "description": "轻度压缩（去除冗余）"},
    "moderate": {"min_ratio": 0.6, "description": "中度压缩（精简结构）"},
    "aggressive": {"min_ratio": 0.4, "description": "激进压缩（核心信息）"},
}


# ─── 重要性权重学习 ──────────────────────────────────────────

# 默认重要性权重（可被 ImportanceLearner 调整）
_DEFAULT_IMPORTANCE_WEIGHTS = {
    "base": 0.5,
    "length_short": -0.4,       # < 5 字符
    "length_medium": 0.1,       # 10-200 字符
    "number": 0.1,
    "instruction": 0.15,
    "format": 0.15,
    "example": 0.1,
    "separator": 0.2,
}


class ImportanceLearner:
    """
    学习哪些特征对任务成功率更有价值。

    设计模式：观察者 — 由 task_router.py 在任务完成后调用 record_outcome()。
    通过跟踪特征 → 成功率的映射，自适应调整重要性权重。
    """

    LEARNING_RATE = 0.02
    DECAY = 0.995  # 权重衰减，防止过拟合

    def __init__(self, cache_dir: str):
        self.data_file = os.path.join(cache_dir, "importance_weights.jsonl")
        self._weights = dict(_DEFAULT_IMPORTANCE_WEIGHTS)
        self._load()

    def _load(self) -> None:
        """加载历史权重"""
        entries = read_jsonl(self.data_file)
        if entries:
            last = entries[-1]
            for key in _DEFAULT_IMPORTANCE_WEIGHTS:
                if key in last:
                    self._weights[key] = last[key]

    def get_weights(self) -> dict[str, float]:
        return dict(self._weights)

    def record_outcome(self, sentence_features: dict[str, bool], success: bool) -> None:
        """
        记录压缩结果的反馈。

        Args:
            sentence_features: 被保留句子的特征 (e.g. {"instruction": True, "number": False})
            success: 任务是否成功
        """
        delta = self.LEARNING_RATE if success else -self.LEARNING_RATE * 0.5
        for feature, present in sentence_features.items():
            if present and feature in self._weights:
                self._weights[feature] = max(0.01, self._weights[feature] + delta)
        # 衰减所有权重（防止无限增长）
        for key in self._weights:
            self._weights[key] *= self.DECAY
        # 持久化
        append_jsonl(self.data_file, {**self._weights, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})


_importance_learner = None


def get_importance_learner(cache_dir: str | None = None) -> ImportanceLearner:
    global _importance_learner
    if _importance_learner is None:
        if cache_dir is None:
            from config import get_config
            cache_dir = get_config().cache_dir
        _importance_learner = ImportanceLearner(cache_dir)
    return _importance_learner


def estimate_importance(sentence: str, weights: dict[str, float] | None = None) -> float:
    """
    估计句子的语义重要性。

    高重要性信号：
    - 包含数字、专有名词
    - 包含指令性动词
    - 包含格式要求
    - 句子长度适中（非空、非过短）

    Args:
        sentence: 待评估的句子
        weights: 自定义权重（来自 ImportanceLearner，None 则用默认）
    """
    w = weights or _DEFAULT_IMPORTANCE_WEIGHTS
    score = w.get("base", 0.5)

    # 长度信号
    length = len(sentence.strip())
    if length < 5:
        return max(0.01, w.get("base", 0.5) + w.get("length_short", -0.4))
    if 10 < length < 200:
        score += w.get("length_medium", 0.1)

    # 数字信号（数据、指标）
    if re.search(r'\d+', sentence):
        score += w.get("number", 0.1)

    # 指令性动词
    instruction_verbs = [
        "请", "需要", "必须", "确保", "注意", "要求",
        "please", "must", "ensure", "should", "require",
    ]
    if any(v in sentence.lower() for v in instruction_verbs):
        score += w.get("instruction", 0.15)

    # 格式要求
    format_signals = ["格式", "输出", "返回", "表格", "列表", "JSON", "XML"]
    if any(s in sentence for s in format_signals):
        score += w.get("format", 0.15)

    # 示例标记
    example_signals = ["示例", "例如", "比如", "example", "e.g.", "for instance"]
    if any(s in sentence.lower() for s in example_signals):
        score += w.get("example", 0.1)

    # 分隔符（结构标记）
    if re.match(r'^[-=*#]+$', sentence.strip()):
        score += w.get("separator", 0.2)

    return min(1.0, score)


def extract_sentence_features(sentence: str) -> dict[str, bool]:
    """提取句子的特征向量（用于 ImportanceLearner 反馈）"""
    length = len(sentence.strip())
    instruction_verbs = ["请", "需要", "必须", "确保", "注意", "要求", "please", "must", "ensure", "should"]
    format_signals = ["格式", "输出", "返回", "表格", "列表", "JSON", "XML"]
    example_signals = ["示例", "例如", "比如", "example", "e.g."]

    return {
        "number": bool(re.search(r'\d+', sentence)),
        "instruction": any(v in sentence.lower() for v in instruction_verbs),
        "format": any(s in sentence for s in format_signals),
        "example": any(s in sentence.lower() for s in example_signals),
        "separator": bool(re.match(r'^[-=*#]+$', sentence.strip())),
        "length_medium": 10 < length < 200,
    }


def compress_adaptive(
    prompt: str,
    confidence: float = 0.5,
    task_type: str = "",
    target_ratio: float | None = None,
) -> CompressionResult:
    """
    自适应 Prompt 压缩。

    根据置信度选择压缩级别：
    - confidence >= 0.8 → aggressive (保留 40%)
    - confidence >= 0.6 → moderate (保留 60%)
    - confidence >= 0.4 → light (保留 80%)
    - confidence < 0.4 → none (不压缩)

    Args:
        prompt: 原始 prompt
        confidence: 模型置信度 (0-1)
        task_type: 任务类型
        target_ratio: 强制目标压缩比（覆盖自动选择）
    """
    original_length = len(prompt)

    # 短 prompt 不压缩
    if original_length < 200:
        return CompressionResult(
            compressed_prompt=prompt,
            original_length=original_length,
            compressed_length=original_length,
            compression_ratio=1.0,
            level="none",
        )

    # 选择压缩级别
    if target_ratio is not None:
        level = "custom"
        ratio = target_ratio
    elif confidence >= 0.8:
        level = "aggressive"
        ratio = 0.4
    elif confidence >= 0.6:
        level = "moderate"
        ratio = 0.6
    elif confidence >= 0.4:
        level = "light"
        ratio = 0.8
    else:
        level = "none"
        ratio = 1.0

    if ratio >= 1.0:
        return CompressionResult(
            compressed_prompt=prompt,
            original_length=original_length,
            compressed_length=original_length,
            compression_ratio=1.0,
            level="none",
        )

    # 分句并评估重要性
    lines = prompt.split("\n")
    scored_lines = []
    for line in lines:
        importance = estimate_importance(line)
        scored_lines.append((importance, line))

    # 按重要性排序，保留 top-ratio 的行
    target_lines = max(3, int(len(lines) * ratio))
    scored_lines.sort(key=lambda x: x[0], reverse=True)
    kept_lines = scored_lines[:target_lines]

    # 恢复原始顺序 — 用索引而非 id() 去重（id() 在字符串 interning 不确定时不可靠）
    kept_indices = set()
    kept_values = [line for _, line in kept_lines]
    remaining = list(kept_values)
    for i, line in enumerate(lines):
        for j, kept_line in enumerate(remaining):
            if line == kept_line:
                kept_indices.add(i)
                remaining.pop(j)
                break

    result_lines = [line for i, line in enumerate(lines) if i in kept_indices]
    compressed = "\n".join(result_lines)
    compressed_length = len(compressed)

    return CompressionResult(
        compressed_prompt=compressed,
        original_length=original_length,
        compressed_length=compressed_length,
        compression_ratio=compressed_length / original_length if original_length > 0 else 1.0,
        level=level,
    )


def get_token_savings(confidence: float, original_tokens: int) -> int:
    """估算 token 节省量"""
    if confidence >= 0.8:
        return int(original_tokens * 0.6)  # 节省 60%
    elif confidence >= 0.6:
        return int(original_tokens * 0.4)  # 节省 40%
    elif confidence >= 0.4:
        return int(original_tokens * 0.2)  # 节省 20%
    return 0
