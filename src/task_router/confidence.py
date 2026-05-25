"""
置信度门控级联 — 先执行本地模型，置信度不够再升级到云端

核心思路（DiSRouter + FrugalGPT + UCCI）：
1. 本地模型执行任务，同时提取 token 级置信度信号
2. 用等调回归（isotonic regression）校准原始置信度
3. 校准后置信度低于阈值 → 升级到云端
4. 记录每次升级原因，用于闭环蒸馏

关键数据：
- FrugalGPT: 级联可在同等质量下减少 98% 成本
- UCCI: 等调回归将校准误差从 0.12 降到 0.03
"""

import math
import os
import time
from typing import Optional

from task_router.io_utils import read_jsonl, append_jsonl


# ─── 置信度信号提取 ──────────────────────────────────────────────

def extract_confidence(logprobs: list[dict]) -> dict:
    """
    从 token 级 logprobs 提取置信度信号。

    Ollama API 返回格式（logprobs=true）：
    [{"token": "Hello", "logprob": -0.1, "top_logprobs": {"Hello": -0.1, "Hi": -1.2}}, ...]

    返回:
        {
            "entropy": float,        # 平均 token 熵（越低越确定）
            "margin": float,         # 平均 top-1 vs top-2 差距（越大越确定）
            "confidence": float,     # 综合置信度 [0, 1]（越高越确定）
            "token_count": int,      # 输出 token 数
            "max_entropy": float,    # 最大单 token 熵（最不确定的 token）
        }
    """
    if not logprobs:
        return {"entropy": 1.0, "margin": 0.0, "confidence": 0.0, "token_count": 0, "max_entropy": 1.0}

    entropies = []
    margins = []

    for lp in logprobs:
        top = lp.get("top_logprobs", {})
        if not top:
            # 没有 top_logprobs，用 logprob 本身
            logp = lp.get("logprob", -10.0)
            entropies.append(-logp)
            margins.append(abs(logp))
            continue

        # 计算熵: H = -sum(p * log(p))
        probs = []
        for v in top.values():
            try:
                p = math.exp(v)
                probs.append(p)
            except (ValueError, OverflowError):
                continue

        if not probs:
            entropies.append(1.0)
            margins.append(0.0)
            continue

        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]

        entropy = -sum(p * math.log(max(p, 1e-10)) for p in probs)
        entropies.append(entropy)

        # 计算 margin: top-1 - top-2
        sorted_probs = sorted(probs, reverse=True)
        margin = sorted_probs[0] - (sorted_probs[1] if len(sorted_probs) > 1 else 0.0)
        margins.append(margin)

    avg_entropy = sum(entropies) / len(entropies)
    avg_margin = sum(margins) / len(margins)
    max_entropy = max(entropies)

    # 综合置信度：margin 越大越确定，entropy 越低越确定
    # 归一化到 [0, 1]
    # margin 在 [0, 1] 之间，越大越好
    # entropy 在 [0, log(vocab_size)] 之间，越小越好（假设 vocab=32000, max=10.4）
    entropy_score = max(0, 1.0 - avg_entropy / 5.0)  # 归一化
    margin_score = min(1.0, avg_margin * 2.0)  # 归一化

    # 综合：70% margin + 30% entropy（margin 更稳定）
    confidence = 0.7 * margin_score + 0.3 * entropy_score

    return {
        "entropy": round(avg_entropy, 4),
        "margin": round(avg_margin, 4),
        "confidence": round(confidence, 4),
        "token_count": len(logprobs),
        "max_entropy": round(max_entropy, 4),
    }


def extract_confidence_from_text(text: str) -> dict:
    """
    从纯文本输出提取启发式置信度（无 logprobs 时的降级方案）。

    基于输出特征：
    - 长度适中（不太短也不太长）→ 更可能有信心
    - 无失败信号词 → 更可能有信心
    - 无重复内容 → 更可能有信心
    """
    if not text or not text.strip():
        return {"entropy": 1.0, "margin": 0.0, "confidence": 0.0, "token_count": 0, "max_entropy": 1.0}

    text = text.strip()

    # 长度分数：太短或太长都不好
    length = len(text)
    if length < 5:
        length_score = 0.2
    elif length < 20:
        length_score = 0.5
    elif length < 200:
        length_score = 0.8
    else:
        length_score = 0.7  # 过长可能是废话

    # 失败信号检测
    failure_signals = ["抱歉", "无法", "不能", "不懂", "作为AI", "我无法", "error", "Error"]
    has_failure = any(s in text for s in failure_signals)
    failure_score = 0.1 if has_failure else 0.9

    # 重复检测
    words = text.split()
    if len(words) >= 3:
        repeat = sum(1 for i in range(len(words)-2) if words[i] == words[i+1] == words[i+2])
        repeat_score = max(0.3, 1.0 - repeat * 0.3)
    else:
        repeat_score = 0.6

    confidence = 0.4 * length_score + 0.4 * failure_score + 0.2 * repeat_score

    return {
        "entropy": round(1.0 - confidence, 4),
        "margin": round(confidence, 4),
        "confidence": round(confidence, 4),
        "token_count": len(words),
        "max_entropy": round(1.0 - confidence, 4),
    }


# ─── 置信度校准器（等调回归） ────────────────────────────────────

class ConfidenceCalibrator:
    """
    等调回归校准器 — 将原始置信度映射到实际正确概率。

    UCCI 研究表明，原始置信度是 miscalibrated 的。
    等调回归（isotonic regression）是一种非参数校准方法：
    - 输入：(raw_confidence, was_correct) 对
    - 输出：校准函数 f(raw_conf) → P(correct)
    """

    def __init__(self, cache_dir: str):
        self.data_file = os.path.join(cache_dir, "confidence_history.jsonl")
        self.calibration_file = os.path.join(cache_dir, "confidence_calibration.json")
        self._calibration_data: list[tuple[float, bool]] = []
        self._calibration_fn: Optional[list[tuple[float, float]]] = None
        self._load()

    def _load(self) -> None:
        """加载历史校准数据"""
        entries = read_jsonl(self.data_file)
        self._calibration_data = [
            (e["confidence"], e["was_correct"])
            for e in entries
            if "confidence" in e and "was_correct" in e
        ]
        if len(self._calibration_data) >= 20:
            self._fit_isotonic()

    def record(self, confidence: float, was_correct: bool) -> None:
        """记录一次置信度-正确性对"""
        entry = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "confidence": round(confidence, 4),
            "was_correct": was_correct,
        }
        append_jsonl(self.data_file, entry)
        self._calibration_data.append((confidence, was_correct))

        # 每 10 条数据重新校准
        if len(self._calibration_data) % 10 == 0 and len(self._calibration_data) >= 20:
            self._fit_isotonic()

    def _fit_isotonic(self) -> None:
        """拟合等调回归（简化实现，无需 sklearn）"""
        data = sorted(self._calibration_data, key=lambda x: x[0])

        # Pool Adjacent Violators 算法
        # 将数据按置信度排序，合并违反单调性的相邻桶
        buckets = []
        for conf, correct in data:
            buckets.append([conf, 1.0 if correct else 0.0, 1])  # [sum_conf, sum_correct, count]

        # 合并违反单调性的桶
        i = 0
        while i < len(buckets) - 1:
            if buckets[i][1] / buckets[i][2] > buckets[i+1][1] / buckets[i+1][2]:
                # 合并
                merged = [
                    buckets[i][0] + buckets[i+1][0],
                    buckets[i][1] + buckets[i+1][1],
                    buckets[i][2] + buckets[i+1][2],
                ]
                buckets[i:i+2] = [merged]
                if i > 0:
                    i -= 1
            else:
                i += 1

        # 转换为 (threshold, calibrated_probability) 对
        self._calibration_fn = []
        for bucket in buckets:
            avg_conf = bucket[0] / bucket[2]
            prob_correct = bucket[1] / bucket[2]
            self._calibration_fn.append((avg_conf, prob_correct))

    def calibrate(self, raw_confidence: float) -> float:
        """
        校准原始置信度 → P(correct)。

        如果没有足够数据做校准，返回原始置信度。
        """
        if not self._calibration_fn or len(self._calibration_fn) < 2:
            return raw_confidence

        # 在校准函数中查找（线性插值）
        fn = self._calibration_fn

        if raw_confidence <= fn[0][0]:
            return fn[0][1]
        if raw_confidence >= fn[-1][0]:
            return fn[-1][1]

        for i in range(len(fn) - 1):
            if fn[i][0] <= raw_confidence <= fn[i+1][0]:
                dx = fn[i+1][0] - fn[i][0]
                if dx < 1e-10:
                    return fn[i][1]
                t = (raw_confidence - fn[i][0]) / dx
                return fn[i][1] + t * (fn[i+1][1] - fn[i][1])

        return raw_confidence

    def get_stats(self) -> dict:
        """获取校准统计"""
        total = len(self._calibration_data)
        correct = sum(1 for _, c in self._calibration_data if c)
        return {
            "total_samples": total,
            "overall_accuracy": round(correct / max(1, total), 3),
            "is_calibrated": self._calibration_fn is not None and len(self._calibration_fn) >= 2,
            "calibration_buckets": len(self._calibration_fn) if self._calibration_fn else 0,
        }


# ─── 级联决策器 ──────────────────────────────────────────────

class CascadeDecision:
    """级联决策：本地执行后判断是否需要升级到云端"""

    def __init__(self, cache_dir: str, escalation_threshold: float = 0.4):
        """
        Args:
            cache_dir: 缓存目录
            escalation_threshold: 校准后置信度低于此值则升级到云端
        """
        self.calibrator = ConfidenceCalibrator(cache_dir)
        self.escalation_threshold = escalation_threshold
        self.history_file = os.path.join(cache_dir, "cascade_history.jsonl")

    def should_escalate(self, confidence_data: dict) -> dict:
        """
        判断是否需要升级到云端。

        返回:
            {
                "escalate": bool,
                "raw_confidence": float,
                "calibrated_confidence": float,
                "reason": str,
            }
        """
        raw_conf = confidence_data.get("confidence", 0.0)
        calibrated = self.calibrator.calibrate(raw_conf)

        # 数据不足时用启发式规则
        if not self.calibrator.get_stats()["is_calibrated"]:
            # 没有足够校准数据，用原始置信度 + 保守阈值
            threshold = self.escalation_threshold + 0.1  # 更保守
            escalate = calibrated < threshold
            reason = f"未校准模式: 原始={raw_conf:.2f}, 阈值={threshold:.2f}"
        else:
            escalate = calibrated < self.escalation_threshold
            reason = f"已校准: 原始={raw_conf:.2f}, 校准后={calibrated:.2f}, 阈值={self.escalation_threshold:.2f}"

        return {
            "escalate": escalate,
            "raw_confidence": raw_conf,
            "calibrated_confidence": round(calibrated, 4),
            "reason": reason,
        }

    def record_outcome(self, confidence_data: dict, was_correct: bool, escalated: bool) -> None:
        """记录级联决策结果"""
        entry = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "confidence": confidence_data.get("confidence", 0.0),
            "calibrated": self.calibrator.calibrate(confidence_data.get("confidence", 0.0)),
            "was_correct": was_correct,
            "escalated": escalated,
        }
        append_jsonl(self.history_file, entry, max_lines=10000)

        # 只有未升级的结果才用于校准（升级的结果由云端处理）
        if not escalated:
            self.calibrator.record(confidence_data.get("confidence", 0.0), was_correct)

    def get_stats(self) -> dict:
        """获取级联统计"""
        entries = read_jsonl(self.history_file)
        if not entries:
            return {"total": 0, "escalated": 0, "escalation_rate": 0.0}

        total = len(entries)
        escalated = sum(1 for e in entries if e.get("escalated"))
        correct_local = sum(1 for e in entries if not e.get("escalated") and e.get("was_correct"))

        return {
            "total": total,
            "escalated": escalated,
            "local_kept": total - escalated,
            "escalation_rate": round(escalated / max(1, total), 3),
            "local_accuracy": round(correct_local / max(1, total - escalated), 3),
            "calibration": self.calibrator.get_stats(),
        }


# 全局实例（延迟初始化）
_cascade: Optional[CascadeDecision] = None


def get_cascade(cache_dir: Optional[str] = None) -> CascadeDecision:
    global _cascade
    if _cascade is None:
        if cache_dir is None:
            from task_router.config import get_config
            cache_dir = get_config().cache_dir
        _cascade = CascadeDecision(cache_dir)
    return _cascade
