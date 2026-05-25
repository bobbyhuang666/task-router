"""
结果感知缓存优化（OATS-Inspired Quality Cache）

核心创新：将 OATS (Outcome-Aware Tool Selection) 论文的核心思想
应用到语义缓存中——缓存条目根据历史结果动态调整优先级。

OATS 论文核心思想：
- 嵌入向量根据成功/失败结果离线优化
- 向成功结果的质心移动，远离失败结果
- 零在线开销（所有优化在离线完成）

应用到缓存：
- 缓存命中时，根据条目的历史成功率调整匹配阈值
- 高成功率条目 → 放宽匹配阈值（更容易命中）
- 低成功率条目 → 收紧匹配阈值（更难命中）
- 定期离线优化（不影响在线性能）

参考文献：
[1] OATS: Outcome-Aware Tool Selection. vLLM-SR 2026.
"""

import os
import time
import threading
from typing import Optional



class OutcomeAwareCache:
    """
    结果感知缓存管理器。

    在标准语义缓存之上添加质量权重层：
    - 每个缓存条目有质量分（基于历史成功率）
    - 质量分影响缓存命中优先级
    - 定期离线优化质量分
    """

    DEFAULT_QUALITY = 0.5   # 默认质量分
    LEARNING_RATE = 0.1     # 质量分更新速率
    MIN_QUALITY = 0.1       # 最低质量分（防止完全被淘汰）
    MAX_QUALITY = 1.0       # 最高质量分

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.quality_file = os.path.join(cache_dir, "cache_quality.jsonl")
        self.model_file = os.path.join(cache_dir, "cache_quality_model.json")
        self._lock = threading.Lock()
        self._quality_scores: dict[str, float] = {}  # cache_key -> quality_score
        self._outcome_history: dict[str, list[bool]] = {}  # cache_key -> [success, ...]
        self._load()

    def _load(self) -> None:
        """加载质量模型"""
        import json
        if os.path.exists(self.model_file):
            try:
                with open(self.model_file) as f:
                    data = json.load(f)
                self._quality_scores = data.get("quality_scores", {})
            except (json.JSONDecodeError, TypeError):
                pass

    def _save(self) -> None:
        """保存质量模型"""
        import json
        with open(self.model_file, "w") as f:
            json.dump({
                "quality_scores": self._quality_scores,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }, f, indent=2)

    def get_quality(self, cache_key: str) -> float:
        """获取缓存条目的质量分"""
        return self._quality_scores.get(cache_key, self.DEFAULT_QUALITY)

    def get_effective_threshold(self, cache_key: str, base_threshold: float = 0.85) -> float:
        """
        获取缓存条目的有效匹配阈值。

        核心创新（来自 OATS）：
        - 高质量条目 → 阈值降低（更容易命中）
        - 低质量条目 → 阈值升高（更难命中）

        这实现了"结果感知的嵌入空间优化"：
        成功的缓存条目在语义空间中的"影响范围"更大。
        """
        quality = self.get_quality(cache_key)
        # 质量分映射到阈值偏移
        # quality=1.0 → threshold -= 0.1（更容易命中）
        # quality=0.1 → threshold += 0.1（更难命中）
        offset = (self.DEFAULT_QUALITY - quality) * 0.2
        return max(0.5, min(0.95, base_threshold + offset))

    def record_outcome(self, cache_key: str, success: bool) -> None:
        """
        记录缓存命中的结果并更新质量分。

        使用指数移动平均（EMA）更新，类似 OATS 的在线优化。
        """
        with self._lock:
            # EMA 更新
            current = self._quality_scores.get(cache_key, self.DEFAULT_QUALITY)
            reward = 1.0 if success else 0.0
            new_quality = current * (1 - self.LEARNING_RATE) + reward * self.LEARNING_RATE
            new_quality = max(self.MIN_QUALITY, min(self.MAX_QUALITY, new_quality))
            self._quality_scores[cache_key] = new_quality

            # 记录历史
            if cache_key not in self._outcome_history:
                self._outcome_history[cache_key] = []
            self._outcome_history[cache_key].append(success)
            # 只保留最近 100 条
            if len(self._outcome_history[cache_key]) > 100:
                self._outcome_history[cache_key] = self._outcome_history[cache_key][-100:]

            # 每 10 次更新保存一次
            total_updates = sum(len(v) for v in self._outcome_history.values())
            if total_updates % 10 == 0:
                self._save()

    def get_stats(self) -> dict:
        """获取缓存质量统计"""
        n_entries = len(self._quality_scores)
        if n_entries == 0:
            return {"total_entries": 0, "avg_quality": 0, "high_quality": 0, "low_quality": 0}

        qualities = list(self._quality_scores.values())
        avg_quality = sum(qualities) / len(qualities)
        high_quality = sum(1 for q in qualities if q > 0.7)
        low_quality = sum(1 for q in qualities if q < 0.3)

        return {
            "total_entries": n_entries,
            "avg_quality": round(avg_quality, 3),
            "high_quality_count": high_quality,
            "low_quality_count": low_quality,
        }

    def flush(self) -> None:
        """强制持久化所有未保存的质量分数据。"""
        with self._lock:
            self._save()

    def cleanup_stale(self, min_quality: float = 0.15) -> int:
        """
        清理质量过低的缓存条目标记。

        注意：这不删除缓存条目本身，只是将极低质量的条目标记为"过期"。
        实际的缓存淘汰由 SemanticCache 的 TTL 机制处理。
        """
        with self._lock:
            stale = [k for k, v in self._quality_scores.items() if v < min_quality]
            for k in stale:
                del self._quality_scores[k]
            self._save()
            return len(stale)


# ─── 全局实例 ──────────────────────────────────────────────────

_quality_cache: Optional[OutcomeAwareCache] = None
_quality_cache_lock = threading.Lock()


def get_outcome_cache(cache_dir: Optional[str] = None) -> OutcomeAwareCache:
    """获取全局 OutcomeAwareCache 实例"""
    global _quality_cache
    if _quality_cache is None:
        with _quality_cache_lock:
            if _quality_cache is None:
                if cache_dir is None:
                    from task_router.config import get_config
                    cache_dir = get_config().cache_dir
                _quality_cache = OutcomeAwareCache(cache_dir)
    return _quality_cache
