"""
A3M 可学习权重 — 从路由反馈中自动调整评分参数

核心思路：每次任务执行后记录 (route, score, success)，用增量学习调整权重。
- 本地成功 → 降低该类型任务的阈值（更多走本地）
- 本地失败 → 升高该类型任务的阈值（更多走云端）
- 云端成功 → 确认当前权重合理
"""

import json
import os
import time
import threading
from dataclasses import dataclass, asdict

from io_utils import read_jsonl, append_jsonl


# ─── 默认权重配置 ──────────────────────────────────────────────

@dataclass
class A3MWeights:
    """A3M 评分权重（可学习）"""
    # 信号权重乘数
    verb_multiplier: float = 3.0
    multi_step_weight: float = 0.3
    domain_weight: float = 0.5
    text_long_threshold: int = 2000
    text_mid_threshold: int = 1000
    text_short_threshold: int = 500
    text_long_score: float = 3.0
    text_mid_score: float = 2.0
    text_short_score: float = 1.0
    text_tiny_penalty: float = -1.0
    text_tiny_max_len: int = 50
    file_many_threshold: int = 20
    file_some_threshold: int = 5
    file_many_score: float = 2.0
    file_some_score: float = 1.0
    local_pattern_weight: float = -1.0
    local_pattern_max: int = 3
    action_long_threshold: int = 100
    action_short_threshold: int = 10
    action_long_score: float = 0.5
    action_short_score: float = -1.0

    # 路由阈值（可学习）
    base_threshold: float = 3.0

    # 学习率
    learning_rate: float = 0.05

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "A3MWeights":
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})


# ─── 权重追踪器 ──────────────────────────────────────────────

class WeightTracker:
    """从路由反馈中学习权重调整"""

    def __init__(self, cache_dir: str):
        self.data_file = os.path.join(cache_dir, "weight_history.jsonl")
        self.weights_file = os.path.join(cache_dir, "a3m_weights.json")
        self._lock = threading.Lock()
        os.makedirs(cache_dir, exist_ok=True)
        self.weights = self._load_weights()

    def _load_weights(self) -> A3MWeights:
        """加载已学习的权重"""
        if os.path.exists(self.weights_file):
            try:
                with open(self.weights_file) as f:
                    return A3MWeights.from_dict(json.load(f))
            except (json.JSONDecodeError, TypeError):
                pass
        return A3MWeights()

    def _save_weights(self) -> None:
        """保存权重到文件"""
        with open(self.weights_file, "w") as f:
            json.dump(self.weights.to_dict(), f, indent=2)

    def record_outcome(
        self,
        task_type: str,
        route: str,
        score: float,
        success: bool,
        local_model: str = "",
    ) -> None:
        """记录一次路由结果"""
        entry = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "task_type": task_type,
            "route": route,
            "score": round(score, 3),
            "success": success,
            "local_model": local_model,
        }
        append_jsonl(self.data_file, entry)

        # 根据结果调整权重
        self._update_weights(task_type, route, score, success)

    def _update_weights(
        self, task_type: str, route: str, score: float, success: bool
    ) -> None:
        """增量学习：根据路由结果微调权重（考虑任务复杂度，线程安全）"""
        with self._lock:
            lr = self.weights.learning_rate
            is_local = route.startswith("local")

            if is_local:
                # 本地任务：根据路由评分调整幅度
                # 低分任务（明确该走本地）成功 → 大幅鼓励
                # 高分任务（边缘任务）成功 → 小幅鼓励
                confidence = max(0.3, 1.0 - score / 10.0)
                if success:
                    self.weights.base_threshold += lr * 0.5 * confidence
                else:
                    self.weights.base_threshold -= lr * 1.0
            elif route.startswith("cloud"):
                if success:
                    # 云端成功 → 阈值微降（确认复杂任务确实该走云端）
                    self.weights.base_threshold -= lr * 0.1

            # 限制阈值范围
            self.weights.base_threshold = max(1.0, min(8.0, self.weights.base_threshold))

            self._save_weights()

    def get_weights(self) -> A3MWeights:
        """获取当前权重"""
        return self.weights

    def get_stats(self) -> dict:
        """获取学习统计"""
        entries = read_jsonl(self.data_file)
        if not entries:
            return {"total": 0, "local_success": 0, "local_fail": 0,
                    "cloud_success": 0, "current_threshold": self.weights.base_threshold}

        local_success = sum(1 for e in entries if "local" in e.get("route", "") and e.get("success"))
        local_fail = sum(1 for e in entries if "local" in e.get("route", "") and not e.get("success"))
        cloud_success = sum(1 for e in entries if "cloud" in e.get("route", "") and e.get("success"))

        return {
            "total": len(entries),
            "local_success": local_success,
            "local_fail": local_fail,
            "cloud_success": cloud_success,
            "local_success_rate": round(local_success / max(1, local_success + local_fail), 2),
            "current_threshold": round(self.weights.base_threshold, 3),
            "learning_rate": self.weights.learning_rate,
        }

    def reset(self) -> None:
        """重置为默认权重"""
        self.weights = A3MWeights()
        self._save_weights()


# 全局实例（延迟初始化，线程安全）
_tracker: WeightTracker | None = None
_tracker_lock = threading.Lock()


def get_weight_tracker(cache_dir: str | None = None) -> WeightTracker:
    global _tracker
    if _tracker is None:
        with _tracker_lock:
            if _tracker is None:
                if cache_dir is None:
                    from config import get_config
                    cache_dir = get_config().cache_dir
                _tracker = WeightTracker(cache_dir)
    return _tracker
