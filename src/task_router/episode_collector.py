"""
路由快照收集器 — 记录每次路由决策的完整上下文

为 Self-Reflective Routing (SRR) 提供数据基础。
每次 run_task() 执行后自动收集决策快照，供 QualityJudge 和 ReflectionEngine 消费。

设计原则：
- 零侵入：只读取 Task 属性，不修改执行逻辑
- 延迟写入：内存缓冲，批量 flush 到磁盘
- PII 安全：action/text 截断且可选脱敏
"""

import hashlib
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from task_router.io_utils import read_jsonl, append_jsonl, write_jsonl


# ─── 数据结构 ──────────────────────────────────────────────────


@dataclass
class Episode:
    """一次路由决策的完整快照。

    包含输入、决策信号、路由结果和输出，
    供 QualityJudge 和 ReflectionEngine 做多维度分析。
    """

    # 标识
    episode_id: str = ""
    timestamp: str = ""

    # 输入
    action: str = ""
    text: str = ""
    task_type: str = ""
    capability: str = ""

    # 决策信号
    complexity_score: float = 0.0
    confidence_data: dict = field(default_factory=dict)
    strategy: str = ""
    strategy_reason: str = ""

    # 路由决策
    route: str = ""
    model_used: str = ""
    routing_signals: dict = field(default_factory=dict)

    # 输出
    output: str = ""
    tokens_input: int = 0
    tokens_output: int = 0
    time_ms: int = 0
    cost_saved: float = 0.0

    # 后验（反思阶段填充）
    quality_scores: dict = field(default_factory=dict)
    optimal_route: str = ""
    optimal_strategy: str = ""
    routing_error: str = ""
    reflection_notes: str = ""

    def __post_init__(self) -> None:
        if not self.episode_id:
            raw = f"{self.action}{self.text[:100]}{time.time()}"
            self.episode_id = hashlib.md5(raw.encode()).hexdigest()[:16]
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Episode":
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})


# ─── 文本截断常量 ──────────────────────────────────────────────

MAX_ACTION_LEN = 500
MAX_TEXT_LEN = 500
MAX_OUTPUT_LEN = 500


def _truncate(text: str, max_len: int) -> str:
    """安全截断文本"""
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"... (共 {len(text)} 字符)"


# ─── 收集器 ──────────────────────────────────────────────────


class EpisodeCollector:
    """路由决策快照收集器。

    在 run_task() 执行后记录完整决策上下文，
    供 QualityJudge 和 ReflectionEngine 消费。

    使用方式：
        collector = EpisodeCollector(cache_dir="/path/to/cache")
        collector.record(task, routing_context={...})
        collector.flush()  # 显式刷盘（也可依赖自动刷盘）

    线程安全：内部使用 threading.Lock。
    """

    MAX_BUFFER_SIZE = 20  # 缓冲区最大条目数，超过自动 flush
    MAX_EPISODES = 10000  # episodes.jsonl 最大行数

    def __init__(self, cache_dir: str, scrub_pii: bool = True):
        self.episodes_file = f"{cache_dir}/episodes.jsonl"
        self.scrub_pii = scrub_pii
        self._buffer: list[dict] = []
        self._lock = threading.Lock()

    def record(self, task: Any, routing_context: Optional[dict] = None) -> str:
        """记录一次路由决策快照。

        参数:
            task: Task 对象（来自 task_router.Task）
            routing_context: 路由上下文信息，包含:
                - complexity_score: float
                - confidence_data: dict
                - strategy: str
                - strategy_reason: str
                - routing_signals: dict（五层决策信号）
                - task_type: str
                - capability: str

        返回:
            episode_id: 本次快照的唯一标识
        """
        ctx = routing_context or {}

        episode = Episode(
            action=_truncate(task.action, MAX_ACTION_LEN),
            text=_truncate(getattr(task, "text", ""), MAX_TEXT_LEN),
            task_type=ctx.get("task_type", ""),
            capability=ctx.get("capability", ""),
            complexity_score=ctx.get("complexity_score", 0.0),
            confidence_data=ctx.get("confidence_data", {}),
            strategy=ctx.get("strategy", ""),
            strategy_reason=ctx.get("strategy_reason", ""),
            route=task.route,
            model_used=task.model_used,
            routing_signals=ctx.get("routing_signals", {}),
            output=_truncate(task.output, MAX_OUTPUT_LEN),
            tokens_input=task.tokens_input,
            tokens_output=task.tokens_output,
            time_ms=task.time_ms,
            cost_saved=task.cost_saved,
        )

        # PII 脱敏（延迟导入避免循环依赖）
        if self.scrub_pii:
            try:
                from task_router.privacy import get_privacy_filter
                pf = get_privacy_filter()
                action_result = pf.anonymize(episode.action)
                episode.action = action_result.text
                text_result = pf.anonymize(episode.text)
                episode.text = text_result.text
            except Exception:
                pass  # 脱敏失败不阻塞记录

        entry = episode.to_dict()

        with self._lock:
            self._buffer.append(entry)
            if len(self._buffer) >= self.MAX_BUFFER_SIZE:
                self._flush_buffer()

        return episode.episode_id

    def flush(self) -> None:
        """显式将缓冲写入磁盘"""
        with self._lock:
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        """内部 flush（调用者需持有锁）"""
        if not self._buffer:
            return
        for entry in self._buffer:
            append_jsonl(self.episodes_file, entry, max_lines=self.MAX_EPISODES)
        self._buffer.clear()

    def get_recent(self, n: int = 100) -> list[dict]:
        """获取最近 N 条 episode（从磁盘读取，不含缓冲区）"""
        episodes = read_jsonl(self.episodes_file)
        return episodes[-n:]

    def get_all(self) -> list[dict]:
        """获取所有 episode"""
        self.flush()
        return read_jsonl(self.episodes_file)

    def count(self) -> int:
        """获取 episode 总数（含缓冲区）"""
        with self._lock:
            buffer_count = len(self._buffer)
        disk_episodes = read_jsonl(self.episodes_file)
        return len(disk_episodes) + buffer_count

    def get_since(self, timestamp: str) -> list[dict]:
        """获取指定时间之后的 episode"""
        episodes = read_jsonl(self.episodes_file)
        return [e for e in episodes if e.get("timestamp", "") > timestamp]

    def clear(self) -> None:
        """清空所有 episode（测试用）"""
        with self._lock:
            self._buffer.clear()
        write_jsonl(self.episodes_file, [])


# ─── 全局实例（延迟初始化，线程安全）────────────────────────


_collector: Optional[EpisodeCollector] = None
_collector_lock = threading.Lock()


def get_episode_collector(cache_dir: Optional[str] = None) -> EpisodeCollector:
    """获取全局 EpisodeCollector 实例"""
    global _collector
    if _collector is None:
        with _collector_lock:
            if _collector is None:
                if cache_dir is None:
                    from task_router.config import get_config
                    cache_dir = get_config().cache_dir
                _collector = EpisodeCollector(cache_dir=cache_dir)
    return _collector


def reset_episode_collector() -> None:
    """重置全局实例（测试用）"""
    global _collector
    with _collector_lock:
        _collector = None
