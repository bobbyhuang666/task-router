"""
路由快照收集器测试

测试覆盖:
- Episode 数据结构
- EpisodeCollector 记录与 flush
- 文本截断
- 缓冲区管理
- 线程安全
- 全局实例管理
"""

import os
import json
import pytest
import tempfile
import threading

from task_router.episode_collector import (
    Episode,
    EpisodeCollector,
    get_episode_collector,
    reset_episode_collector,
    _truncate,
)
from task_router.routing import Task


# ─── 辅助函数 ──────────────────────────────────────────────


def _make_task(action="分类文件", text="test", route="local", model="qwen-tool"):
    """创建测试用 Task"""
    return Task(action=action, text=text, route=route, model_used=model)


def _make_context(**overrides):
    """创建测试用 routing_context"""
    ctx = {
        "complexity_score": 2.5,
        "confidence_data": {"confidence": 0.8, "entropy": 0.3},
        "strategy": "direct",
        "strategy_reason": "简单任务",
        "routing_signals": {"cascade": {"escalate": False}},
        "task_type": "general_classify",
        "capability": "classification",
    }
    ctx.update(overrides)
    return ctx


# ─── Episode 数据结构测试 ──────────────────────────────────────


class TestEpisode:
    """Episode dataclass 测试"""

    def test_auto_id(self):
        """Episode 自动生成唯一 ID"""
        ep1 = Episode(action="test", text="hello")
        ep2 = Episode(action="test", text="hello")
        assert ep1.episode_id != ""
        assert ep2.episode_id != ""
        assert len(ep1.episode_id) == 16

    def test_auto_timestamp(self):
        """Episode 自动生成时间戳"""
        ep = Episode(action="test")
        assert ep.timestamp != ""
        assert "T" in ep.timestamp  # ISO 格式

    def test_explicit_id(self):
        """可以显式指定 ID"""
        ep = Episode(episode_id="custom123", action="test")
        assert ep.episode_id == "custom123"

    def test_to_dict_roundtrip(self):
        """to_dict → from_dict 往返一致"""
        ep = Episode(
            action="翻译", text="Hello", route="local",
            task_type="translate_en2zh", tokens_input=10, tokens_output=5,
        )
        d = ep.to_dict()
        ep2 = Episode.from_dict(d)
        assert ep2.action == "翻译"
        assert ep2.text == "Hello"
        assert ep2.route == "local"
        assert ep2.tokens_input == 10

    def test_from_dict_ignores_extra_keys(self):
        """from_dict 忽略多余字段"""
        d = {"action": "test", "unknown_field": "value", "extra": 123}
        ep = Episode.from_dict(d)
        assert ep.action == "test"

    def test_default_values(self):
        """默认值正确"""
        ep = Episode()
        assert ep.route == ""
        assert ep.tokens_input == 0
        assert ep.cost_saved == 0.0
        assert ep.quality_scores == {}
        assert ep.routing_error == ""


# ─── 文本截断测试 ──────────────────────────────────────────────


class TestTruncate:
    """_truncate 函数测试"""

    def test_short_text_unchanged(self):
        """短文本不截断"""
        assert _truncate("hello", 10) == "hello"

    def test_exact_length(self):
        """恰好等于最大长度"""
        text = "a" * 10
        assert _truncate(text, 10) == text

    def test_long_text_truncated(self):
        """长文本被截断并添加提示"""
        text = "a" * 100
        result = _truncate(text, 10)
        assert result.startswith("aaaaaaaaaa")
        assert "共 100 字符" in result
        assert len(result) < len(text) + 50

    def test_empty_text(self):
        """空文本返回空字符串"""
        assert _truncate("", 10) == ""
        assert _truncate(None, 10) == ""


# ─── EpisodeCollector 测试 ──────────────────────────────────────


class TestEpisodeCollector:
    """EpisodeCollector 核心功能测试"""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.collector = EpisodeCollector(cache_dir=self.tmpdir, scrub_pii=False)

    def test_record_basic(self):
        """基本记录功能"""
        task = _make_task()
        ctx = _make_context()
        episode_id = self.collector.record(task, ctx)
        assert episode_id != ""
        assert len(episode_id) == 16

    def test_record_to_buffer(self):
        """记录先进入缓冲区"""
        task = _make_task()
        self.collector.record(task, _make_context())
        # 未 flush 时，磁盘文件应为空
        from task_router.io_utils import read_jsonl
        disk = read_jsonl(self.collector.episodes_file)
        assert len(disk) == 0
        # 但 count 应包含缓冲区
        assert self.collector.count() == 1

    def test_flush_writes_to_disk(self):
        """flush 将缓冲区写入磁盘"""
        task = _make_task()
        self.collector.record(task, _make_context())
        self.collector.flush()
        from task_router.io_utils import read_jsonl
        disk = read_jsonl(self.collector.episodes_file)
        assert len(disk) == 1
        assert disk[0]["action"] == "分类文件"

    def test_auto_flush_on_buffer_full(self):
        """缓冲区满时自动 flush"""
        small_collector = EpisodeCollector(cache_dir=self.tmpdir, scrub_pii=False)
        small_collector.MAX_BUFFER_SIZE = 3

        for i in range(5):
            task = Task(action=f"任务{i}", text=f"text{i}", route="local", model_used="test")
            small_collector.record(task, _make_context())

        # 5 条记录，缓冲区大小 3，应该已经自动 flush 了至少一次
        from task_router.io_utils import read_jsonl
        disk = read_jsonl(small_collector.episodes_file)
        assert len(disk) >= 3

    def test_record_preserves_routing_context(self):
        """记录保留完整的路由上下文"""
        task = _make_task(action="翻译成中文", route="cloud", model="deepseek")
        ctx = _make_context(
            complexity_score=4.5,
            strategy="cot",
            task_type="translate_en2zh",
        )
        self.collector.record(task, ctx)
        self.collector.flush()

        episodes = self.collector.get_recent(1)
        assert len(episodes) == 1
        ep = episodes[0]
        assert ep["complexity_score"] == 4.5
        assert ep["strategy"] == "cot"
        assert ep["task_type"] == "translate_en2zh"
        assert ep["route"] == "cloud"

    def test_get_recent(self):
        """get_recent 返回最近 N 条"""
        for i in range(10):
            task = Task(action=f"任务{i}", text="", route="local", model_used="test")
            self.collector.record(task, _make_context())
        self.collector.flush()

        recent = self.collector.get_recent(3)
        assert len(recent) == 3
        # 应该是最后 3 条
        assert recent[-1]["action"] == "任务9"

    def test_get_since(self):
        """get_since 按时间过滤"""
        task1 = _make_task(action="旧任务")
        self.collector.record(task1, _make_context())
        self.collector.flush()

        cutoff = "2099-01-01T00:00:00"  # 未来时间
        task2 = Task(action="新任务", text="", route="local", model_used="test")
        self.collector.record(task2, _make_context())
        self.collector.flush()

        recent = self.collector.get_since(cutoff)
        assert len(recent) == 0  # 两条都在 cutoff 之前

        old = self.collector.get_since("2020-01-01T00:00:00")
        assert len(old) == 2

    def test_clear(self):
        """clear 清空所有数据"""
        for i in range(5):
            task = Task(action=f"任务{i}", text="", route="local", model_used="test")
            self.collector.record(task, _make_context())
        self.collector.flush()
        assert self.collector.count() == 5

        self.collector.clear()
        assert self.collector.count() == 0

    def test_truncation_in_record(self):
        """记录时自动截断长文本"""
        long_text = "x" * 2000
        task = Task(action=long_text, text=long_text, route="local", model_used="test")
        self.collector.record(task, _make_context())
        self.collector.flush()

        episodes = self.collector.get_recent(1)
        assert len(episodes[0]["action"]) < 600  # 截断到 500 + 提示
        assert "共 2000 字符" in episodes[0]["action"]

    def test_record_without_context(self):
        """没有 routing_context 也能记录"""
        task = _make_task()
        episode_id = self.collector.record(task)
        assert episode_id != ""
        self.collector.flush()
        episodes = self.collector.get_recent(1)
        assert episodes[0]["complexity_score"] == 0.0


# ─── 线程安全测试 ──────────────────────────────────────────────


class TestEpisodeCollectorConcurrency:
    """并发安全性测试"""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.collector = EpisodeCollector(cache_dir=self.tmpdir, scrub_pii=False)

    def test_concurrent_record(self):
        """多线程并发记录不崩溃"""
        errors = []

        def record_task(i):
            try:
                task = Task(action=f"并发任务{i}", text="", route="local", model_used="test")
                self.collector.record(task, _make_context())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_task, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert self.collector.count() == 20

    def test_concurrent_flush(self):
        """多线程同时 flush 不崩溃"""
        for i in range(10):
            task = Task(action=f"任务{i}", text="", route="local", model_used="test")
            self.collector.record(task, _make_context())

        errors = []

        def do_flush():
            try:
                self.collector.flush()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=do_flush) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        from task_router.io_utils import read_jsonl
        disk = read_jsonl(self.collector.episodes_file)
        assert len(disk) == 10


# ─── 全局实例管理测试 ──────────────────────────────────────────────


class TestGlobalCollector:
    """全局 EpisodeCollector 实例管理"""

    def setup_method(self):
        reset_episode_collector()

    def teardown_method(self):
        reset_episode_collector()

    def test_singleton(self):
        """get_episode_collector 返回单例"""
        c1 = get_episode_collector(cache_dir=tempfile.mkdtemp())
        c2 = get_episode_collector()
        assert c1 is c2

    def test_reset(self):
        """reset_episode_collector 重置单例"""
        tmpdir = tempfile.mkdtemp()
        c1 = get_episode_collector(cache_dir=tmpdir)
        reset_episode_collector()
        c2 = get_episode_collector(cache_dir=tmpdir)
        assert c1 is not c2
