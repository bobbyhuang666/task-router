"""
io_utils 测试

测试覆盖:
- append_jsonl: 基础追加、轮转、max_lines
- read_jsonl: 读取、损坏行跳过
- write_jsonl: 全量重写
"""

import os
import tempfile

import pytest

from task_router.io_utils import append_jsonl, read_jsonl, write_jsonl


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


class TestAppendJsonl:
    def test_basic_append(self, tmp_dir):
        path = os.path.join(tmp_dir, "test.jsonl")
        append_jsonl(path, {"a": 1})
        append_jsonl(path, {"b": 2})
        lines = read_jsonl(path)
        assert len(lines) == 2
        assert lines[0] == {"a": 1}
        assert lines[1] == {"b": 2}

    def test_creates_parent_dir(self, tmp_dir):
        path = os.path.join(tmp_dir, "sub", "dir", "test.jsonl")
        append_jsonl(path, {"x": 1})
        assert os.path.exists(path)

    def test_no_rotation_when_max_lines_zero(self, tmp_dir):
        path = os.path.join(tmp_dir, "test.jsonl")
        for i in range(100):
            append_jsonl(path, {"i": i}, max_lines=0)
        lines = read_jsonl(path)
        assert len(lines) == 100

    def test_rotation_triggers_at_max_lines(self, tmp_dir):
        path = os.path.join(tmp_dir, "test.jsonl")
        # 写入 max_lines + 1 条（第 21 条触发轮转）
        for i in range(21):
            append_jsonl(path, {"i": i}, max_lines=20)
        lines = read_jsonl(path)
        # 轮转后保留最新的一半（10条）+ 刚追加的 1 条 = 11
        assert len(lines) == 11
        # 保留的是最新的数据
        assert lines[-1]["i"] == 20

    def test_rotation_preserves_newest(self, tmp_dir):
        path = os.path.join(tmp_dir, "test.jsonl")
        for i in range(25):
            append_jsonl(path, {"i": i}, max_lines=10)
        lines = read_jsonl(path)
        assert len(lines) <= 10
        assert lines[-1]["i"] == 24


class TestReadJsonl:
    def test_empty_file(self, tmp_dir):
        path = os.path.join(tmp_dir, "empty.jsonl")
        assert read_jsonl(path) == []

    def test_nonexistent_file(self):
        assert read_jsonl("/nonexistent/file.jsonl") == []

    def test_skips_corrupted_lines(self, tmp_dir):
        path = os.path.join(tmp_dir, "bad.jsonl")
        with open(path, "w") as f:
            f.write('{"valid": 1}\n')
            f.write('not json\n')
            f.write('{"valid": 2}\n')
        lines = read_jsonl(path)
        assert len(lines) == 2
        assert lines[0] == {"valid": 1}

    def test_skips_empty_lines(self, tmp_dir):
        path = os.path.join(tmp_dir, "sparse.jsonl")
        with open(path, "w") as f:
            f.write('{"a": 1}\n\n{"b": 2}\n')
        lines = read_jsonl(path)
        assert len(lines) == 2


class TestWriteJsonl:
    def test_overwrites_file(self, tmp_dir):
        path = os.path.join(tmp_dir, "test.jsonl")
        write_jsonl(path, [{"a": 1}, {"b": 2}])
        write_jsonl(path, [{"c": 3}])
        lines = read_jsonl(path)
        assert len(lines) == 1
        assert lines[0] == {"c": 3}

    def test_empty_list(self, tmp_dir):
        path = os.path.join(tmp_dir, "test.jsonl")
        write_jsonl(path, [{"a": 1}])
        write_jsonl(path, [])
        assert os.path.exists(path)
        assert read_jsonl(path) == []
