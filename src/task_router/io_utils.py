"""
I/O 工具 — 共享的 JSONL 读写和文件操作

并发安全策略：
- append_jsonl: 单行追加在 POSIX 上是原子的（对小写入），轮转使用原子 rename
- write_jsonl:  使用 temp 文件 + os.rename() 保证原子性
"""

import os
import json
import tempfile
import threading

# 文件级锁：防止同一文件的轮转和写入竞争
_file_locks: dict[str, threading.Lock] = {}
_file_locks_global = threading.Lock()


def _get_file_lock(path: str) -> threading.Lock:
    """获取文件级别的锁"""
    with _file_locks_global:
        if path not in _file_locks:
            _file_locks[path] = threading.Lock()
        return _file_locks[path]


def _ensure_parent_dir(path: str) -> None:
    """确保父目录存在（安全处理空路径）"""
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)


def read_jsonl(path: str) -> list[dict]:
    """读取 JSONL 文件，跳过损坏行"""
    if not os.path.exists(path):
        return []
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def append_jsonl(path: str, entry: dict, max_lines: int = 0) -> None:
    """追加单条记录到 JSONL 文件。

    并发安全：
    - 单行追加在 POSIX 上对小写入是原子的
    - 轮转操作使用文件级锁 + 原子 rename

    参数:
        max_lines: 最大行数限制。超过时保留最新的一半。
                   0 表示不限制（向后兼容）。
    """
    _ensure_parent_dir(path)

    # 轮转检查
    if max_lines > 0 and os.path.exists(path):
        lock = _get_file_lock(path)
        with lock:
            try:
                with open(path, encoding="utf-8") as f:
                    line_count = sum(1 for _ in f)
                if line_count >= max_lines:
                    keep = max_lines // 2
                    with open(path, encoding="utf-8") as f:
                        lines = f.readlines()
                    # 原子写入：先写 temp 文件，再 rename
                    dir_name = os.path.dirname(path) or "."
                    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".jsonl.tmp")
                    try:
                        with os.fdopen(fd, "w", encoding="utf-8") as tmp:
                            tmp.writelines(lines[-keep:])
                        os.replace(tmp_path, path)  # 原子替换
                    except OSError:
                        # 清理 temp 文件
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass
            except OSError:
                pass  # 文件读取失败时继续追加

    # 追加单行（小写入在 POSIX 上是原子的）
    line = json.dumps(entry, ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


def write_jsonl(path: str, entries: list[dict]) -> None:
    """全量重写 JSONL 文件（原子操作）。

    使用 temp 文件 + os.replace() 保证写入过程不会产生半写文件。
    """
    _ensure_parent_dir(path)
    dir_name = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".jsonl.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        os.replace(tmp_path, path)  # 原子替换
    except OSError:
        # 清理 temp 文件
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def ensure_dir(path: str) -> None:
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)
