"""
I/O 工具 — 共享的 JSONL 读写和文件操作
"""

import os
import json


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

    参数:
        max_lines: 最大行数限制。超过时保留最新的一半。
                   0 表示不限制（向后兼容）。
    """
    _ensure_parent_dir(path)
    # 轮转检查
    if max_lines > 0 and os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                line_count = sum(1 for _ in f)
            if line_count >= max_lines:
                # 保留最新的一半
                keep = max_lines // 2
                with open(path, encoding="utf-8") as f:
                    lines = f.readlines()
                with open(path, "w", encoding="utf-8") as f:
                    f.writelines(lines[-keep:])
        except OSError:
            pass  # 文件读取失败时继续追加
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def write_jsonl(path: str, entries: list[dict]) -> None:
    """全量重写 JSONL 文件"""
    _ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def ensure_dir(path: str) -> None:
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)
