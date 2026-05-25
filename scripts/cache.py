"""
语义缓存 — Trigram Jaccard 模糊匹配 + TTL 过期（线程安全）
"""

import os
import re
import time
import unicodedata
import hashlib
import threading
from typing import Optional

from io_utils import read_jsonl, append_jsonl, write_jsonl


class SemanticCache:
    """语义缓存，支持精确/归一化/模糊三级匹配"""

    def __init__(self, cache_dir: str, max_entries: int = 1000, fuzzy_threshold: float = 0.85):
        self.cache_dir = cache_dir
        self.max_entries = max_entries
        self.fuzzy_threshold = fuzzy_threshold
        self.cache_file = os.path.join(cache_dir, "semantic_cache.jsonl")
        self._entries: list[dict] = []
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        raw = read_jsonl(self.cache_file)
        # 去重：同一 key 只保留最后一条（解决重启后旧条目残留）
        seen: dict[str, int] = {}
        for i, e in enumerate(raw):
            k = e.get("key", "")
            if k:
                seen[k] = i
        keep = set(seen.values())
        self._entries = [e for i, e in enumerate(raw) if i in keep]

    def _save(self) -> None:
        write_jsonl(self.cache_file, self._entries)

    def _append_entry(self, entry: dict) -> None:
        """追加单条记录到文件（避免全量重写）"""
        append_jsonl(self.cache_file, entry)

    @staticmethod
    def _normalize(text: str) -> str:
        """归一化：去空格、统一标点"""
        text = text.strip()
        text = "".join(c for c in text if not unicodedata.category(c).startswith("Z"))
        text = text.replace("，", ",").replace("。", ".").replace("；", ";")
        text = text.replace("：", ":").replace("！", "!").replace("？", "?")
        text = re.sub(r",+", ",", text)
        return text.lower()

    def _trigrams(self, text: str) -> set[str]:
        """生成字符 trigram 集合"""
        text = self._normalize(text)
        if len(text) < 3:
            return {text}
        return {text[i:i+3] for i in range(len(text) - 2)}

    def _jaccard(self, a: set, b: set) -> float:
        """Jaccard 相似度"""
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def _make_key(self, action: str, text: str) -> str:
        """生成缓存 key"""
        raw = f"{action}|{text}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _is_expired(self, entry: dict) -> bool:
        """检查缓存是否过期"""
        created_ts = entry.get("created_ts", 0)
        ttl_hours = entry.get("ttl_hours", 48)
        if created_ts == 0:
            return False
        return (time.time() - created_ts) > ttl_hours * 3600

    def get(self, action: str, text: str) -> Optional[dict]:
        """查找缓存（三级匹配，线程安全）"""
        key = self._make_key(action, text)
        query_tri = self._trigrams(f"{action}|{text}")
        query_norm = self._normalize(f"{action}|{text}")

        with self._lock:
            best_match = None
            best_score = 0.0

            for entry in self._entries:
                if self._is_expired(entry):
                    continue

                entry_key = entry.get("key", "")
                entry_norm = entry.get("normalized", "")
                entry_tri = set(entry.get("trigrams", []))

                # 级别 1：精确匹配
                if entry_key == key:
                    return entry

                # 级别 2：归一化匹配
                if entry_norm and entry_norm == query_norm:
                    return entry

                # 级别 3：模糊匹配
                if entry_tri:
                    score = self._jaccard(query_tri, entry_tri)
                    if score >= self.fuzzy_threshold and score > best_score:
                        best_score = score
                        best_match = entry

            return best_match

    def set(self, action: str, text: str, result: dict, ttl_hours: int = 48) -> None:
        """写入缓存（线程安全，I/O 在锁内保证一致性）"""
        key = self._make_key(action, text)
        combined = f"{action}|{text}"

        entry = {
            "key": key,
            "normalized": self._normalize(combined),
            "trigrams": list(self._trigrams(combined)),
            "action": action[:100],
            "text_preview": text[:100],
            "result": result,
            "created_ts": time.time(),
            "ttl_hours": ttl_hours,
        }

        with self._lock:
            # 移除旧条目
            self._entries = [e for e in self._entries if e.get("key") != key]
            self._entries.append(entry)

            # 淘汰旧条目
            if len(self._entries) > self.max_entries:
                self._entries = self._entries[-self.max_entries:]
                self._save()  # 淘汰后全量重写
            else:
                self._append_entry(entry)  # 追加写入

    def cleanup_expired(self) -> int:
        """清理过期条目（线程安全）"""
        with self._lock:
            before = len(self._entries)
            self._entries = [e for e in self._entries if not self._is_expired(e)]
            removed = before - len(self._entries)
            if removed > 0:
                self._save()
        return removed

    def stats(self) -> dict:
        """缓存统计（线程安全）"""
        with self._lock:
            total = len(self._entries)
            expired = sum(1 for e in self._entries if self._is_expired(e))
        return {
            "total": total,
            "active": total - expired,
            "expired": expired,
            "estimated_cost_saved": (total - expired) * 0.001,
        }
