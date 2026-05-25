"""
参数修正应用器 — 将反思引擎的修正建议应用到路由参数

安全机制：
1. 置信度过滤：只应用 confidence >= 0.6 的修正
2. 幅度限制：单次阈值修正不超过 ±0.5
3. 回滚能力：保存历史参数，支持回滚
4. 审批模式：confidence < 0.8 的修正写入待审批队列
"""

import json
import logging
import os
import threading
import time
from typing import Any

from task_router.io_utils import read_jsonl, append_jsonl, write_jsonl

log = logging.getLogger(__name__)


# ─── 安全常量 ──────────────────────────────────────────────────

# 最低自动应用置信度
AUTO_APPLY_CONFIDENCE = 0.8

# 最低记录置信度（低于此值完全忽略）
MIN_CONFIDENCE = 0.6

# 阈值修正幅度限制
MAX_THRESHOLD_CHANGE = 0.5

# 阈值允许范围
THRESHOLD_MIN = 1.0
THRESHOLD_MAX = 8.0


# ─── 修正状态 ──────────────────────────────────────────────────

STATUS_APPLIED = "applied"
STATUS_PENDING = "pending"
STATUS_SKIPPED = "skipped"
STATUS_ROLLED_BACK = "rolled_back"


# ─── CorrectionApplier ──────────────────────────────────────────


class CorrectionApplier:
    """将修正建议应用到系统参数。

    使用方式：
        applier = CorrectionApplier(cache_dir="/path/to/cache")
        result = applier.apply_corrections(corrections)

    安全机制：
    - confidence >= 0.8 → 自动应用
    - 0.6 <= confidence < 0.8 → 写入待审批队列
    - confidence < 0.6 → 忽略
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.history_file = os.path.join(cache_dir, "correction_history.jsonl")
        self.pending_file = os.path.join(cache_dir, "corrections_pending.jsonl")
        self.a3m_weights_file = os.path.join(cache_dir, "a3m_weights.json")
        self.strategy_params_file = os.path.join(cache_dir, "strategy_params.json")
        self.routing_policies_file = os.path.join(cache_dir, "routing_policies.json")
        self._lock = threading.Lock()
        os.makedirs(cache_dir, exist_ok=True)

    def apply_corrections(self, corrections: list) -> dict:
        """应用修正建议列表。

        参数:
            corrections: Correction 对象或 dict 列表

        返回:
            {
                "applied": int,
                "pending": int,
                "skipped": int,
                "details": [{"correction_id": str, "status": str, "reason": str}]
            }
        """
        applied = 0
        pending = 0
        skipped = 0
        details = []

        for correction in corrections:
            c = correction.to_dict() if hasattr(correction, "to_dict") else correction
            cid = c.get("correction_id", "")
            confidence = c.get("confidence", 0.0)
            reason = c.get("reason", "")

            # 置信度过滤
            if confidence < MIN_CONFIDENCE:
                skipped += 1
                details.append({
                    "correction_id": cid,
                    "status": STATUS_SKIPPED,
                    "reason": f"置信度 {confidence:.0%} < {MIN_CONFIDENCE:.0%}",
                })
                self._log_correction(c, STATUS_SKIPPED)
                continue

            # 高置信度 → 自动应用
            if confidence >= AUTO_APPLY_CONFIDENCE:
                success = self._apply_single(c)
                if success:
                    applied += 1
                    details.append({
                        "correction_id": cid,
                        "status": STATUS_APPLIED,
                        "reason": reason,
                    })
                    self._log_correction(c, STATUS_APPLIED)
                else:
                    skipped += 1
                    details.append({
                        "correction_id": cid,
                        "status": STATUS_SKIPPED,
                        "reason": "应用失败",
                    })
                    self._log_correction(c, STATUS_SKIPPED)
            else:
                # 中等置信度 → 写入待审批队列
                pending += 1
                self._add_to_pending(c)
                details.append({
                    "correction_id": cid,
                    "status": STATUS_PENDING,
                    "reason": f"待审批（置信度 {confidence:.0%}）",
                })
                self._log_correction(c, STATUS_PENDING)

        return {
            "applied": applied,
            "pending": pending,
            "skipped": skipped,
            "details": details,
        }

    def _apply_single(self, correction: dict) -> bool:
        """应用单条修正。"""
        target = correction.get("target", "")
        parameter = correction.get("parameter", "")
        new_value = correction.get("new_value")

        if target == "threshold":
            return self._apply_threshold(parameter, new_value)
        elif target == "strategy_weight":
            return self._apply_strategy_weight(parameter, new_value)
        elif target == "routing_policy":
            return self._apply_routing_policy(parameter, new_value)
        else:
            log.warning("未知修正目标: %s", target)
            return False

    def _apply_threshold(self, parameter: str, new_value: Any) -> bool:
        """应用阈值修正 → 更新 a3m_weights.json。"""
        try:
            current = self._load_json(self.a3m_weights_file)
            old_threshold = current.get("base_threshold", 3.0)

            # new_value 是增量
            if isinstance(new_value, (int, float)):
                # 幅度限制
                change = max(-MAX_THRESHOLD_CHANGE, min(MAX_THRESHOLD_CHANGE, new_value))
                new_threshold = old_threshold + change
                new_threshold = max(THRESHOLD_MIN, min(THRESHOLD_MAX, new_threshold))
                current["base_threshold"] = round(new_threshold, 3)
            else:
                current["base_threshold"] = new_value

            self._save_json(self.a3m_weights_file, current)
            log.info("阈值修正: %.3f → %.3f", old_threshold, current["base_threshold"])
            return True
        except Exception as e:
            log.error("阈值修正失败: %s", e)
            return False

    def _apply_strategy_weight(self, parameter: str, new_value: Any) -> bool:
        """应用策略权重修正 → 更新 strategy_params.json。"""
        try:
            current = self._load_json(self.strategy_params_file)
            current[parameter] = new_value
            current["_last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            self._save_json(self.strategy_params_file, current)
            log.info("策略权重修正: %s = %s", parameter, new_value)
            return True
        except Exception as e:
            log.error("策略权重修正失败: %s", e)
            return False

    def _apply_routing_policy(self, parameter: str, new_value: Any) -> bool:
        """应用路由策略修正 → 更新 routing_policies.json。"""
        try:
            current = self._load_json(self.routing_policies_file)
            current[parameter] = new_value
            current["_last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            self._save_json(self.routing_policies_file, current)
            log.info("路由策略修正: %s = %s", parameter, new_value)
            return True
        except Exception as e:
            log.error("路由策略修正失败: %s", e)
            return False

    # ── 审批 ──

    def approve_pending(self, correction_id: str) -> bool:
        """审批通过一条待审批修正。"""
        with self._lock:
            pending = read_jsonl(self.pending_file)
            remaining = []
            target = None
            for p in pending:
                if p.get("correction_id") == correction_id:
                    target = p
                else:
                    remaining.append(p)

            if not target:
                return False

            write_jsonl(self.pending_file, remaining)
            success = self._apply_single(target)
            status = STATUS_APPLIED if success else STATUS_SKIPPED
            self._log_correction(target, status)
            return success

    def reject_pending(self, correction_id: str) -> bool:
        """拒绝一条待审批修正。"""
        with self._lock:
            pending = read_jsonl(self.pending_file)
            remaining = []
            found = False
            for p in pending:
                if p.get("correction_id") == correction_id:
                    found = True
                    self._log_correction(p, STATUS_SKIPPED)
                else:
                    remaining.append(p)

            if found:
                write_jsonl(self.pending_file, remaining)
            return found

    def get_pending(self) -> list[dict]:
        """获取待审批的修正列表。"""
        return read_jsonl(self.pending_file)

    # ── 回滚 ──

    def rollback(self, correction_id: str) -> bool:
        """回滚指定修正（需要历史记录中有 old_value）。"""
        history = read_jsonl(self.history_file)
        target = None
        for h in history:
            if h.get("correction_id") == correction_id and h.get("status") == STATUS_APPLIED:
                target = h
                break

        if not target:
            return False

        # 构造反向修正
        reverse = dict(target)
        reverse["new_value"] = target.get("old_value")
        reverse["old_value"] = target.get("new_value")
        reverse["reason"] = f"回滚修正 {correction_id}"

        success = self._apply_single(reverse)
        if success:
            self._log_correction(reverse, STATUS_ROLLED_BACK)
        return success

    # ── 持久化 ──

    def _add_to_pending(self, correction: dict) -> None:
        """添加到待审批队列。"""
        with self._lock:
            append_jsonl(self.pending_file, correction)

    def _log_correction(self, correction: dict, status: str) -> None:
        """记录修正历史。"""
        entry = dict(correction)
        entry["status"] = status
        entry["applied_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        append_jsonl(self.history_file, entry)

    def _load_json(self, path: str) -> dict:
        """加载 JSON 文件。"""
        if not os.path.exists(path):
            return {}
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_json(self, path: str, data: dict) -> None:
        """保存 JSON 文件。"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ── 查询 ──

    def get_history(self, n: int = 20) -> list[dict]:
        """获取最近 N 条修正历史。"""
        history = read_jsonl(self.history_file)
        return history[-n:]

    def get_stats(self) -> dict:
        """获取修正统计。"""
        history = read_jsonl(self.history_file)
        by_status = {}
        for h in history:
            s = h.get("status", "unknown")
            by_status[s] = by_status.get(s, 0) + 1
        return {
            "total": len(history),
            "by_status": by_status,
            "pending": len(self.get_pending()),
        }
