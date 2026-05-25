"""
成本计算、使用日志、统计报告、能力追踪器
"""

import os
import time
import logging
import threading
from typing import Any

from task_router.config import get_config
from task_router.io_utils import read_jsonl, append_jsonl

log = logging.getLogger("task_router")

_log_lock = threading.Lock()


# ─── PII 脱敏辅助 ──────────────────────────────────────────────


def _scrub_pii(text: str) -> str:
    """对日志中的文本进行 PII 脱敏（延迟导入避免循环依赖）"""
    try:
        from task_router.privacy import get_privacy_filter
        result = get_privacy_filter().anonymize(text)
        return result.text
    except Exception:
        return text


# ─── 成本计算 ──────────────────────────────────────────────────


def calc_cost(input_tokens: int, output_tokens: int) -> float:
    """计算云端 API 调用成本"""
    config = get_config()
    return (input_tokens / 1000 * config.cost_per_1k_input
            + output_tokens / 1000 * config.cost_per_1k_output)


def calc_savings(task: Any) -> float:
    """计算本地执行节约的成本"""
    if task.route != "local":
        return 0.0
    return round(calc_cost(task.tokens_input, task.tokens_output), 6)


# ─── 使用日志 ──────────────────────────────────────────────────


def log_usage(task: Any) -> None:
    """记录任务使用日志"""
    config = get_config()
    log_file = os.path.join(config.cache_dir, "usage.jsonl")
    entry = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "date": time.strftime("%Y-%m-%d"),
        "action": _scrub_pii(task.action[:100]),
        "route": task.route,
        "model": task.model_used,
        "tokens_input": task.tokens_input,
        "tokens_output": task.tokens_output,
        "cost_saved": task.cost_saved,
        "time_ms": task.time_ms,
    }
    with _log_lock:
        append_jsonl(log_file, entry, max_lines=10000)


# ─── 统计 ──────────────────────────────────────────────────────


def _classify_route(route: str) -> str:
    """路由分类: local / cache / cloud"""
    if route == "local":
        return "local"
    if "cache" in route:
        return "cache"
    return "cloud"


def show_usage_stats() -> str:
    """显示使用统计"""
    from task_router.models import circuit_breaker
    from task_router.task_router import cache

    config = get_config()
    entries = read_jsonl(os.path.join(config.cache_dir, "usage.jsonl"))
    if not entries:
        return "暂无使用记录"

    totals = {"local": 0, "cloud": 0, "cache": 0}
    total_input = total_output = 0
    total_saved = 0.0
    daily: dict[str, dict] = {}

    for e in entries:
        cat = _classify_route(e.get("route", ""))
        totals[cat] += 1
        total_input += e.get("tokens_input", 0)
        total_output += e.get("tokens_output", 0)
        total_saved += e.get("cost_saved", 0)

        day = e.get("date", e.get("time", "")[:10])
        if day:
            if day not in daily:
                daily[day] = {"local": 0, "cloud": 0, "cache": 0, "saved": 0.0}
            daily[day][cat] += 1
            daily[day]["saved"] += e.get("cost_saved", 0)

    cs = cache.stats()
    result = (
        f"TaskRouter 使用统计\n{'='*40}\n"
        f"本地调用: {totals['local']} 次\n云端调用: {totals['cloud']} 次\n缓存命中: {totals['cache']} 次\n"
        f"总输入 tokens: {total_input:,}\n总输出 tokens: {total_output:,}\n"
        f"理论最大节约: ${total_saved:.4f} (若全部走云端需支付的费用)\n"
    )
    if cs["total"] > 0:
        result += f"\n语义缓存:\n  缓存条目: {cs['total']}\n  缓存节约: ${cs['estimated_cost_saved']:.4f}\n"

    for day in sorted(daily, reverse=True)[:7]:
        d = daily[day]
        result += f"  {day}: {d['local']+d['cloud']+d['cache']}次 (本地{d['local']}/云端{d['cloud']}/缓存{d['cache']}) 省${d['saved']:.4f}\n"

    if circuit_breaker.state == "open":
        result += f"\n云端熔断器: 🔴 熔断中 ({circuit_breaker.remaining_seconds()}秒后恢复)\n"
    elif circuit_breaker.state == "half_open":
        result += "\n云端熔断器: 🟡 半开状态 (探测中)\n"
    elif circuit_breaker.failures > 0:
        result += "\n云端熔断器: 🟢 已恢复\n"

    return result


# ─── CapabilityTracker ──────────────────────────────────────────


class CapabilityTracker:
    """自适应阈值追踪器 — 按能力维度追踪成功率"""

    LOOKBACK = 20

    def __init__(self, cache_dir: str):
        self.data_file = os.path.join(cache_dir, "capability_tracker.jsonl")
        os.makedirs(cache_dir, exist_ok=True)
        self._default_threshold = get_config().base_threshold

    def record(self, capability: str, success: bool, score: float = 0.0, task_type: str = "") -> None:
        entry = {
            "capability": capability, "success": success, "score": score,
            "task_type": task_type, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        append_jsonl(self.data_file, entry)

    def get_success_rate(self, capability: str) -> float:
        records = self._load_recent(capability)
        if not records:
            return 0.8  # 默认
        return sum(1 for r in records if r.get("success")) / len(records)

    def get_adjusted_threshold(self, capability: str) -> float:
        rate = self.get_success_rate(capability)
        if rate >= 0.9:
            return self._default_threshold - 0.5
        elif rate >= 0.7:
            return self._default_threshold
        elif rate >= 0.5:
            return self._default_threshold + 0.5
        else:
            return self._default_threshold + 1.0

    def get_all_adjustments(self) -> dict:
        all_entries = read_jsonl(self.data_file)
        all_caps = {e.get("capability", "") for e in all_entries if e.get("capability")}
        result = {}
        for cap in sorted(all_caps):
            rate = self.get_success_rate(cap)
            threshold = self.get_adjusted_threshold(cap)
            diff = round(threshold - self._default_threshold, 1)
            result[cap] = {
                "success_rate": round(rate, 2), "threshold": threshold,
                "adjustment": f"{'+' if diff > 0 else ''}{diff}",
                "direction": "上调" if diff > 0 else "下调" if diff < 0 else "不变",
            }
        return result

    def _load_recent(self, capability: str, n: int = LOOKBACK) -> list[dict]:
        all_entries = read_jsonl(self.data_file)
        return [e for e in all_entries if e.get("capability") == capability][-n:]
