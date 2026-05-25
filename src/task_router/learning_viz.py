"""
学习闭环可视化 — 展示系统如何随时间进化

核心功能:
- 成本节约趋势图（ASCII 图表）
- 本地/云端路由比例变化
- 缓存命中率趋势
- 模型性能进化曲线
- 蒸馏学习进度

使用方式：
    python3 learning_viz.py                # 显示完整报告
    python3 learning_viz.py --cost         # 只看成本趋势
    python3 learning_viz.py --models       # 只看模型性能
    python3 learning_viz.py --days 30      # 最近30天
"""

import os
import json
import time
from collections import defaultdict
from datetime import datetime


# ─── 数据加载 ──────────────────────────────────────────────────

def load_usage_data(cache_dir: str, days: int = 30) -> list[dict]:
    """加载使用数据"""
    log_file = os.path.join(cache_dir, "usage.jsonl")
    entries = []
    if not os.path.exists(log_file):
        return entries

    cutoff = time.time() - days * 86400
    with open(log_file) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                ts = entry.get("timestamp", "")
                if ts:
                    # 解析时间戳
                    try:
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        if dt.timestamp() < cutoff:
                            continue
                    except ValueError:
                        pass
                entries.append(entry)
            except json.JSONDecodeError:
                pass
    return entries


def load_model_data(cache_dir: str) -> dict:
    """加载模型数据"""
    model_file = os.path.join(cache_dir, "model_registry.json")
    if not os.path.exists(model_file):
        return {}
    try:
        with open(model_file) as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}


def load_threshold_data(cache_dir: str) -> list[dict]:
    """加载自适应阈值数据"""
    threshold_file = os.path.join(cache_dir, "capability_tracker.jsonl")
    entries = []
    if not os.path.exists(threshold_file):
        return entries
    with open(threshold_file) as f:
        for line in f:
            try:
                entries.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                pass
    return entries


# ─── 统计计算 ──────────────────────────────────────────────────

def calculate_daily_stats(entries: list[dict]) -> dict:
    """按天统计"""
    daily = defaultdict(lambda: {
        "total": 0, "local": 0, "cloud": 0, "cache": 0,
        "cost_saved": 0.0, "tokens_input": 0, "tokens_output": 0,
        "duration_ms": 0,
    })

    for e in entries:
        date = e.get("date", e.get("timestamp", "")[:10])
        if not date:
            continue
        stats = daily[date]
        stats["total"] += 1

        route = e.get("route", "")
        if "cache" in route:
            stats["cache"] += 1
        elif "local" in route:
            stats["local"] += 1
        elif "cloud" in route:
            stats["cloud"] += 1

        stats["cost_saved"] += e.get("cost_saved", 0.0)
        stats["tokens_input"] += e.get("tokens_input", 0)
        stats["tokens_output"] += e.get("tokens_output", 0)
        stats["duration_ms"] += e.get("duration_ms", 0)

    return dict(sorted(daily.items()))


def calculate_cumulative_savings(daily_stats: dict) -> list[tuple]:
    """计算累计节约"""
    cumulative = []
    total = 0.0
    for date, stats in daily_stats.items():
        total += stats["cost_saved"]
        cumulative.append((date, total))
    return cumulative


# ─── ASCII 图表 ──────────────────────────────────────────────────

def ascii_bar_chart(data: dict, title: str, width: int = 40, value_fmt: str = ".2f") -> str:
    """生成 ASCII 柱状图"""
    if not data:
        return f"{title}\n  (无数据)"

    max_val = max(data.values()) if data.values() else 1
    lines = [f"\n  {title}", "  " + "─" * (width + 20)]

    for label, value in data.items():
        bar_len = int((value / max_val) * width) if max_val > 0 else 0
        bar = "█" * bar_len
        lines.append(f"  {label:>10s} │{bar:<{width}s} {value:{value_fmt}}")

    lines.append("  " + "─" * (width + 20))
    return "\n".join(lines)


def ascii_line_chart(values: list[float], title: str, width: int = 60, height: int = 10) -> str:
    """生成 ASCII 折线图"""
    if not values:
        return f"{title}\n  (无数据)"

    min_val = min(values)
    max_val = max(values)
    val_range = max_val - min_val if max_val != min_val else 1

    lines = [f"\n  {title}", "  " + "─" * (width + 10)]

    for row in range(height, -1, -1):
        threshold = min_val + (row / height) * val_range
        line = f"  {threshold:>8.2f} │"
        for v in values:
            scaled = (v - min_val) / val_range * height
            if scaled >= row:
                line += "█"
            else:
                line += " "
        lines.append(line)

    lines.append("  " + " " * 9 + "└" + "─" * width)
    return "\n".join(lines)


# ─── 报告生成 ──────────────────────────────────────────────────

def generate_report(cache_dir: str, days: int = 30) -> str:
    """生成完整的学习闭环报告"""
    entries = load_usage_data(cache_dir, days)
    model_data = load_model_data(cache_dir)
    threshold_data = load_threshold_data(cache_dir)

    if not entries:
        return "  (无使用数据，请先执行一些任务)"

    daily_stats = calculate_daily_stats(entries)
    cumulative = calculate_cumulative_savings(daily_stats)

    report = []
    report.append("\n" + "=" * 60)
    report.append("  TaskRouter 学习闭环报告")
    report.append(f"  时间范围: 最近 {days} 天 | 数据点: {len(entries)} 条")
    report.append("=" * 60)

    # ── 1. 总体统计 ──
    total_tasks = sum(s["total"] for s in daily_stats.values())
    total_local = sum(s["local"] for s in daily_stats.values())
    total_cloud = sum(s["cloud"] for s in daily_stats.values())
    total_cache = sum(s["cache"] for s in daily_stats.values())
    total_saved = sum(s["cost_saved"] for s in daily_stats.values())
    total_tokens_in = sum(s["tokens_input"] for s in daily_stats.values())
    total_tokens_out = sum(s["tokens_output"] for s in daily_stats.values())

    report.append("\n  ── 总体统计 ──")
    report.append(f"  总任务数: {total_tasks}")
    report.append(f"  本地执行: {total_local} ({total_local/total_tasks*100:.1f}%)" if total_tasks else "")
    report.append(f"  云端执行: {total_cloud} ({total_cloud/total_tasks*100:.1f}%)" if total_tasks else "")
    report.append(f"  缓存命中: {total_cache} ({total_cache/total_tasks*100:.1f}%)" if total_tasks else "")
    report.append(f"  累计节约: ${total_saved:.4f}")
    report.append(f"  Token 消耗: 输入 {total_tokens_in:,} + 输出 {total_tokens_out:,}")

    # ── 2. 路由分布 ──
    route_data = {
        "本地": total_local,
        "云端": total_cloud,
        "缓存": total_cache,
    }
    report.append(ascii_bar_chart(route_data, "路由分布"))

    # ── 3. 每日成本节约趋势 ──
    daily_savings = {d: s["cost_saved"] for d, s in daily_stats.items() if s["cost_saved"] > 0}
    if daily_savings:
        report.append(ascii_bar_chart(
            {d[-5:]: v for d, v in daily_savings.items()},
            "每日成本节约 ($)",
        ))

    # ── 4. 累计节约曲线 ──
    if cumulative:
        cum_values = [v for _, v in cumulative]
        report.append(ascii_line_chart(cum_values, "累计成本节约趋势 ($)"))

    # ── 5. 模型性能 ──
    if model_data:
        report.append("\n  ── 模型性能 ──")
        for name, info in sorted(model_data.items(),
                                  key=lambda x: x[1].get("success_rate", 0),
                                  reverse=True)[:10]:
            calls = info.get("total_calls", 0)
            rate = info.get("success_rate", 0) * 100
            latency = info.get("avg_latency_ms", 0)
            speed = info.get("tokens_per_second", 0)
            report.append(
                f"  {name:20s} | 调用: {calls:4d} | 成功率: {rate:5.1f}% | "
                f"延迟: {latency:6.0f}ms | 速度: {speed:5.1f} t/s"
            )

    # ── 6. 自适应阈值 ──
    if threshold_data:
        report.append("\n  ── 自适应阈值进化 ──")
        for entry in threshold_data[-5:]:
            cap = entry.get("capability", "?")
            threshold = entry.get("threshold", 0)
            success = entry.get("success_rate", 0) * 100
            count = entry.get("total", 0)
            report.append(f"  {cap:15s} | 阈值: {threshold:.2f} | 成功率: {success:.1f}% | 样本: {count}")

    # ── 7. 学习进度 ──
    report.append("\n  ── 学习闭环进度 ──")
    report.append(f"  蒸馏对数: {len(threshold_data)} 条")
    if total_tasks > 0:
        local_rate = (total_local + total_cache) / total_tasks * 100
        report.append(f"  本地化率: {local_rate:.1f}% (目标: >80%)")
        if local_rate >= 80:
            report.append("  状态: ✓ 已达到目标")
        else:
            report.append(f"  状态: 进化中 (还需 {80-local_rate:.1f}%)")

    report.append("\n" + "=" * 60)
    return "\n".join(report)


def generate_summary_json(cache_dir: str, days: int = 30) -> dict:
    """生成 JSON 格式的摘要（供 API 使用）"""
    entries = load_usage_data(cache_dir, days)
    daily_stats = calculate_daily_stats(entries)

    total_tasks = sum(s["total"] for s in daily_stats.values())
    total_local = sum(s["local"] for s in daily_stats.values())
    total_cloud = sum(s["cloud"] for s in daily_stats.values())
    total_cache = sum(s["cache"] for s in daily_stats.values())
    total_saved = sum(s["cost_saved"] for s in daily_stats.values())

    return {
        "period_days": days,
        "total_tasks": total_tasks,
        "local_tasks": total_local,
        "cloud_tasks": total_cloud,
        "cache_hits": total_cache,
        "local_rate": (total_local + total_cache) / total_tasks * 100 if total_tasks else 0,
        "total_cost_saved": total_saved,
        "daily_stats": {d: s for d, s in daily_stats.items()},
    }


# ─── CLI 入口 ──────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TaskRouter 学习闭环可视化")
    parser.add_argument("--days", type=int, default=30, help="时间范围（天）")
    parser.add_argument("--cost", action="store_true", help="只看成本趋势")
    parser.add_argument("--models", action="store_true", help="只看模型性能")
    parser.add_argument("--json", action="store_true", help="JSON 格式输出")
    parser.add_argument("--cache-dir", default=None, help="缓存目录")
    args = parser.parse_args()

    if args.cache_dir:
        cache_dir = args.cache_dir
    else:
        from task_router.config import get_config
        cache_dir = get_config().cache_dir

    if args.json:
        summary = generate_summary_json(cache_dir, args.days)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        report = generate_report(cache_dir, args.days)
        print(report)
