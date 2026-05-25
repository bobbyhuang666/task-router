"""
Self-Reflective Routing 反思引擎

三层反思框架：
1. RouteAnalyzer:   路由决策反思 — "选对了吗？"
2. StrategyReflector: 策略选择反思 — "token 浪费了吗？"
3. JointReflector:   联合反思 — "(route, strategy) 配对最优吗？"

反思产出：
- ReflectionReport: 人类可读的分析报告
- Correction 列表: 参数修正建议（供 CorrectionApplier 消费）
"""

import json
import logging
import os
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

from task_router.io_utils import read_jsonl, append_jsonl
from task_router.episode_collector import EpisodeCollector
from task_router.quality_judge import (
    QualityJudge,
)

log = logging.getLogger(__name__)


# ─── 数据结构 ──────────────────────────────────────────────────


@dataclass
class Correction:
    """一次参数修正建议"""

    correction_id: str = ""
    timestamp: str = ""

    # 触发来源
    trigger_episode_ids: list[str] = field(default_factory=list)

    # 修正目标
    target: str = ""       # "threshold" / "strategy_weight" / "routing_policy"
    parameter: str = ""    # 具体参数名
    old_value: Any = None
    new_value: Any = None

    # 修正依据
    reason: str = ""
    confidence: float = 0.0   # [0, 1]
    expected_impact: str = ""
    evidence_count: int = 0

    def __post_init__(self) -> None:
        if not self.correction_id:
            import hashlib
            raw = f"{self.target}{self.parameter}{time.time()}"
            self.correction_id = hashlib.md5(raw.encode()).hexdigest()[:12]
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

    def to_dict(self) -> dict:
        return {
            "correction_id": self.correction_id,
            "timestamp": self.timestamp,
            "trigger_episode_ids": self.trigger_episode_ids,
            "target": self.target,
            "parameter": self.parameter,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "reason": self.reason,
            "confidence": self.confidence,
            "expected_impact": self.expected_impact,
            "evidence_count": self.evidence_count,
        }


@dataclass
class ReflectionReport:
    """反思分析报告"""

    timestamp: str = ""
    episodes_analyzed: int = 0

    # 路由分析
    routing_accuracy: float = 0.0
    over_escalation_rate: float = 0.0
    under_escalation_rate: float = 0.0
    routing_errors_by_type: dict = field(default_factory=dict)

    # 策略分析
    avg_token_waste_ratio: float = 0.0
    strategy_errors: list = field(default_factory=list)

    # 联合分析
    joint_recommendations: list = field(default_factory=list)

    # 修正建议
    corrections: list = field(default_factory=list)

    # 人类可读摘要
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "episodes_analyzed": self.episodes_analyzed,
            "routing_accuracy": round(self.routing_accuracy, 3),
            "over_escalation_rate": round(self.over_escalation_rate, 3),
            "under_escalation_rate": round(self.under_escalation_rate, 3),
            "routing_errors_by_type": self.routing_errors_by_type,
            "avg_token_waste_ratio": round(self.avg_token_waste_ratio, 3),
            "strategy_errors": self.strategy_errors,
            "joint_recommendations": self.joint_recommendations,
            "corrections": [c.to_dict() if isinstance(c, Correction) else c
                           for c in self.corrections],
            "summary": self.summary,
        }


# ─── 第一层：路由反思 ──────────────────────────────────────────


class RouteAnalyzer:
    """路由决策反思 — 分析哪些任务路由选错了。

    分析维度：
    - over_escalated: 该走本地但走了云端（浪费钱）
    - under_escalated: 该走云端但走了本地（质量差）
    - optimal: 路由正确
    """

    def analyze(self, judged_episodes: list[dict]) -> dict:
        """分析路由决策错误。

        参数:
            judged_episodes: 包含 quality_scores 的 episode 列表

        返回:
            {
                "total": int,
                "correct": int,
                "over_escalated": int,
                "under_escalated": int,
                "accuracy": float,
                "by_task_type": {task_type: {error_type: count}},
                "over_escalated_episodes": [episode_ids],
                "under_escalated_episodes": [episode_ids],
            }
        """
        if not judged_episodes:
            return {"total": 0, "correct": 0, "accuracy": 0.0, "by_task_type": {}}

        total = len(judged_episodes)
        correct = 0
        over_escalated = 0
        under_escalated = 0
        by_task_type: dict[str, dict[str, int]] = {}
        over_ids = []
        under_ids = []

        for ep in judged_episodes:
            error = ep.get("routing_error", "none")
            task_type = ep.get("task_type", "unknown")
            episode_id = ep.get("episode_id", "")

            if task_type not in by_task_type:
                by_task_type[task_type] = {"correct": 0, "over_escalated": 0,
                                           "under_escalated": 0, "total": 0}
            by_task_type[task_type]["total"] += 1

            if error == "none" or error == "":
                correct += 1
                by_task_type[task_type]["correct"] += 1
            elif error == "over_escalated":
                over_escalated += 1
                by_task_type[task_type]["over_escalated"] += 1
                over_ids.append(episode_id)
            elif error == "under_escalated":
                under_escalated += 1
                by_task_type[task_type]["under_escalated"] += 1
                under_ids.append(episode_id)
            else:
                # wrong_strategy, token_waste 等也计入正确（路由层面）
                correct += 1
                by_task_type[task_type]["correct"] += 1

        return {
            "total": total,
            "correct": correct,
            "over_escalated": over_escalated,
            "under_escalated": under_escalated,
            "accuracy": correct / total if total > 0 else 0.0,
            "by_task_type": by_task_type,
            "over_escalated_episodes": over_ids,
            "under_escalated_episodes": under_ids,
        }


# ─── 第二层：策略反思 ──────────────────────────────────────────


class StrategyReflector:
    """策略选择反思 — 分析哪些策略选择浪费了 token。

    分析维度：
    - token_waste: 策略消耗了过多 token（如用 CoT 跑简单分类）
    - under_reasoning: 策略过于简单（如用 direct 跑复杂推理）
    """

    # 策略 token 倍数参考值（来自 reasoning.py）
    TOKEN_MULTIPLIER = {
        "direct": 1.0,
        "cod": 0.3,
        "cot": 1.8,
        "few_shot": 1.5,
        "structured": 1.3,
    }

    def analyze(self, judged_episodes: list[dict]) -> dict:
        """分析策略选择的 token 浪费。

        返回:
            {
                "total": int,
                "waste_count": int,
                "avg_waste_ratio": float,
                "by_task_type": {task_type: {waste: count, optimal: count, total: count}},
                "waste_details": [{episode_id, task_type, used_strategy, optimal_strategy, waste_ratio}],
            }
        """
        if not judged_episodes:
            return {"total": 0, "waste_count": 0, "avg_waste_ratio": 0.0, "by_task_type": {}}

        total = 0
        waste_count = 0
        waste_ratios = []
        by_task_type: dict[str, dict[str, int]] = {}
        waste_details = []

        for ep in judged_episodes:
            used_strategy = ep.get("strategy", "direct")
            optimal_strategy = ep.get("optimal_strategy", "")
            task_type = ep.get("task_type", "unknown")

            if not optimal_strategy or optimal_strategy == used_strategy:
                continue

            total += 1
            if task_type not in by_task_type:
                by_task_type[task_type] = {"waste": 0, "optimal": 0, "total": 0}
            by_task_type[task_type]["total"] += 1

            used_tokens = self.TOKEN_MULTIPLIER.get(used_strategy, 1.0)
            optimal_tokens = self.TOKEN_MULTIPLIER.get(optimal_strategy, 1.0)

            if used_tokens > optimal_tokens * 1.3:  # 超过 30% 视为浪费
                waste_ratio = (used_tokens - optimal_tokens) / used_tokens
                waste_ratios.append(waste_ratio)
                waste_count += 1
                by_task_type[task_type]["waste"] += 1
                waste_details.append({
                    "episode_id": ep.get("episode_id", ""),
                    "task_type": task_type,
                    "used_strategy": used_strategy,
                    "optimal_strategy": optimal_strategy,
                    "waste_ratio": round(waste_ratio, 2),
                    "used_multiplier": used_tokens,
                    "optimal_multiplier": optimal_tokens,
                })
            else:
                by_task_type[task_type]["optimal"] += 1

        return {
            "total": total,
            "waste_count": waste_count,
            "avg_waste_ratio": round(
                sum(waste_ratios) / len(waste_ratios) if waste_ratios else 0.0, 3
            ),
            "by_task_type": by_task_type,
            "waste_details": waste_details[:20],  # 最多返回 20 条详情
        }


# ─── 第三层：联合反思 ──────────────────────────────────────────


class JointReflector:
    """联合反思 — 分析 (route, strategy) 配对效率。

    核心问题：是否存在某个 (local, cod) 配对，
    比当前常用的 (local, cot) 效果一样好但 token 消耗少 50%？
    """

    def analyze(self, judged_episodes: list[dict]) -> dict:
        """分析 (route, strategy) 联合配对的效率。

        返回:
            {
                "pair_stats": {
                    "local:direct": {"count": int, "avg_quality": float, "avg_tokens": float},
                    ...
                },
                "recommendations": [{
                    "task_type": str,
                    "current_pair": "local:cot",
                    "recommended_pair": "local:cod",
                    "quality_delta": float,
                    "token_savings": float,
                    "reason": str,
                }],
            }
        """
        if not judged_episodes:
            return {"pair_stats": {}, "recommendations": []}

        # 按 (task_type, route, strategy) 聚合
        pair_stats: dict[str, dict] = {}
        task_type_pairs: dict[str, dict[str, list]] = {}  # task_type → pair_key → [episodes]

        for ep in judged_episodes:
            route = ep.get("route", "local")
            strategy = ep.get("strategy", "direct")
            task_type = ep.get("task_type", "unknown")

            # 归一化 route
            if "cache" in route:
                route_key = "cache"
            elif "cloud" in route or "cascade" in route:
                route_key = "cloud"
            else:
                route_key = "local"

            pair_key = f"{route_key}:{strategy}"

            if pair_key not in pair_stats:
                pair_stats[pair_key] = {
                    "count": 0, "quality_sum": 0.0, "token_sum": 0,
                    "success_count": 0,
                }
            pair_stats[pair_key]["count"] += 1

            # 计算质量分
            qs = ep.get("quality_scores", {})
            quality = 0.0
            if isinstance(qs, dict) and qs:
                quality = (
                    qs.get("relevance", 0) * 0.20
                    + qs.get("completeness", 0) * 0.20
                    + qs.get("accuracy", 0) * 0.25
                    + qs.get("efficiency", 0) * 0.15
                    + qs.get("correctness", 0) * 0.20
                ) / 10.0
            pair_stats[pair_key]["quality_sum"] += quality

            tokens = ep.get("tokens_input", 0) + ep.get("tokens_output", 0)
            pair_stats[pair_key]["token_sum"] += tokens

            if quality >= 0.6:
                pair_stats[pair_key]["success_count"] += 1

            # 按 task_type 聚合
            if task_type not in task_type_pairs:
                task_type_pairs[task_type] = {}
            if pair_key not in task_type_pairs[task_type]:
                task_type_pairs[task_type][pair_key] = []
            task_type_pairs[task_type][pair_key].append({
                "quality": quality,
                "tokens": tokens,
                "episode_id": ep.get("episode_id", ""),
            })

        # 计算平均值
        for pair_key, stats in pair_stats.items():
            count = stats["count"]
            stats["avg_quality"] = round(stats["quality_sum"] / count, 3) if count > 0 else 0
            stats["avg_tokens"] = round(stats["token_sum"] / count, 1) if count > 0 else 0
            stats["success_rate"] = round(stats["success_count"] / count, 3) if count > 0 else 0
            del stats["quality_sum"]
            del stats["token_sum"]
            del stats["success_count"]

        # 生成配对建议
        recommendations = self._generate_recommendations(task_type_pairs, pair_stats)

        return {
            "pair_stats": pair_stats,
            "recommendations": recommendations,
        }

    def _generate_recommendations(
        self,
        task_type_pairs: dict[str, dict[str, list]],
        pair_stats: dict[str, dict],
    ) -> list[dict]:
        """为每种 task_type 找最优配对建议。"""
        recommendations = []

        for task_type, pairs in task_type_pairs.items():
            if len(pairs) < 2:
                continue

            # 找最常用的配对和潜在更优配对
            pair_summaries = {}
            for pair_key, episodes in pairs.items():
                if len(episodes) < 2:
                    continue
                avg_quality = sum(e["quality"] for e in episodes) / len(episodes)
                avg_tokens = sum(e["tokens"] for e in episodes) / len(episodes)
                pair_summaries[pair_key] = {
                    "avg_quality": avg_quality,
                    "avg_tokens": avg_tokens,
                    "count": len(episodes),
                }

            if len(pair_summaries) < 2:
                continue

            # 按使用频率排序
            sorted_pairs = sorted(pair_summaries.items(), key=lambda x: x[1]["count"], reverse=True)
            current_key, current = sorted_pairs[0]

            # 找质量不低于当前、但 token 更少的配对
            best_alt = None
            best_savings = 0.0

            for alt_key, alt in sorted_pairs[1:]:
                quality_delta = alt["avg_quality"] - current["avg_quality"]
                if quality_delta >= -0.05:  # 质量损失不超过 5%
                    token_savings = 1.0 - (alt["avg_tokens"] / max(current["avg_tokens"], 1))
                    if token_savings > best_savings and token_savings > 0.1:  # 至少省 10%
                        best_savings = token_savings
                        best_alt = (alt_key, alt)

            if best_alt:
                alt_key, alt = best_alt
                recommendations.append({
                    "task_type": task_type,
                    "current_pair": current_key,
                    "recommended_pair": alt_key,
                    "current_quality": round(current["avg_quality"], 3),
                    "recommended_quality": round(alt["avg_quality"], 3),
                    "quality_delta": round(alt["avg_quality"] - current["avg_quality"], 3),
                    "token_savings": round(best_savings, 3),
                    "evidence_count": alt["count"],
                    "reason": (f"使用 {alt_key} 配对，质量差异 "
                              f"{alt['avg_quality'] - current['avg_quality']:+.2f}，"
                              f"token 节省 {best_savings:.0%}"),
                })

        return recommendations


# ─── 修正生成器 ──────────────────────────────────────────────────


class CorrectionGenerator:
    """根据反思分析结果生成参数修正建议。

    修正类型：
    1. threshold: 调整 base_threshold（更激进/保守地路由到本地）
    2. strategy_weight: 调整策略选择偏好
    3. routing_policy: 新增硬规则（如"法律类任务强制云端"）
    """

    # 最小证据数量（低于此数不生成修正）
    MIN_EVIDENCE = 3

    def generate(
        self,
        route_analysis: dict,
        strategy_analysis: dict,
        joint_analysis: dict,
    ) -> list[Correction]:
        """根据三层分析结果生成修正建议。"""
        corrections = []

        corrections.extend(self._threshold_corrections(route_analysis))
        corrections.extend(self._strategy_corrections(strategy_analysis))
        corrections.extend(self._joint_corrections(joint_analysis))

        return corrections

    def _threshold_corrections(self, route_analysis: dict) -> list[Correction]:
        """根据路由分析生成阈值修正。"""
        corrections = []
        over = route_analysis.get("over_escalated", 0)
        under = route_analysis.get("under_escalated", 0)
        total = route_analysis.get("total", 0)

        if total < self.MIN_EVIDENCE:
            return corrections

        # 过度升级过多 → 提高阈值（让更多任务走本地）
        if over > under and over >= self.MIN_EVIDENCE:
            rate = over / total
            if rate > 0.15:  # 超过 15% 过度升级
                adjustment = min(0.5, rate * 2.0)  # 最多调 0.5
                corrections.append(Correction(
                    trigger_episode_ids=route_analysis.get("over_escalated_episodes", [])[:5],
                    target="threshold",
                    parameter="base_threshold",
                    old_value=None,  # 由 applier 读取当前值
                    new_value=round(adjustment, 2),  # 增量（正=提高=更多本地）
                    reason=f"过度升级率 {rate:.0%}（{over}/{total}），建议提高阈值让更多任务走本地",
                    confidence=min(0.9, 0.5 + rate),
                    expected_impact=f"预计减少 {rate:.0%} 的不必要云端调用",
                    evidence_count=over,
                ))

        # 升级不足过多 → 降低阈值（让更多任务走云端）
        if under > over and under >= self.MIN_EVIDENCE:
            rate = under / total
            if rate > 0.15:
                adjustment = -min(0.5, rate * 2.0)
                corrections.append(Correction(
                    trigger_episode_ids=route_analysis.get("under_escalated_episodes", [])[:5],
                    target="threshold",
                    parameter="base_threshold",
                    old_value=None,
                    new_value=round(adjustment, 2),  # 负增量=降低=更多云端
                    reason=f"升级不足率 {rate:.0%}（{under}/{total}），建议降低阈值让更多任务走云端",
                    confidence=min(0.9, 0.5 + rate),
                    expected_impact=f"预计提升 {rate:.0%} 的路由质量",
                    evidence_count=under,
                ))

        # 按任务类型分析
        by_type = route_analysis.get("by_task_type", {})
        for task_type, stats in by_type.items():
            type_total = stats.get("total", 0)
            if type_total < self.MIN_EVIDENCE:
                continue

            type_over = stats.get("over_escalated", 0)
            type_under = stats.get("under_escalated", 0)

            # 特定任务类型过度升级严重 → 添加路由策略
            if type_over >= self.MIN_EVIDENCE and type_over > type_under:
                corrections.append(Correction(
                    target="routing_policy",
                    parameter=f"task_type.{task_type}.prefer_route",
                    old_value="auto",
                    new_value="local",
                    reason=f"任务类型 {task_type} 过度升级率高（{type_over}/{type_total}），建议优先本地",
                    confidence=min(0.85, 0.5 + type_over / type_total),
                    expected_impact=f"任务类型 {task_type} 更多走本地",
                    evidence_count=type_over,
                ))

        return corrections

    def _strategy_corrections(self, strategy_analysis: dict) -> list[Correction]:
        """根据策略分析生成策略权重修正。"""
        corrections = []
        waste_count = strategy_analysis.get("waste_count", 0)
        total = strategy_analysis.get("total", 0)
        avg_waste = strategy_analysis.get("avg_waste_ratio", 0.0)

        if total < self.MIN_EVIDENCE:
            return corrections

        # 全局策略浪费
        if waste_count >= self.MIN_EVIDENCE and avg_waste > 0.2:
            corrections.append(Correction(
                target="strategy_weight",
                parameter="prefer_lightweight_strategy",
                old_value=False,
                new_value=True,
                reason=f"策略浪费率 {avg_waste:.0%}（{waste_count}/{total}），建议偏好轻量策略",
                confidence=min(0.8, 0.4 + avg_waste),
                expected_impact=f"预计节省 {avg_waste:.0%} token",
                evidence_count=waste_count,
            ))

        # 按任务类型分析浪费
        by_type = strategy_analysis.get("by_task_type", {})
        for task_type, stats in by_type.items():
            type_waste = stats.get("waste", 0)
            type_total = stats.get("total", 0)
            if type_total < self.MIN_EVIDENCE:
                continue
            if type_waste >= self.MIN_EVIDENCE and type_waste > type_total * 0.3:
                corrections.append(Correction(
                    target="strategy_weight",
                    parameter=f"task_type.{task_type}.preferred_strategy",
                    old_value="auto",
                    new_value="cod",  # 偏好更轻量的策略
                    reason=f"任务类型 {task_type} 策略浪费严重（{type_waste}/{type_total}），建议偏好 CoD",
                    confidence=min(0.8, 0.4 + type_waste / type_total),
                    expected_impact=f"任务类型 {task_type} 策略更轻量",
                    evidence_count=type_waste,
                ))

        return corrections

    def _joint_corrections(self, joint_analysis: dict) -> list[Correction]:
        """根据联合分析生成配对修正。"""
        corrections = []
        recommendations = joint_analysis.get("recommendations", [])

        for rec in recommendations:
            if rec.get("evidence_count", 0) < self.MIN_EVIDENCE:
                continue
            if rec.get("token_savings", 0) < 0.15:  # 至少省 15%
                continue

            task_type = rec.get("task_type", "")
            corrections.append(Correction(
                target="routing_policy",
                parameter=f"task_type.{task_type}.preferred_pair",
                old_value=rec.get("current_pair", ""),
                new_value=rec.get("recommended_pair", ""),
                reason=rec.get("reason", ""),
                confidence=min(0.85, 0.5 + rec.get("token_savings", 0)),
                expected_impact=f"预计节省 {rec.get('token_savings', 0):.0%} token",
                evidence_count=rec.get("evidence_count", 0),
            ))

        return corrections


# ─── 反思引擎主类 ──────────────────────────────────────────────────


class ReflectionEngine:
    """自反思路由引擎 — SRR 的核心编排器。

    使用方式：
        engine = ReflectionEngine(cache_dir="/path/to/cache")
        report = engine.reflect(n_episodes=100)
        print(report.summary)
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.collector = EpisodeCollector(cache_dir)
        self.judge = QualityJudge(cache_dir)
        self.route_analyzer = RouteAnalyzer()
        self.strategy_reflector = StrategyReflector()
        self.joint_reflector = JointReflector()
        self.correction_generator = CorrectionGenerator()
        self.reports_file = os.path.join(cache_dir, "reflection_reports.jsonl")
        self.insights_file = os.path.join(cache_dir, "reflection_insights.json")
        self._lock = threading.Lock()

    def reflect(
        self,
        n_episodes: int = 100,
        episodes: Optional[list[dict]] = None,
    ) -> ReflectionReport:
        """运行一次完整反思。

        步骤：
        1. 获取最近 N 条 episode（或使用传入的 episodes）
        2. 对未评估的 episode 调用 QualityJudge
        3. 合并 episode + quality_scores
        4. 运行三层分析
        5. 生成修正建议
        6. 写入报告

        参数:
            n_episodes: 分析最近 N 条 episode
            episodes: 直接传入 episode 列表（测试时用）

        返回:
            ReflectionReport
        """
        # 1. 获取 episodes
        if episodes is None:
            episodes = self.collector.get_recent(n_episodes)

        if not episodes:
            return ReflectionReport(
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                summary="没有可分析的 episode",
            )

        # 2. 评估未评估的 episodes（跳过已有 quality_scores 的）
        needs_judging = [ep for ep in episodes if not ep.get("quality_scores")]
        if needs_judging:
            self.judge.judge_batch(needs_judging)

        # 3. 合并 episode + quality_scores
        judged_episodes = self._merge_judgments(episodes)

        # 4. 三层分析
        route_result = self.route_analyzer.analyze(judged_episodes)
        strategy_result = self.strategy_reflector.analyze(judged_episodes)
        joint_result = self.joint_reflector.analyze(judged_episodes)

        # 5. 生成修正建议
        corrections = self.correction_generator.generate(
            route_result, strategy_result, joint_result
        )

        # 6. 构建报告
        report = self._build_report(
            judged_episodes, route_result, strategy_result, joint_result, corrections
        )

        # 7. 持久化
        self._save_report(report)

        return report

    def _merge_judgments(self, episodes: list[dict]) -> list[dict]:
        """将 episode 与 quality_scores 合并。

        如果 episode 已有 quality_scores（如从外部传入的预评估数据），
        保留原有数据不覆盖。
        """
        merged = []
        for ep in episodes:
            ep_copy = dict(ep)
            ep_id = ep.get("episode_id", "")

            # 如果已有 quality_scores，保留原有数据
            if ep_copy.get("quality_scores") and ep_copy.get("routing_error"):
                merged.append(ep_copy)
                continue

            scores = self.judge.get_judgment(ep_id)
            if scores:
                ep_copy["quality_scores"] = scores.to_dict()
                # 仅在 episode 未预设 routing_error 时覆盖
                if not ep_copy.get("routing_error"):
                    ep_copy["routing_error"] = scores.routing_error
                if not ep_copy.get("optimal_route"):
                    ep_copy["optimal_route"] = scores.optimal_route
                if not ep_copy.get("optimal_strategy"):
                    ep_copy["optimal_strategy"] = scores.optimal_strategy
            merged.append(ep_copy)
        return merged

    def _build_report(
        self,
        judged_episodes: list[dict],
        route_result: dict,
        strategy_result: dict,
        joint_result: dict,
        corrections: list[Correction],
    ) -> ReflectionReport:
        """构建反思报告。"""
        report = ReflectionReport(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            episodes_analyzed=len(judged_episodes),
            routing_accuracy=route_result.get("accuracy", 0.0),
            over_escalation_rate=(
                route_result.get("over_escalated", 0) / max(route_result.get("total", 1), 1)
            ),
            under_escalation_rate=(
                route_result.get("under_escalated", 0) / max(route_result.get("total", 1), 1)
            ),
            routing_errors_by_type=route_result.get("by_task_type", {}),
            avg_token_waste_ratio=strategy_result.get("avg_waste_ratio", 0.0),
            strategy_errors=strategy_result.get("waste_details", []),
            joint_recommendations=joint_result.get("recommendations", []),
            corrections=corrections,
        )

        # 生成人类可读摘要
        report.summary = self._generate_summary(report, route_result, strategy_result, joint_result)

        return report

    def _generate_summary(
        self,
        report: ReflectionReport,
        route_result: dict,
        strategy_result: dict,
        joint_result: dict,
    ) -> str:
        """生成人类可读的反思摘要。"""
        lines = [
            f"=== Self-Reflective Routing 反思报告 ===",
            f"分析时间: {report.timestamp}",
            f"分析 episode 数: {report.episodes_analyzed}",
            "",
            f"## 路由质量",
            f"路由准确率: {report.routing_accuracy:.1%}",
            f"过度升级率: {report.over_escalation_rate:.1%} (该本地走了云端)",
            f"升级不足率: {report.under_escalation_rate:.1%} (该云端走了本地)",
        ]

        # 按任务类型的错误分布
        by_type = route_result.get("by_task_type", {})
        if by_type:
            lines.append("\n## 按任务类型的路由表现")
            for tt, stats in sorted(by_type.items()):
                t = stats.get("total", 0)
                c = stats.get("correct", 0)
                lines.append(f"  {tt}: {c}/{t} 正确 ({c/t:.0%})" if t > 0 else f"  {tt}: 无数据")

        # 策略浪费
        lines.append(f"\n## 策略效率")
        lines.append(f"平均 token 浪费率: {report.avg_token_waste_ratio:.1%}")
        lines.append(f"浪费实例数: {strategy_result.get('waste_count', 0)}")

        # 联合配对建议
        recs = joint_result.get("recommendations", [])
        if recs:
            lines.append(f"\n## 联合配对建议")
            for rec in recs[:5]:
                lines.append(
                    f"  {rec['task_type']}: {rec['current_pair']} → {rec['recommended_pair']} "
                    f"(省 {rec['token_savings']:.0%} token, 质量差 {rec['quality_delta']:+.2f})"
                )

        # 修正建议
        if report.corrections:
            lines.append(f"\n## 修正建议 ({len(report.corrections)} 条)")
            for c in report.corrections[:5]:
                cor = c if isinstance(c, Correction) else Correction()
                lines.append(
                    f"  [{cor.target}] {cor.parameter}: {cor.reason} "
                    f"(置信度: {cor.confidence:.0%})"
                )

        if not report.corrections:
            lines.append("\n## 当前路由表现良好，无需修正")

        return "\n".join(lines)

    def _save_report(self, report: ReflectionReport) -> None:
        """持久化反思报告。"""
        with self._lock:
            # JSONL 历史记录
            append_jsonl(self.reports_file, report.to_dict())
            # 最新报告（JSON，方便读取）
            try:
                with open(self.insights_file, "w", encoding="utf-8") as f:
                    json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
            except OSError as e:
                log.warning("保存反思报告失败: %s", e)

    def get_latest_report(self) -> Optional[dict]:
        """获取最新反思报告。"""
        if os.path.exists(self.insights_file):
            try:
                with open(self.insights_file, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        reports = read_jsonl(self.reports_file)
        return reports[-1] if reports else None

    def get_report_history(self, n: int = 10) -> list[dict]:
        """获取最近 N 份反思报告。"""
        reports = read_jsonl(self.reports_file)
        return reports[-n:]
