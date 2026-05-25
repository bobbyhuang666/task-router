"""
Self-Reflective Routing 反思引擎测试

测试覆盖:
- RouteAnalyzer: 路由决策分析
- StrategyReflector: 策略浪费检测
- JointReflector: 联合配对分析
- CorrectionGenerator: 修正建议生成
- ReflectionEngine: 端到端反思流程
- ReflectionReport: 报告结构
"""

import json
import os
import pytest
import tempfile

from task_router.reflection import (
    RouteAnalyzer,
    StrategyReflector,
    JointReflector,
    CorrectionGenerator,
    Correction,
    ReflectionReport,
    ReflectionEngine,
)
from task_router.quality_judge import QualityJudge, reset_quality_judge
from task_router.episode_collector import EpisodeCollector, reset_episode_collector


# ─── 辅助函数 ──────────────────────────────────────────────


def _make_judged_episode(
    episode_id="ep001",
    task_type="translate_en2zh",
    route="local",
    strategy="direct",
    routing_error="none",
    optimal_route="local",
    optimal_strategy="direct",
    tokens_input=50,
    tokens_output=20,
    relevance=8.0,
    completeness=8.0,
    accuracy=8.0,
    efficiency=8.0,
    correctness=8.0,
    **overrides,
):
    """创建带评分的 episode dict"""
    ep = {
        "episode_id": episode_id,
        "action": f"测试任务 {episode_id}",
        "text": "测试文本",
        "task_type": task_type,
        "capability": "translation",
        "complexity_score": 2.0,
        "confidence_data": {"confidence": 0.8},
        "strategy": strategy,
        "route": route,
        "model_used": "qwen-tool",
        "output": "测试输出",
        "tokens_input": tokens_input,
        "tokens_output": tokens_output,
        "time_ms": 500,
        "cost_saved": 0.001,
        "quality_scores": {
            "relevance": relevance,
            "completeness": completeness,
            "accuracy": accuracy,
            "efficiency": efficiency,
            "correctness": correctness,
        },
        "routing_error": routing_error,
        "optimal_route": optimal_route,
        "optimal_strategy": optimal_strategy,
    }
    ep.update(overrides)
    return ep


# ─── Correction 数据结构测试 ──────────────────────────────────────


class TestCorrection:
    """Correction dataclass 测试"""

    def test_auto_id_and_timestamp(self):
        c = Correction(target="threshold", parameter="base_threshold")
        assert c.correction_id != ""
        assert c.timestamp != ""
        assert len(c.correction_id) == 12

    def test_to_dict(self):
        c = Correction(
            target="threshold",
            parameter="base_threshold",
            confidence=0.85,
            reason="测试",
        )
        d = c.to_dict()
        assert d["target"] == "threshold"
        assert d["confidence"] == 0.85
        assert "correction_id" in d
        assert "timestamp" in d


# ─── ReflectionReport 测试 ──────────────────────────────────────


class TestReflectionReport:
    """ReflectionReport 测试"""

    def test_to_dict(self):
        report = ReflectionReport(
            episodes_analyzed=10,
            routing_accuracy=0.85,
            corrections=[Correction(target="threshold", parameter="test")],
        )
        d = report.to_dict()
        assert d["episodes_analyzed"] == 10
        assert d["routing_accuracy"] == 0.85
        assert len(d["corrections"]) == 1


# ─── RouteAnalyzer 测试 ──────────────────────────────────────


class TestRouteAnalyzer:
    """路由决策分析测试"""

    def setup_method(self):
        self.analyzer = RouteAnalyzer()

    def test_empty_episodes(self):
        result = self.analyzer.analyze([])
        assert result["total"] == 0
        assert result["accuracy"] == 0.0

    def test_all_correct(self):
        episodes = [
            _make_judged_episode(episode_id=f"ep{i}", routing_error="none")
            for i in range(5)
        ]
        result = self.analyzer.analyze(episodes)
        assert result["total"] == 5
        assert result["correct"] == 5
        assert result["accuracy"] == 1.0
        assert result["over_escalated"] == 0
        assert result["under_escalated"] == 0

    def test_over_escalation(self):
        """该本地走云端 → over_escalated"""
        episodes = [
            _make_judged_episode(routing_error="none"),
            _make_judged_episode(episode_id="ep002", routing_error="over_escalated"),
            _make_judged_episode(episode_id="ep003", routing_error="over_escalated"),
        ]
        result = self.analyzer.analyze(episodes)
        assert result["over_escalated"] == 2
        assert result["accuracy"] == pytest.approx(1 / 3, abs=0.01)
        assert len(result["over_escalated_episodes"]) == 2

    def test_under_escalation(self):
        """该云端走本地 → under_escalated"""
        episodes = [
            _make_judged_episode(routing_error="under_escalated"),
            _make_judged_episode(episode_id="ep002", routing_error="under_escalated"),
            _make_judged_episode(episode_id="ep003", routing_error="none"),
        ]
        result = self.analyzer.analyze(episodes)
        assert result["under_escalated"] == 2

    def test_by_task_type_breakdown(self):
        """按任务类型聚类"""
        episodes = [
            _make_judged_episode(task_type="translate", routing_error="none"),
            _make_judged_episode(episode_id="ep002", task_type="translate", routing_error="over_escalated"),
            _make_judged_episode(episode_id="ep003", task_type="classify", routing_error="none"),
        ]
        result = self.analyzer.analyze(episodes)
        assert "translate" in result["by_task_type"]
        assert "classify" in result["by_task_type"]
        assert result["by_task_type"]["translate"]["over_escalated"] == 1
        assert result["by_task_type"]["classify"]["correct"] == 1


# ─── StrategyReflector 测试 ──────────────────────────────────────


class TestStrategyReflector:
    """策略浪费检测测试"""

    def setup_method(self):
        self.reflector = StrategyReflector()

    def test_empty_episodes(self):
        result = self.reflector.analyze([])
        assert result["total"] == 0

    def test_no_waste(self):
        """策略与最优一致 → 无浪费"""
        episodes = [
            _make_judged_episode(strategy="direct", optimal_strategy="direct"),
            _make_judged_episode(episode_id="ep002", strategy="cot", optimal_strategy="cot"),
        ]
        result = self.reflector.analyze(episodes)
        assert result["waste_count"] == 0

    def test_detect_waste(self):
        """用 CoT 但 direct 就够 → 浪费"""
        episodes = [
            _make_judged_episode(strategy="cot", optimal_strategy="direct"),
            _make_judged_episode(episode_id="ep002", strategy="cot", optimal_strategy="direct"),
            _make_judged_episode(episode_id="ep003", strategy="cot", optimal_strategy="direct"),
        ]
        result = self.reflector.analyze(episodes)
        assert result["waste_count"] == 3
        assert result["avg_waste_ratio"] > 0.4  # CoT 1.8x vs direct 1.0x

    def test_waste_details(self):
        """浪费详情包含具体信息"""
        episodes = [
            _make_judged_episode(
                strategy="structured", optimal_strategy="cod",
                task_type="classify", episode_id="ep_waste",
            ),
            _make_judged_episode(
                episode_id="ep002", strategy="structured", optimal_strategy="cod",
                task_type="classify",
            ),
            _make_judged_episode(
                episode_id="ep003", strategy="structured", optimal_strategy="cod",
                task_type="classify",
            ),
        ]
        result = self.reflector.analyze(episodes)
        assert len(result["waste_details"]) > 0
        assert result["waste_details"][0]["episode_id"] == "ep_waste"

    def test_by_task_type_aggregation(self):
        """按任务类型聚合浪费"""
        episodes = [
            _make_judged_episode(strategy="cot", optimal_strategy="direct", task_type="翻译"),
            _make_judged_episode(episode_id="ep002", strategy="direct", optimal_strategy="direct", task_type="分类"),
        ]
        # 补充更多 episodes
        for i in range(3, 6):
            episodes.append(_make_judged_episode(
                episode_id=f"ep{i:03d}", strategy="cot", optimal_strategy="direct", task_type="翻译",
            ))
        for i in range(6, 8):
            episodes.append(_make_judged_episode(
                episode_id=f"ep{i:03d}", strategy="direct", optimal_strategy="direct", task_type="分类",
            ))

        result = self.reflector.analyze(episodes)
        assert "翻译" in result["by_task_type"]
        assert result["by_task_type"]["翻译"]["waste"] > 0


# ─── JointReflector 测试 ──────────────────────────────────────


class TestJointReflector:
    """联合配对分析测试"""

    def setup_method(self):
        self.reflector = JointReflector()

    def test_empty_episodes(self):
        result = self.reflector.analyze([])
        assert result["pair_stats"] == {}
        assert result["recommendations"] == []

    def test_pair_stats(self):
        """配对统计正确"""
        episodes = [
            _make_judged_episode(route="local", strategy="direct", tokens_input=50, tokens_output=20),
            _make_judged_episode(episode_id="ep002", route="local", strategy="cot", tokens_input=100, tokens_output=80),
        ]
        result = self.reflector.analyze(episodes)
        assert "local:direct" in result["pair_stats"]
        assert "local:cot" in result["pair_stats"]
        assert result["pair_stats"]["local:direct"]["count"] == 1
        assert result["pair_stats"]["local:cot"]["count"] == 1

    def test_recommendation_generation(self):
        """当存在更优配对时生成建议"""
        # local:cot 平均质量 0.8，token 180
        # local:cod 平均质量 0.78，token 30 — 好建议
        episodes = []
        for i in range(4):
            episodes.append(_make_judged_episode(
                episode_id=f"cot_{i}", route="local", strategy="cot",
                tokens_input=100, tokens_output=80,
                task_type="classify",
                relevance=8, completeness=8, accuracy=8, efficiency=6, correctness=8,
            ))
        for i in range(4):
            episodes.append(_make_judged_episode(
                episode_id=f"cod_{i}", route="local", strategy="cod",
                tokens_input=20, tokens_output=10,
                task_type="classify",
                relevance=8, completeness=7, accuracy=8, efficiency=9, correctness=8,
            ))

        result = self.reflector.analyze(episodes)
        recs = result["recommendations"]
        assert len(recs) >= 1
        # 应该建议从 cot 切换到 cod
        assert any("cod" in r.get("recommended_pair", "") for r in recs)

    def test_no_recommendation_when_quality_degrades(self):
        """质量大幅下降时不推荐"""
        episodes = []
        for i in range(4):
            episodes.append(_make_judged_episode(
                episode_id=f"cot_{i}", route="local", strategy="cot",
                tokens_input=100, tokens_output=80, task_type="complex",
                relevance=9, completeness=9, accuracy=9, efficiency=6, correctness=9,
            ))
        for i in range(4):
            episodes.append(_make_judged_episode(
                episode_id=f"direct_{i}", route="local", strategy="direct",
                tokens_input=10, tokens_output=5, task_type="complex",
                relevance=3, completeness=3, accuracy=3, efficiency=10, correctness=3,
            ))

        result = self.reflector.analyze(episodes)
        # 质量差距太大，不应推荐 direct
        recs = result["recommendations"]
        for r in recs:
            if "direct" in r.get("recommended_pair", ""):
                assert r.get("quality_delta", 0) > -0.05


# ─── CorrectionGenerator 测试 ──────────────────────────────────────


class TestCorrectionGenerator:
    """修正建议生成测试"""

    def setup_method(self):
        self.generator = CorrectionGenerator()

    def test_no_corrections_for_good_performance(self):
        """表现良好时无修正"""
        corrections = self.generator.generate(
            route_analysis={"total": 10, "correct": 9, "over_escalated": 0,
                           "under_escalated": 1, "accuracy": 0.9,
                           "by_task_type": {}},
            strategy_analysis={"total": 10, "waste_count": 0, "avg_waste_ratio": 0.0,
                              "by_task_type": {}},
            joint_analysis={"recommendations": []},
        )
        assert len(corrections) == 0

    def test_threshold_correction_for_over_escalation(self):
        """过度升级 → 提高阈值"""
        corrections = self.generator.generate(
            route_analysis={
                "total": 20, "correct": 10, "over_escalated": 8,
                "under_escalated": 2, "accuracy": 0.5,
                "over_escalated_episodes": [f"ep{i}" for i in range(8)],
                "under_escalated_episodes": [],
                "by_task_type": {},
            },
            strategy_analysis={"total": 0, "waste_count": 0, "avg_waste_ratio": 0.0,
                              "by_task_type": {}},
            joint_analysis={"recommendations": []},
        )
        threshold_corrections = [c for c in corrections if c.target == "threshold"]
        assert len(threshold_corrections) >= 1
        assert threshold_corrections[0].new_value > 0  # 正增量=提高阈值

    def test_threshold_correction_for_under_escalation(self):
        """升级不足 → 降低阈值"""
        corrections = self.generator.generate(
            route_analysis={
                "total": 20, "correct": 10, "over_escalated": 2,
                "under_escalated": 8, "accuracy": 0.5,
                "over_escalated_episodes": [],
                "under_escalated_episodes": [f"ep{i}" for i in range(8)],
                "by_task_type": {},
            },
            strategy_analysis={"total": 0, "waste_count": 0, "avg_waste_ratio": 0.0,
                              "by_task_type": {}},
            joint_analysis={"recommendations": []},
        )
        threshold_corrections = [c for c in corrections if c.target == "threshold"]
        assert len(threshold_corrections) >= 1
        assert threshold_corrections[0].new_value < 0  # 负增量=降低阈值

    def test_strategy_correction_for_waste(self):
        """策略浪费 → 策略权重修正"""
        corrections = self.generator.generate(
            route_analysis={"total": 0, "correct": 0, "over_escalated": 0,
                           "under_escalated": 0, "by_task_type": {}},
            strategy_analysis={
                "total": 15, "waste_count": 10, "avg_waste_ratio": 0.35,
                "by_task_type": {},
            },
            joint_analysis={"recommendations": []},
        )
        strategy_corrections = [c for c in corrections if c.target == "strategy_weight"]
        assert len(strategy_corrections) >= 1
        assert strategy_corrections[0].new_value is True

    def test_joint_correction_for_pair_recommendation(self):
        """联合配对建议 → 路由策略修正"""
        corrections = self.generator.generate(
            route_analysis={"total": 0, "correct": 0, "over_escalated": 0,
                           "under_escalated": 0, "by_task_type": {}},
            strategy_analysis={"total": 0, "waste_count": 0, "avg_waste_ratio": 0.0,
                              "by_task_type": {}},
            joint_analysis={
                "recommendations": [{
                    "task_type": "classify",
                    "current_pair": "local:cot",
                    "recommended_pair": "local:cod",
                    "token_savings": 0.5,
                    "evidence_count": 5,
                    "reason": "CoD 比 CoT 节省 50% token",
                }],
            },
        )
        policy_corrections = [c for c in corrections if c.target == "routing_policy"]
        assert len(policy_corrections) >= 1

    def test_min_evidence_threshold(self):
        """证据不足时不生成修正"""
        corrections = self.generator.generate(
            route_analysis={
                "total": 2, "correct": 0, "over_escalated": 2,
                "under_escalated": 0, "accuracy": 0.0,
                "over_escalated_episodes": ["ep1", "ep2"],
                "under_escalated_episodes": [],
                "by_task_type": {},
            },
            strategy_analysis={"total": 0, "waste_count": 0, "avg_waste_ratio": 0.0,
                              "by_task_type": {}},
            joint_analysis={"recommendations": []},
        )
        # 只有 2 条证据，不够 MIN_EVIDENCE=3
        threshold_corrections = [c for c in corrections if c.target == "threshold"]
        assert len(threshold_corrections) == 0


# ─── ReflectionEngine 端到端测试 ──────────────────────────────────────


class TestReflectionEngine:
    """ReflectionEngine 端到端测试"""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        reset_episode_collector()
        reset_quality_judge()

        # 预填充 episodes
        self.collector = EpisodeCollector(cache_dir=self.tmpdir, scrub_pii=False)
        self.judge = QualityJudge(cache_dir=self.tmpdir, use_llm=False)
        self.engine = ReflectionEngine(cache_dir=self.tmpdir)
        self.engine.collector = self.collector
        self.engine.judge = self.judge

    def teardown_method(self):
        reset_episode_collector()
        reset_quality_judge()

    def test_reflect_with_pre_judged_episodes(self):
        """用预评估的 episodes 反思"""
        episodes = [
            _make_judged_episode(episode_id=f"ep{i}", routing_error="none")
            for i in range(10)
        ]
        report = self.engine.reflect(episodes=episodes)
        assert report.episodes_analyzed == 10
        assert report.routing_accuracy == 1.0
        assert len(report.corrections) == 0  # 表现好，无修正

    def test_reflect_identifies_over_escalation(self):
        """识别过度升级"""
        episodes = [
            _make_judged_episode(episode_id=f"good_{i}", routing_error="none")
            for i in range(5)
        ] + [
            _make_judged_episode(episode_id=f"over_{i}", routing_error="over_escalated")
            for i in range(8)
        ]
        report = self.engine.reflect(episodes=episodes)
        assert report.over_escalation_rate > 0.5
        assert len(report.corrections) > 0

    def test_reflect_generates_summary(self):
        """反思报告包含人类可读摘要"""
        episodes = [_make_judged_episode(episode_id=f"ep{i}") for i in range(5)]
        report = self.engine.reflect(episodes=episodes)
        assert "Self-Reflective Routing" in report.summary
        assert "路由准确率" in report.summary

    def test_reflect_saves_report(self):
        """反思报告持久化"""
        episodes = [_make_judged_episode(episode_id=f"ep{i}") for i in range(3)]
        self.engine.reflect(episodes=episodes)

        latest = self.engine.get_latest_report()
        assert latest is not None
        assert latest["episodes_analyzed"] == 3

    def test_reflect_empty_episodes(self):
        """无 episode 时不崩溃"""
        report = self.engine.reflect(episodes=[])
        assert report.episodes_analyzed == 0
        assert "没有" in report.summary

    def test_reflect_with_mixed_errors(self):
        """混合错误类型的分析"""
        episodes = [
            _make_judged_episode(episode_id="good1", routing_error="none"),
            _make_judged_episode(episode_id="good2", routing_error="none"),
            _make_judged_episode(episode_id="over1", routing_error="over_escalated"),
            _make_judged_episode(
                episode_id="under1", routing_error="under_escalated",
                strategy="cot", optimal_strategy="cot",
            ),
            _make_judged_episode(
                episode_id="waste1", routing_error="none",
                strategy="structured", optimal_strategy="direct",
            ),
        ]
        report = self.engine.reflect(episodes=episodes)
        assert report.episodes_analyzed == 5
        assert report.over_escalation_rate == pytest.approx(1 / 5)
        assert report.under_escalation_rate == pytest.approx(1 / 5)

    def test_reflect_with_task_type_analysis(self):
        """按任务类型分析"""
        episodes = []
        # 翻译任务：表现好
        for i in range(4):
            episodes.append(_make_judged_episode(
                episode_id=f"trans_{i}", task_type="translate",
                routing_error="none",
            ))
        # 分类任务：过度升级严重
        for i in range(4):
            episodes.append(_make_judged_episode(
                episode_id=f"class_{i}", task_type="classify",
                routing_error="over_escalated",
            ))

        report = self.engine.reflect(episodes=episodes)
        assert "translate" in report.routing_errors_by_type
        assert "classify" in report.routing_errors_by_type
        assert report.routing_errors_by_type["translate"]["correct"] == 4
        assert report.routing_errors_by_type["classify"]["over_escalated"] == 4

    def test_reflect_with_real_collector_and_judge(self):
        """用真实 Collector + Judge 的集成测试"""
        collector = EpisodeCollector(cache_dir=self.tmpdir, scrub_pii=False)
        judge = QualityJudge(cache_dir=self.tmpdir, use_llm=False)
        engine = ReflectionEngine(cache_dir=self.tmpdir)
        engine.collector = collector
        engine.judge = judge

        # 模拟 run_task 记录
        from task_router.routing import Task
        for i in range(5):
            task = Task(
                action=f"翻译第{i}条",
                text=f"Hello {i}",
                route="local",
                model_used="qwen-tool",
            )
            task.output = f"你好 {i}"
            task.tokens_input = 40
            task.tokens_output = 15
            task.time_ms = 500
            task.cost_saved = 0.001

            collector.record(task, {
                "complexity_score": 1.5,
                "confidence_data": {"confidence": 0.9},
                "strategy": "direct",
                "strategy_reason": "简单翻译",
                "routing_signals": {},
                "task_type": "translate_en2zh",
                "capability": "translation",
            })

        collector.flush()

        # 运行反思
        report = engine.reflect(n_episodes=10)
        assert report.episodes_analyzed == 5
        assert report.routing_accuracy >= 0.0
        assert report.summary != ""
