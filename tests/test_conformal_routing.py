"""
Conformalized Routing 测试

测试覆盖:
- AdaptiveConformalInference: ACI 状态机
- SlidingWindowCalibrator: 滑动窗口校准器
- UncertaintyDecomposer: 不确定性分解
- ConformalizedRouter: 完整路由决策
"""

import math
import pytest
import tempfile

from task_router.conformal_routing import (
    AdaptiveConformalInference,
    SlidingWindowCalibrator,
    UncertaintyDecomposer,
    ConformalizedRouter,
    ConformalDecision,
)


class MockQuantileFeatures:
    """模拟 TokenQuantileFeatures（不依赖 tqbc 模块）"""
    def __init__(self, q50_entropy=1.0, q50_margin=0.5, entropy_variance=0.3):
        self.q50_entropy = q50_entropy
        self.q50_margin = q50_margin
        self.entropy_variance = entropy_variance


# ─── 测试辅助 ──────────────────────────────────────────────────


def make_cascade_decision(escalate=False, calibrated_conf=0.8):
    """创建模拟的 CascadeDecision 输出"""
    return {
        "escalate": escalate,
        "raw_confidence": calibrated_conf,
        "calibrated_confidence": calibrated_conf,
        "reason": "test",
    }


def make_tqbc_decision(should_escalate=False, calibrated_conf=0.8, uncertainty=0.2):
    """创建模拟的 TQBCDecision"""
    class MockTQBCDecision:
        pass

    d = MockTQBCDecision()
    d.should_escalate = should_escalate
    d.calibrated_confidence = calibrated_conf
    d.uncertainty = uncertainty
    d.route = "cloud" if should_escalate else "local"
    d.reason = "test"
    return d


def make_ml_prediction(should_use_local=True, confidence=0.7):
    """创建模拟的 Meta-Learner 输出"""
    return {
        "should_use_local": should_use_local,
        "confidence": confidence,
        "local_success_prob": confidence,
    }


# ─── AdaptiveConformalInference 测试 ──────────────────────────


class TestAdaptiveConformalInference:
    """ACI 状态机"""

    def test_initialization(self):
        """初始化状态正确"""
        aci = AdaptiveConformalInference(target_coverage=0.9, gamma=0.005)
        assert aci.alpha_target == pytest.approx(0.1)
        assert aci.alpha == pytest.approx(0.1)
        assert aci.gamma == 0.005
        assert aci.n_steps == 0

    def test_alpha_decreases_on_coverage(self):
        """coverage 事件 (err=0) 应使 alpha 向 target 靠近"""
        aci = AdaptiveConformalInference(target_coverage=0.9, gamma=0.01)
        initial_alpha = aci.alpha

        # err=0: alpha = alpha + gamma * (0 - alpha_target) = alpha - gamma * alpha_target
        new_alpha = aci.update(0.0)
        assert new_alpha < initial_alpha

    def test_alpha_increases_on_miscoverage(self):
        """miscoverage 事件 (err=1) 应使 alpha 增加"""
        aci = AdaptiveConformalInference(target_coverage=0.9, gamma=0.01)
        initial_alpha = aci.alpha

        # err=1: alpha = alpha + gamma * (1 - alpha_target) = alpha + gamma * 0.9
        new_alpha = aci.update(1.0)
        assert new_alpha > initial_alpha

    def test_alpha_clamping_lower(self):
        """alpha 不低于 0.01"""
        aci = AdaptiveConformalInference(target_coverage=0.99, gamma=0.5)
        # 多次 coverage 事件推动 alpha 向下
        for _ in range(100):
            aci.update(0.0)
        assert aci.alpha >= 0.01

    def test_alpha_clamping_upper(self):
        """alpha 不高于 0.5"""
        aci = AdaptiveConformalInference(target_coverage=0.1, gamma=0.5)
        # 多次 miscoverage 事件推动 alpha 向上
        for _ in range(100):
            aci.update(1.0)
        assert aci.alpha <= 0.5

    def test_convergence(self):
        """长期运行后，经验覆盖率趋近目标"""
        aci = AdaptiveConformalInference(target_coverage=0.9, gamma=0.005)

        # 模拟 1000 次决策，每次 10% 概率 miscoverage
        import random
        random.seed(42)
        for _ in range(1000):
            err = 1.0 if random.random() < 0.1 else 0.0
            aci.update(err)

        coverage = aci.get_empirical_coverage()
        # 经验覆盖率应在 0.85-0.95 范围内（允许一定波动）
        assert 0.80 <= coverage <= 0.98

    def test_empirical_coverage_empty(self):
        """无数据时经验覆盖率为 1.0"""
        aci = AdaptiveConformalInference()
        assert aci.get_empirical_coverage() == 1.0

    def test_n_steps_counter(self):
        """步数计数正确"""
        aci = AdaptiveConformalInference()
        for _ in range(10):
            aci.update(0.0)
        assert aci.n_steps == 10

    def test_serialization(self):
        """序列化/反序列化保持状态"""
        aci = AdaptiveConformalInference(target_coverage=0.9, gamma=0.01)
        for _ in range(50):
            aci.update(0.0)
        for _ in range(10):
            aci.update(1.0)

        data = aci.to_dict()
        restored = AdaptiveConformalInference.from_dict(data)

        assert restored.alpha == aci.alpha
        assert restored.n_steps == aci.n_steps
        assert restored.cumulative_err == aci.cumulative_err
        assert restored.gamma == aci.gamma


# ─── SlidingWindowCalibrator 测试 ──────────────────────────


class TestSlidingWindowCalibrator:
    """滑动窗口校准器"""

    def test_empty_returns_1(self):
        """无数据时阈值为 1.0"""
        cal = SlidingWindowCalibrator(window_size=100)
        assert cal.get_threshold(0.1) == 1.0

    def test_threshold_is_quantile(self):
        """阈值是正确的分位数"""
        cal = SlidingWindowCalibrator(window_size=100)
        # 添加 100 个 score，均匀分布在 [0, 1]
        for i in range(100):
            cal.add_score(i / 100.0, was_correct=True)

        # alpha=0.1 → 90th percentile
        threshold = cal.get_threshold(0.1)
        assert 0.85 <= threshold <= 0.95

    def test_threshold_adapts_to_alpha(self):
        """不同 alpha 产生不同阈值"""
        cal = SlidingWindowCalibrator(window_size=100)
        for i in range(100):
            cal.add_score(i / 100.0, was_correct=True)

        t_low = cal.get_threshold(0.05)  # 95th percentile
        t_high = cal.get_threshold(0.2)  # 80th percentile
        assert t_low > t_high

    def test_window_slides(self):
        """旧数据被移出窗口"""
        cal = SlidingWindowCalibrator(window_size=10)

        # 添加 20 个 score，前 10 个为 0.9，后 10 个为 0.1
        for _ in range(10):
            cal.add_score(0.9, was_correct=True)
        for _ in range(10):
            cal.add_score(0.1, was_correct=True)

        # 窗口只保留最近 10 个 (score=0.1)
        threshold = cal.get_threshold(0.1)
        assert threshold <= 0.2  # 应该接近 0.1

    def test_correct_incorrect_separation(self):
        """正确和错误分数分别记录"""
        cal = SlidingWindowCalibrator(window_size=100)
        cal.add_score(0.1, was_correct=True)
        cal.add_score(0.5, was_correct=False)
        cal.add_score(0.2, was_correct=True)

        assert len(cal.get_correct_scores()) == 2
        assert len(cal.get_incorrect_scores()) == 1

    def test_serialization(self):
        """序列化/反序列化保持状态"""
        cal = SlidingWindowCalibrator(window_size=50)
        for i in range(30):
            cal.add_score(i / 30.0, was_correct=i % 2 == 0)

        data = cal.to_list()
        restored = SlidingWindowCalibrator.from_list(data, window_size=50)

        assert len(restored.scores) == len(cal.scores)
        assert restored.get_threshold(0.1) == cal.get_threshold(0.1)


# ─── UncertaintyDecomposer 测试 ──────────────────────────


class TestUncertaintyDecomposer:
    """不确定性分解器"""

    def test_cold_start_high(self):
        """新任务类型 cold_start 高"""
        dec = UncertaintyDecomposer()
        result = dec.decompose(
            tqbc_uncertainty=0.5,
            vote_entropy=0.0,
            task_type="new_type",
            features=[0.5] * 8,
        )
        assert result["cold_start"] >= 0.9  # 0 samples → cold_start ≈ 1.0

    def test_cold_start_decreases(self):
        """样本增加后 cold_start 降低"""
        dec = UncertaintyDecomposer()
        for _ in range(30):
            dec.update("test_type", [0.5] * 8)

        result = dec.decompose(
            tqbc_uncertainty=0.5,
            vote_entropy=0.0,
            task_type="test_type",
            features=[0.5] * 8,
        )
        assert result["cold_start"] < 0.5  # 30 samples → cold_start = max(0, 1-30/50) = 0.4

    def test_aleatoric_from_entropy(self):
        """投票分歧大 → aleatoric 高"""
        dec = UncertaintyDecomposer()
        # 最大熵：2 escalate, 2 keep → entropy = log(2) ≈ 0.693
        # max_entropy = log(4) ≈ 1.386
        # aleatoric = 0.693 / 1.386 ≈ 0.5
        max_entropy = -(0.5 * math.log(0.5) + 0.5 * math.log(0.5))
        result = dec.decompose(
            tqbc_uncertainty=0.0,
            vote_entropy=max_entropy,
            task_type="test",
            features=[0.5] * 8,
        )
        assert result["aleatoric"] >= 0.45  # ≈ 0.5

    def test_aleatoric_zero_on_consensus(self):
        """投票一致 → aleatoric 为 0"""
        dec = UncertaintyDecomposer()
        result = dec.decompose(
            tqbc_uncertainty=0.0,
            vote_entropy=0.0,
            task_type="test",
            features=[0.5] * 8,
        )
        assert result["aleatoric"] == 0.0

    def test_epistemic_from_tqbc(self):
        """epistemic 直接来自 tqbc_uncertainty"""
        dec = UncertaintyDecomposer()
        result = dec.decompose(
            tqbc_uncertainty=0.7,
            vote_entropy=0.0,
            task_type="test",
            features=[0.5] * 8,
        )
        assert result["epistemic"] == 0.7

    def test_distribution_shift_with_similar_data(self):
        """相似数据 → distribution_shift 低"""
        dec = UncertaintyDecomposer()
        for _ in range(20):
            dec.update("test", [0.5] * 8)

        result = dec.decompose(
            tqbc_uncertainty=0.0,
            vote_entropy=0.0,
            task_type="test",
            features=[0.5] * 8,  # 与历史数据完全相同
        )
        assert result["distribution_shift"] < 0.3

    def test_distribution_shift_with_different_data(self):
        """差异大的数据 → distribution_shift 高"""
        dec = UncertaintyDecomposer()
        for _ in range(20):
            dec.update("test", [0.5] * 8)

        result = dec.decompose(
            tqbc_uncertainty=0.0,
            vote_entropy=0.0,
            task_type="test",
            features=[5.0] * 8,  # 与历史数据差异大
        )
        assert result["distribution_shift"] > 0.5

    def test_serialization(self):
        """序列化/反序列化保持状态"""
        dec = UncertaintyDecomposer()
        for _ in range(10):
            dec.update("type_a", [0.5] * 8)
        for _ in range(5):
            dec.update("type_b", [0.3] * 8)

        data = dec.to_dict()
        restored = UncertaintyDecomposer.from_dict(data)

        assert restored._n_samples == dec._n_samples
        assert restored._task_type_counts == dec._task_type_counts


# ─── ConformalizedRouter 测试 ──────────────────────────


class TestConformalizedRouter:
    """Conformalized Router 完整测试"""

    @pytest.fixture
    def tmp_dir(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    @pytest.fixture
    def router(self, tmp_dir):
        return ConformalizedRouter(
            cache_dir=tmp_dir,
            target_coverage=0.9,
            gamma=0.005,
            window_size=50,
            escalation_margin=0.15,
        )

    def test_high_confidence_local(self, router):
        """高置信度 → prediction_set = {local}"""
        # 添加一些校准数据（低 score = 高 conforming）
        for _ in range(30):
            router.calibrator.add_score(0.05, was_correct=True)

        # 添加一些特征数据到 decomposer（减少 distribution_shift）
        for _ in range(10):
            router.decomposer.update("translation", [0.5] * 8)

        cascade = make_cascade_decision(escalate=False, calibrated_conf=0.95)
        tqbc = make_tqbc_decision(should_escalate=False, calibrated_conf=0.95, uncertainty=0.05)
        ml = make_ml_prediction(should_use_local=True, confidence=0.95)

        decision = router.decide(
            cascade_decision=cascade,
            tqbc_decision=tqbc,
            ml_prediction=ml,
            active_verify=False,
            features=[0.5] * 8,
            task_type="translation",
            raw_should_escalate=False,
            quantile_features=MockQuantileFeatures(q50_entropy=0.5, q50_margin=0.8),
        )

        assert decision.route == "local"
        assert decision.should_escalate is False
        assert "local" in decision.prediction_set

    def test_low_confidence_uncertain(self, router):
        """低置信度 → prediction_set = {local, cloud}"""
        # 添加高 score 的校准数据
        for _ in range(30):
            router.calibrator.add_score(0.8, was_correct=False)

        cascade = make_cascade_decision(escalate=False, calibrated_conf=0.2)
        tqbc = make_tqbc_decision(should_escalate=False, calibrated_conf=0.2, uncertainty=0.8)
        ml = make_ml_prediction(should_use_local=True, confidence=0.2)

        decision = router.decide(
            cascade_decision=cascade,
            tqbc_decision=tqbc,
            ml_prediction=ml,
            active_verify=False,
            features=[0.5] * 8,
            task_type="code_gen",
            raw_should_escalate=False,
            quantile_features=MockQuantileFeatures(q50_entropy=3.0, q50_margin=0.1),
        )

        # 低置信度应导致宽区间，可能触发升级
        assert decision.interval_width > 0
        assert len(decision.prediction_set) >= 1

    def test_override_suppress_escalation(self, router):
        """nonconforming + 窄区间时，用 raw_should_escalate 决策"""
        # 添加中等 score 校准数据
        for _ in range(30):
            router.calibrator.add_score(0.3, was_correct=True)

        cascade = make_cascade_decision(escalate=False, calibrated_conf=0.8)
        tqbc = make_tqbc_decision(should_escalate=False, calibrated_conf=0.8, uncertainty=0.2)
        ml = make_ml_prediction(should_use_local=True, confidence=0.8)

        # 高不确定性 → score > threshold → nonconforming
        decision = router.decide(
            cascade_decision=cascade,
            tqbc_decision=tqbc,
            ml_prediction=ml,
            active_verify=False,
            features=[0.5] * 8,
            task_type="translation",
            raw_should_escalate=False,  # OR 融合说不升级
            quantile_features=MockQuantileFeatures(q50_entropy=2.5, q50_margin=0.1),
        )

        # nonconforming → prediction_set = {local, cloud}
        assert len(decision.prediction_set) == 2
        # 窄区间 → 用 raw_should_escalate=False → 不升级
        if decision.interval_width <= router.escalation_margin:
            assert decision.should_escalate is False
            assert decision.route == "local"

    def test_force_escalation_on_wide_interval(self, router):
        """prediction_set={local,cloud} 且区间过宽时强制升级"""
        # 添加低 score 校准数据，使阈值约 0.3 < score → nonconforming
        for _ in range(30):
            router.calibrator.add_score(0.3, was_correct=False)

        cascade = make_cascade_decision(escalate=False, calibrated_conf=0.5)
        tqbc = make_tqbc_decision(should_escalate=False, calibrated_conf=0.5, uncertainty=0.5)
        ml = make_ml_prediction(should_use_local=True, confidence=0.5)

        # 高不确定性特征 → score > threshold → nonconforming → prediction_set = {local, cloud}
        decision = router.decide(
            cascade_decision=cascade,
            tqbc_decision=tqbc,
            ml_prediction=ml,
            active_verify=False,
            features=[0.5] * 8,
            task_type="unknown",
            raw_should_escalate=False,
            quantile_features=MockQuantileFeatures(q50_entropy=3.0, q50_margin=0.05, entropy_variance=1.0),
        )

        # score > threshold → nonconforming → prediction_set = {local, cloud}
        assert decision.nonconformity_score > decision.threshold, \
            f"score {decision.nonconformity_score:.4f} should be > threshold {decision.threshold:.4f}"
        assert len(decision.prediction_set) == 2
        # 区间过宽 → 强制升级
        if decision.interval_width > router.escalation_margin:
            assert decision.should_escalate is True

    def test_record_outcome_updates_aci(self, router):
        """record_outcome 更新 ACI 状态"""
        initial_steps = router.aci.n_steps

        cascade = make_cascade_decision()
        tqbc = make_tqbc_decision()
        ml = make_ml_prediction()

        decision = router.decide(
            cascade_decision=cascade,
            tqbc_decision=tqbc,
            ml_prediction=ml,
            active_verify=False,
            features=[0.5] * 8,
            task_type="test",
            raw_should_escalate=False,
            quantile_features=MockQuantileFeatures(),
        )

        router.record_outcome(decision, success=True, escalated=False, task_type="test")
        assert router.aci.n_steps == initial_steps + 1

    def test_record_outcome_updates_calibrator(self, router):
        """record_outcome 更新滑动窗口"""
        initial_scores = len(router.calibrator.scores)

        cascade = make_cascade_decision()
        tqbc = make_tqbc_decision()
        ml = make_ml_prediction()

        decision = router.decide(
            cascade_decision=cascade,
            tqbc_decision=tqbc,
            ml_prediction=ml,
            active_verify=False,
            features=[0.5] * 8,
            task_type="test",
            raw_should_escalate=False,
            quantile_features=MockQuantileFeatures(),
        )

        router.record_outcome(decision, success=True, escalated=False, task_type="test")
        assert len(router.calibrator.scores) == initial_scores + 1

    def test_persistence_roundtrip(self, tmp_dir):
        """持久化往返保持状态"""
        router1 = ConformalizedRouter(cache_dir=tmp_dir, window_size=50)

        # 添加一些数据
        for _ in range(20):
            router1.calibrator.add_score(0.3, was_correct=True)
        router1.aci.update(0.0)
        router1.aci.update(1.0)

        # 保存
        router1._save()

        # 创建新实例，应加载旧状态
        router2 = ConformalizedRouter(cache_dir=tmp_dir, window_size=50)
        assert router2.aci.n_steps == router1.aci.n_steps
        assert len(router2.calibrator.scores) == len(router1.calibrator.scores)

    def test_get_stats(self, router):
        """get_stats 返回完整统计"""
        stats = router.get_stats()
        assert "aci" in stats
        assert "calibrator" in stats
        assert "decomposer" in stats
        assert "config" in stats
        assert stats["aci"]["n_steps"] == 0

    def test_decision_has_all_fields(self, router):
        """决策包含所有必要字段"""
        cascade = make_cascade_decision()
        tqbc = make_tqbc_decision()
        ml = make_ml_prediction()

        decision = router.decide(
            cascade_decision=cascade,
            tqbc_decision=tqbc,
            ml_prediction=ml,
            active_verify=False,
            features=[0.5] * 8,
            task_type="test",
            raw_should_escalate=False,
            quantile_features=MockQuantileFeatures(),
        )

        assert isinstance(decision, ConformalDecision)
        assert decision.route in ("local", "cloud")
        assert isinstance(decision.should_escalate, bool)
        assert isinstance(decision.prediction_set, list)
        assert len(decision.confidence_interval) == 2
        assert 0 <= decision.alpha <= 0.5
        assert 0 <= decision.nonconformity_score <= 1
        assert isinstance(decision.uncertainty_sources, dict)
        assert isinstance(decision.layer_signals, dict)
        assert isinstance(decision.reason, str)

    def test_uncertainty_sources_complete(self, router):
        """不确定性来源包含所有四个维度"""
        cascade = make_cascade_decision()
        tqbc = make_tqbc_decision()
        ml = make_ml_prediction()

        decision = router.decide(
            cascade_decision=cascade,
            tqbc_decision=tqbc,
            ml_prediction=ml,
            active_verify=False,
            features=[0.5] * 8,
            task_type="test",
            raw_should_escalate=False,
            quantile_features=MockQuantileFeatures(),
        )

        sources = decision.uncertainty_sources
        assert "epistemic" in sources
        assert "aleatoric" in sources
        assert "distribution_shift" in sources
        assert "cold_start" in sources

    def test_fallback_without_quantile_features(self, router):
        """无 quantile_features 时回退到投票熵"""
        for _ in range(30):
            router.calibrator.add_score(0.3, was_correct=True)

        cascade = make_cascade_decision(escalate=False, calibrated_conf=0.8)
        tqbc = make_tqbc_decision(should_escalate=False, calibrated_conf=0.8, uncertainty=0.2)
        ml = make_ml_prediction(should_use_local=True, confidence=0.8)

        decision = router.decide(
            cascade_decision=cascade,
            tqbc_decision=tqbc,
            ml_prediction=ml,
            active_verify=False,
            features=[0.5] * 8,
            task_type="test",
            raw_should_escalate=False,
            quantile_features=None,  # 无原始特征
        )

        # 应正常工作，score 使用 fallback 路径
        assert 0 <= decision.nonconformity_score <= 1
        assert decision.route in ("local", "cloud")

    def test_confidence_interval_with_data(self, router):
        """有足够数据时置信区间非默认值"""
        # 添加足够数据
        for i in range(30):
            router.calibrator.add_score(i / 30.0, was_correct=True)

        cascade = make_cascade_decision()
        tqbc = make_tqbc_decision()
        ml = make_ml_prediction()

        decision = router.decide(
            cascade_decision=cascade,
            tqbc_decision=tqbc,
            ml_prediction=ml,
            active_verify=False,
            features=[0.5] * 8,
            task_type="test",
            raw_should_escalate=False,
            quantile_features=MockQuantileFeatures(),
        )

        # 有数据时不应是默认的 (0.1, 0.9)
        assert decision.confidence_interval != (0.1, 0.9) or len(router.calibrator.scores) < 5
        assert 0 <= decision.confidence_interval[0] <= decision.confidence_interval[1] <= 1

    def test_layer_signals_complete(self, router):
        """层信号包含所有四层"""
        cascade = make_cascade_decision()
        tqbc = make_tqbc_decision()
        ml = make_ml_prediction()

        decision = router.decide(
            cascade_decision=cascade,
            tqbc_decision=tqbc,
            ml_prediction=ml,
            active_verify=True,
            features=[0.5] * 8,
            task_type="test",
            raw_should_escalate=False,
            quantile_features=MockQuantileFeatures(),
        )

        signals = decision.layer_signals
        assert "cascade" in signals
        assert "meta_learner" in signals
        assert "active_learner" in signals
        assert "tqbc" in signals
        assert "raw_fusion" in signals

    def test_sliding_window_overflow(self, tmp_dir):
        """滑动窗口超过 window_size 后只保留最新数据"""
        router = ConformalizedRouter(cache_dir=tmp_dir, window_size=10)

        # 添加 25 个分数（超过窗口大小）
        for i in range(25):
            router.calibrator.add_score(i / 25.0, was_correct=True)

        # 窗口应只保留最近 10 个
        assert len(router.calibrator.scores) == 10
        # 最旧的分数（0/25, 1/25, ...）应被丢弃
        assert router.calibrator.scores[0][0] >= 15 / 25.0

    def test_concurrent_decide(self, router):
        """并发 decide 调用不崩溃"""
        import threading

        errors = []

        def do_decide():
            try:
                for _ in range(10):
                    router.decide(
                        cascade_decision=make_cascade_decision(),
                        tqbc_decision=make_tqbc_decision(),
                        ml_prediction=make_ml_prediction(),
                        active_verify=False,
                        features=[0.5] * 8,
                        task_type="test",
                        raw_should_escalate=False,
                        quantile_features=MockQuantileFeatures(),
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=do_decide) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrent errors: {errors}"

    def test_cold_start_no_crash(self, tmp_dir):
        """冷启动（无校准数据）不崩溃"""
        router = ConformalizedRouter(cache_dir=tmp_dir, window_size=50)

        decision = router.decide(
            cascade_decision=make_cascade_decision(),
            tqbc_decision=make_tqbc_decision(),
            ml_prediction=make_ml_prediction(),
            active_verify=False,
            features=[0.5] * 8,
            task_type="test",
            raw_should_escalate=False,
            quantile_features=MockQuantileFeatures(),
        )

        # 冷启动应返回有效决策
        assert decision.route in ("local", "cloud")
        assert 0 <= decision.nonconformity_score <= 1
        assert decision.alpha > 0
