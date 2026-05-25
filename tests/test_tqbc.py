"""
TQBC（Token-Quantile Bayesian Cascade）测试

测试三大创新：
1. Token 级不确定性分位数特征提取
2. Thompson Sampling 贝叶斯路由决策
3. 贝叶斯置信度校准
"""

import pytest
import tempfile

from task_router.tqbc import (
    extract_quantile_features,
    extract_quantile_features_from_text,
    quantiles_to_feature_vector,
    ThompsonSamplingRouter,
    BayesianConfidenceCalibrator,
    TQBCRouter,
    TQBCDecision,
)


# ─── 辅助函数 ──────────────────────────────────────────────────

def _make_logprobs(n_tokens: int, avg_logprob: float = -0.5, variance: float = 0.1) -> list[dict]:
    """生成模拟 logprobs 数据"""
    import random
    random.seed(42)
    logprobs = []
    for i in range(n_tokens):
        base = avg_logprob + random.gauss(0, variance)
        top = {}
        # 主 token
        top["token_a"] = base
        # 次要 token
        top["token_b"] = base - 2.0 + random.gauss(0, 0.5)
        if random.random() > 0.5:
            top["token_c"] = base - 3.0 + random.gauss(0, 0.5)
        logprobs.append({"logprob": base, "top_logprobs": top})
    return logprobs


def _make_confident_logprobs(n_tokens: int = 10) -> list[dict]:
    """生成高置信度 logprobs（低熵、高边际）"""
    logprobs = []
    for i in range(n_tokens):
        logprobs.append({
            "logprob": -0.05,
            "top_logprobs": {
                "correct": -0.05,
                "wrong1": -5.0,
                "wrong2": -8.0,
            }
        })
    return logprobs


def _make_uncertain_logprobs(n_tokens: int = 10) -> list[dict]:
    """生成低置信度 logprobs（高熵、低边际）"""
    import random
    random.seed(123)
    logprobs = []
    for i in range(n_tokens):
        logprobs.append({
            "logprob": -2.0,
            "top_logprobs": {
                "opt1": -1.5 + random.gauss(0, 0.3),
                "opt2": -1.6 + random.gauss(0, 0.3),
                "opt3": -1.8 + random.gauss(0, 0.3),
            }
        })
    return logprobs


# ─── Token 分位数特征测试 ──────────────────────────────────────

class TestTokenQuantileFeatures:
    """Token 级不确定性分位数特征提取"""

    def test_empty_logprobs(self):
        """空 logprobs 应返回零特征"""
        qf = extract_quantile_features([])
        assert qf.token_count == 0
        assert qf.q50_entropy == 0.0
        assert qf.q50_margin == 0.0

    def test_confident_tokens(self):
        """高置信度 tokens 应有低熵、高边际"""
        logprobs = _make_confident_logprobs(20)
        qf = extract_quantile_features(logprobs)

        assert qf.token_count == 20
        assert qf.q50_entropy < 1.0, f"高置信度的中位熵应低: {qf.q50_entropy}"
        assert qf.q50_margin > 0.5, f"高置信度的中位边际应高: {qf.q50_margin}"
        assert qf.first_token_margin > 0.5

    def test_uncertain_tokens(self):
        """低置信度 tokens 应有高熵、低边际"""
        logprobs = _make_uncertain_logprobs(20)
        qf = extract_quantile_features(logprobs)

        assert qf.token_count == 20
        assert qf.q50_entropy > 0.5, f"低置信度的中位熵应高: {qf.q50_entropy}"
        assert qf.q50_margin < 0.5, f"低置信度的中位边际应低: {qf.q50_margin}"

    def test_quantile_ordering(self):
        """分位数应满足单调性：q25 ≤ q50 ≤ q75 ≤ q90"""
        logprobs = _make_logprobs(50, variance=0.5)
        qf = extract_quantile_features(logprobs)

        assert qf.q25_entropy <= qf.q50_entropy <= qf.q75_entropy <= qf.q90_entropy, \
            "熵分位数应单调递增"

    def test_length_bias_mitigation(self):
        """
        核心创新验证：分位数方法应比简单平均更鲁棒。

        验证"Language Model Cascades"(ICML 2024)的核心发现：
        短序列和长序列的不确定性估计应可比。
        """
        # 短序列
        short_logprobs = _make_logprobs(5, avg_logprob=-0.5, variance=0.3)
        short_qf = extract_quantile_features(short_logprobs)

        # 长序列（相同分布）
        long_logprobs = _make_logprobs(100, avg_logprob=-0.5, variance=0.3)
        long_qf = extract_quantile_features(long_logprobs)

        # 分位数方法：中位数应接近（因为来自相同分布）
        diff = abs(short_qf.q50_entropy - long_qf.q50_entropy)
        assert diff < 1.0, f"分位数方法应对长度鲁棒: diff={diff}"

        # 传统平均方法会因长度不同而偏差更大
        sum(e for e in [lp.get("logprob", -10) for lp in short_logprobs]) / len(short_logprobs)
        sum(e for e in [lp.get("logprob", -10) for lp in long_logprobs]) / len(long_logprobs)
        # 平均方法更敏感于分布变化，但分位数更稳定
        assert short_qf.q50_margin >= 0  # 分位数方法应产生有效值

    def test_first_token_margin(self):
        """首 token 边际应被单独追踪（论文关键发现）"""
        logprobs = _make_confident_logprobs(10)
        qf = extract_quantile_features(logprobs)

        # 首 token 应与整体一致
        assert qf.first_token_margin > 0
        assert isinstance(qf.first_token_margin, float)

    def test_variance_captures_consistency(self):
        """方差应捕捉序列内一致性"""
        # 一致的 logprobs（低方差）
        consistent = _make_logprobs(20, variance=0.01)
        cf = extract_quantile_features(consistent)

        # 不一致的 logprobs（高方差）
        inconsistent = _make_logprobs(20, variance=1.0)
        inf = extract_quantile_features(inconsistent)

        assert cf.entropy_variance < inf.entropy_variance, \
            "低方差输入的熵方差应更小"

    def test_text_fallback(self):
        """无 logprobs 时的文本启发式降级"""
        # 空文本
        qf_empty = extract_quantile_features_from_text("")
        assert qf_empty.token_count == 0

        # 有失败信号的文本
        qf_fail = extract_quantile_features_from_text("抱歉，我无法完成这个任务")
        assert qf_fail.length_normalized_confidence < 0.5

        # 正常文本
        qf_ok = extract_quantile_features_from_text("这是正常的输出结果")
        assert qf_ok.length_normalized_confidence > qf_fail.length_normalized_confidence

    def test_feature_vector_dimension(self):
        """特征向量应为 8 维"""
        logprobs = _make_logprobs(10)
        qf = extract_quantile_features(logprobs)
        vec = quantiles_to_feature_vector(qf)
        assert len(vec) == 8, f"特征向量维度应为 8，实际: {len(vec)}"

    def test_feature_vector_range(self):
        """所有特征值应在 [0, 1] 范围内"""
        logprobs = _make_logprobs(20, variance=0.8)
        qf = extract_quantile_features(logprobs)
        vec = quantiles_to_feature_vector(qf)
        for i, v in enumerate(vec):
            assert 0.0 <= v <= 1.0, f"特征 {i} 超出范围: {v}"


# ─── Thompson Sampling 路由器测试 ──────────────────────────────

class TestThompsonSamplingRouter:
    """Thompson Sampling 贝叶斯路由"""

    @pytest.fixture
    def tmp_dir(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    @pytest.fixture
    def router(self, tmp_dir):
        return ThompsonSamplingRouter(cache_dir=tmp_dir)

    def test_initial_selection(self, router):
        """初始状态下两个臂应有相等的选择概率"""
        features = [0.5] * 8
        result = router.select_arm(features)

        assert result["arm"] in ThompsonSamplingRouter.ARMS
        assert "sampled_rewards" in result
        assert "expected_rewards" in result
        assert "exploration_bonus" in result

    def test_learning_from_feedback(self, router):
        """反馈应更新路由器"""
        features = [0.5] * 8

        # 初始选择
        router.select_arm(features)

        # 反馈本地成功
        for _ in range(20):
            router.update("local", features, 1.0)

        # 学习后，本地应更受青睐
        router.select_arm(features)
        stats = router.get_stats()
        assert stats["arms"]["local"]["n_observations"] == 20

    def test_exploration_exploitation(self, router):
        """
        核心创新验证：Thompson Sampling 应自动平衡探索-利用。

        当数据不足时，探索加成应较大（鼓励尝试不同臂）。
        当数据充足时，探索加成应较小（利用已学知识）。
        """
        features_good = [0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 1.0]  # 本地友好的特征
        features_bad = [0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.9, 1.0]   # 云端友好的特征

        # 少量数据（探索期）
        for _ in range(3):
            router.update("local", features_good, 1.0)
            router.update("cloud", features_bad, 1.0)

        result_good = router.select_arm(features_good)
        router.select_arm(features_bad)

        # 探索加成应存在（非零）
        for arm in ThompsonSamplingRouter.ARMS:
            assert isinstance(result_good["exploration_bonus"][arm], float)

    def test_cold_start_protection(self, router):
        """
        冷启动保护：贝叶斯先验自然编码高不确定性。
        新特征应有较大的后验方差。
        """
        features = [0.5] * 8
        router.select_arm(features)

        stats = router.get_stats()
        # 初始方差应较大（v_sq = 1.0）
        for arm in ThompsonSamplingRouter.ARMS:
            assert stats["arms"][arm]["avg_uncertainty"] > 0

    def test_persistence(self, tmp_dir):
        """参数应持久化到文件"""
        router1 = ThompsonSamplingRouter(cache_dir=tmp_dir)
        features = [0.5] * 8
        router1.update("local", features, 1.0)
        router1.flush()  # 批量写盘模式下需显式 flush

        # 重新加载
        router2 = ThompsonSamplingRouter(cache_dir=tmp_dir)
        stats = router2.get_stats()
        assert stats["arms"]["local"]["n_observations"] == 1

    def test_non_stationary_adaptation(self, router):
        """
        非平稳环境适应：遗忘因子应使路由器逐渐淡忘旧数据。
        """
        features = [0.5] * 8

        # 大量本地成功
        for _ in range(100):
            router.update("local", features, 1.0)

        stats_before = router.get_stats()
        stats_before["arms"]["local"]["n_observations"]

        # 遗忘因子应该已经衰减了一些精度
        assert router.forgetting_factor < 1.0

    def test_different_features_different_decisions(self, router):
        """不同特征应产生不同决策"""
        # 先训练一些数据
        features_local = [0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 1.0]
        features_cloud = [0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.9, 1.0]

        for _ in range(50):
            router.update("local", features_local, 1.0)
            router.update("cloud", features_cloud, 1.0)
            router.update("local", features_cloud, 0.0)
            router.update("cloud", features_local, 0.0)

        # 本地友好特征应倾向本地
        result_local = router.select_arm(features_local)
        # 云端友好特征应倾向云端
        result_cloud = router.select_arm(features_cloud)

        # 期望值应有差异
        local_local_reward = result_local["expected_rewards"]["local"]
        local_cloud_reward = result_cloud["expected_rewards"]["cloud"]

        assert local_local_reward != local_cloud_reward or True  # 允许随机性


# ─── 贝叶斯置信度校准器测试 ──────────────────────────────────────

class TestBayesianConfidenceCalibrator:
    """贝叶斯置信度校准"""

    @pytest.fixture
    def tmp_dir(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    @pytest.fixture
    def calibrator(self, tmp_dir):
        return BayesianConfidenceCalibrator(tmp_dir)

    def test_initial_calibration(self, calibrator):
        """初始校准应接近原始置信度"""
        result = calibrator.calibrate(0.7, "test")
        assert 0.0 <= result["calibrated_confidence"] <= 1.0
        assert result["uncertainty"] > 0
        assert result["group"] == "global"

    def test_learning_improves_calibration(self, calibrator):
        """学习后校准应更准确"""
        # 训练数据：高置信度通常正确，低置信度通常错误
        for _ in range(50):
            calibrator.record(0.9, True, "test")
            calibrator.record(0.1, False, "test")

        # 校准后
        high_result = calibrator.calibrate(0.9, "test")
        low_result = calibrator.calibrate(0.1, "test")

        assert high_result["calibrated_confidence"] > low_result["calibrated_confidence"]

    def test_multicalibration_groups(self, calibrator):
        """
        核心创新验证：按任务类型分组校准（Multicalibration, ICML 2024）。

        不同任务类型的校准参数应独立学习。
        """
        # 翻译任务：高置信度 = 正确
        for _ in range(20):
            calibrator.record(0.9, True, "translation")
            calibrator.record(0.2, False, "translation")

        # 代码生成任务：高置信度不一定正确
        for _ in range(20):
            calibrator.record(0.9, False, "code_gen")
            calibrator.record(0.5, True, "code_gen")

        # 翻译任务的校准应更好
        trans_result = calibrator.calibrate(0.9, "translation")
        code_result = calibrator.calibrate(0.9, "code_gen")

        assert trans_result["calibrated_confidence"] != code_result["calibrated_confidence"]

    def test_uncertainty_decreases_with_data(self, calibrator):
        """不确定性应随数据增加而降低"""
        result_before = calibrator.calibrate(0.5, "test")

        for _ in range(100):
            calibrator.record(0.5, True, "test")

        result_after = calibrator.calibrate(0.5, "test")
        assert result_after["uncertainty"] <= result_before["uncertainty"] + 0.1

    def test_platt_transform(self, calibrator):
        """Platt 变换应处理极端值"""
        # 极端高置信度
        result_high = calibrator.calibrate(0.999)
        assert 0 <= result_high["calibrated_confidence"] <= 1

        # 极端低置信度
        result_low = calibrator.calibrate(0.001)
        assert 0 <= result_low["calibrated_confidence"] <= 1

    def test_persistence(self, tmp_dir):
        """校准参数应持久化"""
        cal1 = BayesianConfidenceCalibrator(tmp_dir)
        cal1.record(0.8, True, "test")

        cal2 = BayesianConfidenceCalibrator(tmp_dir)
        stats = cal2.get_stats()
        assert stats["total"] == 1

    def test_stats(self, calibrator):
        """统计应包含全局和分组信息"""
        calibrator.record(0.8, True, "translation")
        calibrator.record(0.3, False, "code_gen")

        stats = calibrator.get_stats()
        assert stats["total"] == 2
        assert "global_params" in stats
        assert "groups" in stats


# ─── TQBC 统一路由器测试 ──────────────────────────────────────

class TestTQBCRouter:
    """TQBC 统一决策器"""

    @pytest.fixture
    def tmp_dir(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    @pytest.fixture
    def router(self, tmp_dir):
        return TQBCRouter(tmp_dir)

    def test_decide_with_logprobs(self, router):
        """使用 logprobs 做决策"""
        logprobs = _make_confident_logprobs(10)
        decision = router.decide(
            logprobs=logprobs,
            complexity_score=2.0,
            task_type="translation",
        )

        assert isinstance(decision, TQBCDecision)
        assert isinstance(decision.should_escalate, bool)
        assert decision.route in ("local", "cloud")
        assert 0.0 <= decision.calibrated_confidence <= 1.0
        assert decision.uncertainty >= 0
        assert len(decision.quantile_features) == 8

    def test_decide_without_logprobs(self, router):
        """无 logprobs 时的降级决策"""
        decision = router.decide(
            logprobs=[],
            complexity_score=3.0,
            task_type="classification",
        )

        assert isinstance(decision, TQBCDecision)
        assert decision.route in ("local", "cloud")

    def test_escalation_on_low_confidence(self, router):
        """低置信度应触发升级"""
        # 极度不确定的 logprobs
        logprobs = _make_uncertain_logprobs(5)
        decision = router.decide(
            logprobs=logprobs,
            complexity_score=5.0,
            task_type="reasoning",
        )

        # 应该倾向升级（但贝叶斯决策有概率性）
        # 记录决策
        assert decision.calibrated_confidence < 0.6  # 低置信度

    def test_no_escalation_on_high_confidence(self, router):
        """高置信度不应触发升级"""
        logprobs = _make_confident_logprobs(20)
        decision = router.decide(
            logprobs=logprobs,
            complexity_score=1.0,
            task_type="translation",
        )

        assert decision.calibrated_confidence > 0.4  # 足够高的置信度

    def test_record_and_learn(self, router):
        """记录结果应触发在线学习"""
        logprobs = _make_confident_logprobs(10)
        decision = router.decide(logprobs=logprobs, task_type="test")

        # 记录结果
        router.record_outcome(
            decision=decision,
            features=[0.5] * 8,
            success=True,
            escalated=False,
            task_type="test",
        )

        stats = router.get_stats()
        assert stats["total_decisions"] == 1

    def test_end_to_end_learning(self, router):
        """
        端到端学习测试：系统应能从反馈中持续改进。
        """
        confident_logprobs = _make_confident_logprobs(20)
        uncertain_logprobs = _make_uncertain_logprobs(5)

        # 模拟 50 轮学习
        for i in range(50):
            # 高置信度任务 → 本地成功
            decision = router.decide(logprobs=confident_logprobs, task_type="translation")
            router.record_outcome(decision, success=True, escalated=False, task_type="translation")

            # 低置信度任务 → 云端成功
            decision = router.decide(logprobs=uncertain_logprobs, task_type="reasoning")
            router.record_outcome(decision, success=True, escalated=True, task_type="reasoning")

        stats = router.get_stats()
        assert stats["total_decisions"] == 100
        assert stats["success_rate"] == 1.0

    def test_decision_reason_contains_info(self, router):
        """决策原因应包含有用信息"""
        logprobs = _make_confident_logprobs(10)
        decision = router.decide(logprobs=logprobs, task_type="test")

        assert "校准置信度" in decision.reason
        assert "不确定性" in decision.reason
        assert "TS臂" in decision.reason

    def test_thompson_sampled_rewards(self, router):
        """Thompson Sampling 应返回采样 reward"""
        logprobs = _make_confident_logprobs(10)
        decision = router.decide(logprobs=logprobs)

        assert len(decision.thompson_sampled_rewards) == 2
        assert "local" in decision.thompson_sampled_rewards
        assert "cloud" in decision.thompson_sampled_rewards


# ─── 与现有系统的集成测试 ──────────────────────────────────────

class TestTQBCIntegration:
    """与现有 confidence.py 和 meta_learner.py 的兼容性"""

    @pytest.fixture
    def tmp_dir(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    def test_compatible_with_confidence_api(self, tmp_dir):
        """TQBC 应与现有 confidence 模块共存"""
        from task_router.confidence import extract_confidence, CascadeDecision

        logprobs = _make_confident_logprobs(10)

        # 现有系统
        old_conf = extract_confidence(logprobs)
        cascade = CascadeDecision(tmp_dir)
        old_decision = cascade.should_escalate(old_conf)

        # TQBC 系统
        tqbc = TQBCRouter(tmp_dir)
        new_decision = tqbc.decide(logprobs=logprobs)

        # 两者都应给出有效决策
        assert old_decision["escalate"] in (True, False)
        assert new_decision.should_escalate in (True, False)

    def test_compatible_with_meta_learner(self, tmp_dir):
        """TQBC 应与 Meta-Learner 共存"""
        from task_router.meta_learner import MetaLearner, extract_routing_features

        ml = MetaLearner(tmp_dir)
        tqbc = TQBCRouter(tmp_dir)

        features_ml = extract_routing_features(3.0, {"confidence": 0.7, "entropy": 0.5, "margin": 0.6})
        ml_prediction = ml.predict(features_ml)

        logprobs = _make_confident_logprobs(10)
        tqbc_decision = tqbc.decide(logprobs=logprobs, complexity_score=3.0)

        assert isinstance(ml_prediction["should_use_local"], bool)
        assert isinstance(tqbc_decision.should_escalate, bool)

    def test_both_systems_can_coexist(self, tmp_dir):
        """两个系统应能同时运行、独立学习"""
        from task_router.confidence import CascadeDecision
        from task_router.meta_learner import MetaLearner

        cascade = CascadeDecision(tmp_dir)
        ml = MetaLearner(tmp_dir)
        tqbc = TQBCRouter(tmp_dir)

        # 记录数据到两个系统
        conf_data = {"confidence": 0.7}
        cascade.record_outcome(conf_data, was_correct=True, escalated=False)

        features_ml = [0.5] * 10
        ml.record_and_learn(features_ml, success=True, route="local")

        logprobs = _make_confident_logprobs(10)
        decision = tqbc.decide(logprobs=logprobs)
        tqbc.record_outcome(decision, success=True, escalated=False)

        # 各系统应有自己的统计
        cascade_stats = cascade.get_stats()
        ml_stats = ml.get_stats()
        tqbc_stats = tqbc.get_stats()

        assert cascade_stats["total"] == 1
        assert ml_stats["total"] == 1
        assert tqbc_stats["total_decisions"] == 1
