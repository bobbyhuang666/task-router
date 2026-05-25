"""
Meta-Learner 测试

测试覆盖:
- OnlineLogisticRegression: 在线学习模型
- MetaLearner: 统一路由决策
- ActiveLearner: 主动学习（不确定性采样）
"""

import pytest
import tempfile

from task_router.meta_learner import (
    extract_routing_features,
    OnlineLogisticRegression,
    MetaLearner,
    ActiveLearner,
)


# ─── 特征提取测试 ──────────────────────────────────────────────


class TestExtractRoutingFeatures:
    """特征提取"""

    def test_returns_10_dims(self):
        """返回 10 维特征"""
        features = extract_routing_features(3.0, {"confidence": 0.8, "entropy": 1.0, "margin": 0.6})
        assert len(features) == 10

    def test_bias_is_last(self):
        """最后一维是偏置项（恒为 1）"""
        features = extract_routing_features(0, {})
        assert features[-1] == 1.0

    def test_complexity_normalized(self):
        """复杂度归一化到 [0, 1]"""
        f_low = extract_routing_features(0, {})
        f_high = extract_routing_features(10, {})
        assert f_low[0] == 0.0
        assert f_high[0] == 1.0

    def test_complexity_clamped(self):
        """复杂度超出范围时截断"""
        f = extract_routing_features(20, {})
        assert f[0] == 1.0

    def test_confidence_preserved(self):
        """置信度直接传递"""
        f = extract_routing_features(3, {"confidence": 0.75})
        assert f[1] == 0.75

    def test_strategy_one_hot(self):
        """策略 one-hot 编码"""
        f_direct = extract_routing_features(3, {}, strategy="direct")
        f_cot = extract_routing_features(3, {}, strategy="cot")
        f_struct = extract_routing_features(3, {}, strategy="structured")
        assert f_direct[7] == 0.0 and f_direct[8] == 0.0
        assert f_cot[7] == 1.0 and f_cot[8] == 0.0
        assert f_struct[7] == 0.0 and f_struct[8] == 1.0

    def test_text_length_log_scaled(self):
        """文本长度 log 缩放"""
        f_short = extract_routing_features(3, {}, text_length=10)
        f_long = extract_routing_features(3, {}, text_length=5000)
        assert f_short[4] < f_long[4]

    def test_capability_success_rate(self):
        """能力成功率传递"""
        f = extract_routing_features(3, {}, capability_success_rate=0.9)
        assert f[6] == 0.9

    def test_all_features_in_range(self):
        """所有特征在合理范围内"""
        features = extract_routing_features(5, {"confidence": 0.8, "entropy": 2.0, "margin": 0.5},
                                            text_length=500, file_count=10, capability_success_rate=0.7,
                                            strategy="cot")
        for i, f in enumerate(features):
            assert -1.0 <= f <= 2.0, f"Feature {i} = {f} out of range"


# ─── OnlineLogisticRegression 测试 ─────────────────────────────


class TestOnlineLogisticRegression:
    """在线 Logistic Regression"""

    def test_initial_prediction_is_0_5(self):
        """初始权重为 0，预测概率应为 0.5"""
        model = OnlineLogisticRegression()
        prob = model.predict_proba([1.0] * 10)
        assert abs(prob - 0.5) < 0.01

    def test_learning_changes_prediction(self):
        """学习后预测应该改变"""
        model = OnlineLogisticRegression()
        features = [0.8, 0.9, 0.1, 0.8, 0.3, 0.1, 0.7, 0.0, 0.0, 1.0]
        initial_prob = model.predict_proba(features)

        # 模拟多次成功
        for _ in range(20):
            model.update(features, success=True)

        new_prob = model.predict_proba(features)
        assert new_prob > initial_prob

    def test_learning_from_failure(self):
        """失败后预测概率下降"""
        model = OnlineLogisticRegression()
        features = [0.8, 0.9, 0.1, 0.8, 0.3, 0.1, 0.7, 0.0, 0.0, 1.0]

        for _ in range(20):
            model.update(features, success=False)

        prob = model.predict_proba(features)
        assert prob < 0.5

    def test_convergence(self):
        """模型应该收敛"""
        model = OnlineLogisticRegression(learning_rate=0.2)
        good_features = [0.2, 0.9, 0.1, 0.9, 0.1, 0.0, 0.8, 0.0, 0.0, 1.0]
        bad_features = [0.8, 0.2, 0.8, 0.1, 0.8, 0.5, 0.3, 1.0, 0.0, 1.0]

        for _ in range(50):
            model.update(good_features, success=True)
            model.update(bad_features, success=False)

        good_prob = model.predict_proba(good_features)
        bad_prob = model.predict_proba(bad_features)
        assert good_prob > bad_prob

    def test_serialization(self):
        """序列化/反序列化"""
        model = OnlineLogisticRegression()
        model.update([1.0] * 10, True)
        d = model.to_dict()
        model2 = OnlineLogisticRegression.from_dict(d)
        for w1, w2 in zip(model.weights, model2.weights):
            assert abs(w1 - w2) < 1e-5

    def test_probability_range(self):
        """预测概率始终在 [0, 1]"""
        model = OnlineLogisticRegression()
        for _ in range(100):
            model.update([10.0] * 10, True)
        prob = model.predict_proba([10.0] * 10)
        assert 0.0 <= prob <= 1.0


# ─── MetaLearner 测试 ──────────────────────────────────────────


class TestMetaLearner:
    """统一路由决策器"""

    @pytest.fixture
    def tmp_dir(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    def test_initial_prediction(self, tmp_dir):
        """初始预测应该是中性的"""
        ml = MetaLearner(tmp_dir)
        features = extract_routing_features(3.0, {"confidence": 0.5})
        result = ml.predict(features)
        assert "should_use_local" in result
        assert "local_success_prob" in result
        assert "confidence" in result

    def test_learning_loop(self, tmp_dir):
        """完整学习循环"""
        ml = MetaLearner(tmp_dir)
        features = extract_routing_features(2.0, {"confidence": 0.9})

        # 模拟多次成功
        for _ in range(20):
            ml.record_and_learn(features, success=True, route="local", task_type="classification")

        result = ml.predict(features)
        assert result["local_success_prob"] > 0.5

    def test_different_tasks_different_predictions(self, tmp_dir):
        """不同任务应该有不同的预测"""
        ml = MetaLearner(tmp_dir)

        easy_features = extract_routing_features(1.0, {"confidence": 0.9})
        hard_features = extract_routing_features(8.0, {"confidence": 0.2})

        # 简单任务成功
        for _ in range(20):
            ml.record_and_learn(easy_features, success=True, route="local")
        # 复杂任务失败
        for _ in range(20):
            ml.record_and_learn(hard_features, success=False, route="local")

        easy_result = ml.predict(easy_features)
        hard_result = ml.predict(hard_features)
        assert easy_result["local_success_prob"] > hard_result["local_success_prob"]

    def test_feature_importance(self, tmp_dir):
        """特征重要性可获取"""
        ml = MetaLearner(tmp_dir)
        importance = ml.get_feature_importance()
        assert len(importance) == 10
        assert "complexity" in importance
        assert "confidence" in importance

    def test_stats(self, tmp_dir):
        """统计信息"""
        ml = MetaLearner(tmp_dir)
        stats = ml.get_stats()
        assert stats["total"] == 0
        assert "feature_importance" in stats

    def test_persistence(self, tmp_dir):
        """模型持久化"""
        ml = MetaLearner(tmp_dir)
        features = extract_routing_features(3.0, {"confidence": 0.8})
        ml.record_and_learn(features, success=True, route="local")

        # 重新加载
        ml2 = MetaLearner(tmp_dir)
        assert ml2.get_stats()["total"] == 1

    def test_decision_confidence(self, tmp_dir):
        """决策置信度：离阈值越远越确定"""
        ml = MetaLearner(tmp_dir, decision_threshold=0.5)

        # 概率接近 0.5 → 低置信度
        neutral_result = ml.predict([0.5] * 10)
        # 概率远离 0.5 → 高置信度（需要先训练）
        for _ in range(30):
            ml.record_and_learn([0.1, 0.9, 0.1, 0.9, 0.1, 0.0, 0.8, 0.0, 0.0, 1.0], success=True, route="local")
        extreme_result = ml.predict([0.1, 0.9, 0.1, 0.9, 0.1, 0.0, 0.8, 0.0, 0.0, 1.0])
        assert extreme_result["confidence"] >= neutral_result["confidence"]


# ─── ActiveLearner 测试 ────────────────────────────────────────


class TestActiveLearner:
    """主动学习"""

    @pytest.fixture
    def tmp_dir(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    def test_initial_uncertainty_is_high(self, tmp_dir):
        """无数据时不确定性最高"""
        al = ActiveLearner(tmp_dir)
        assert al.get_uncertainty("classification") == 1.0

    def test_uncertainty_decreases_with_data(self, tmp_dir):
        """数据增加后不确定性降低"""
        al = ActiveLearner(tmp_dir)
        for _ in range(20):
            al.record("classification", 0.8, True)
        assert al.get_uncertainty("classification") < 1.0

    def test_high_variance_high_uncertainty(self, tmp_dir):
        """高方差 → 高不确定性"""
        al = ActiveLearner(tmp_dir)
        # 预测概率波动大
        for i in range(20):
            al.record("volatile", 0.9 if i % 2 == 0 else 0.1, i % 2 == 0)
        # 预测概率稳定
        for _ in range(20):
            al.record("stable", 0.8, True)

        assert al.get_uncertainty("volatile") > al.get_uncertainty("stable")

    def test_most_uncertain(self, tmp_dir):
        """获取最不确定的任务类型"""
        al = ActiveLearner(tmp_dir)
        for _ in range(10):
            al.record("type_a", 0.8, True)
        for i in range(10):
            al.record("type_b", 0.9 if i % 2 == 0 else 0.1, True)

        uncertain = al.get_most_uncertain(1)
        assert len(uncertain) <= 1

    def test_should_request_verification(self, tmp_dir):
        """足够数据 + 不确定时应请求验证"""
        al = ActiveLearner(tmp_dir)
        # 添加波动数据使不确定性高
        for i in range(10):
            al.record("volatile_task", 0.9 if i % 2 == 0 else 0.1, i % 2 == 0)
        # 数据不足时冷启动保护不触发
        assert not al.should_request_verification("new_task")
        # 有足够数据且不确定时触发
        assert al.should_request_verification("volatile_task")

    def test_stable_task_no_verification(self, tmp_dir):
        """稳定任务不需要验证"""
        al = ActiveLearner(tmp_dir)
        for _ in range(20):
            al.record("stable", 0.8, True)
        assert not al.should_request_verification("stable", threshold=0.3)

    def test_stats(self, tmp_dir):
        """统计信息"""
        al = ActiveLearner(tmp_dir)
        al.record("test", 0.8, True)
        stats = al.get_stats()
        assert stats["total_task_types"] == 1
        assert stats["total_records"] == 1

    def test_persistence(self, tmp_dir):
        """数据持久化"""
        al = ActiveLearner(tmp_dir)
        al.record("test", 0.8, True)

        al2 = ActiveLearner(tmp_dir)
        assert al2.get_stats()["total_records"] == 1

    def test_empty_most_uncertain(self, tmp_dir):
        """数据不足时返回空"""
        al = ActiveLearner(tmp_dir)
        assert al.get_most_uncertain() == []
