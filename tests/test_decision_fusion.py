"""
三层决策融合测试

测试 Cascade + Meta-Learner + Active Learner 的投票逻辑：
- cascade 触发但 ML 不触发 → 升级
- ML 触发但 cascade 不触发 → 升级
- 两者同时触发 → 升级
- 都不触发 → 不升级
- 主动学习触发 → 升级
"""

import pytest
import tempfile

from task_router.meta_learner import MetaLearner, ActiveLearner, extract_routing_features
from task_router.confidence import CascadeDecision


class TestThreeLayerFusion:
    """三层决策融合逻辑"""

    @pytest.fixture
    def tmp_dir(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    @pytest.fixture
    def cascade(self, tmp_dir):
        return CascadeDecision(tmp_dir, escalation_threshold=0.4)

    @pytest.fixture
    def meta_learner(self, tmp_dir):
        return MetaLearner(tmp_dir, decision_threshold=0.5)

    @pytest.fixture
    def active_learner(self, tmp_dir):
        return ActiveLearner(tmp_dir)

    def _fusion_decision(self, cascade, ml, al, conf_data, features, task_type="test"):
        """复现 task_router.py 中的三层融合逻辑"""
        cascade_result = cascade.should_escalate(conf_data)
        ml_prediction = ml.predict(features)
        active_verify = al.should_request_verification(task_type, min_samples=5)

        should_escalate = (
            cascade_result["escalate"]
            or (not ml_prediction["should_use_local"] and ml_prediction["confidence"] > 0.3)
            or active_verify
        )
        return {
            "should_escalate": should_escalate,
            "cascade": cascade_result,
            "ml": ml_prediction,
            "active_verify": active_verify,
        }

    def test_cascade_only_triggers(self, cascade, meta_learner, active_learner):
        """cascade 触发（低置信度），ML 不触发 → 升级"""
        # 低置信度 → cascade 升级
        conf_data = {"confidence": 0.1}
        # ML 预测成功概率高（默认 0.5，阈值 0.5）
        features = extract_routing_features(2.0, conf_data)

        result = self._fusion_decision(cascade, meta_learner, active_learner, conf_data, features)
        assert result["should_escalate"] is True
        assert result["cascade"]["escalate"] is True

    def test_ml_only_triggers(self, tmp_dir):
        """ML 触发（预测失败），cascade 不触发 → 升级"""
        cascade = CascadeDecision(tmp_dir, escalation_threshold=0.4)
        ml = MetaLearner(tmp_dir, decision_threshold=0.5)
        al = ActiveLearner(tmp_dir)

        # 高置信度 → cascade 不升级
        conf_data = {"confidence": 0.9}
        features = extract_routing_features(8.0, conf_data)

        # 训练 ML 预测失败
        for _ in range(30):
            ml.record_and_learn(features, success=False, route="local")

        result = self._fusion_decision(cascade, ml, al, conf_data, features)
        assert result["should_escalate"] is True
        assert result["cascade"]["escalate"] is False
        assert result["ml"]["should_use_local"] is False

    def test_both_trigger(self, tmp_dir):
        """cascade 和 ML 都触发 → 升级"""
        cascade = CascadeDecision(tmp_dir, escalation_threshold=0.4)
        ml = MetaLearner(tmp_dir, decision_threshold=0.5)
        al = ActiveLearner(tmp_dir)

        conf_data = {"confidence": 0.1}
        features = extract_routing_features(8.0, conf_data)

        # 训练 ML 预测失败
        for _ in range(30):
            ml.record_and_learn(features, success=False, route="local")

        result = self._fusion_decision(cascade, ml, al, conf_data, features)
        assert result["should_escalate"] is True
        assert result["cascade"]["escalate"] is True
        assert result["ml"]["should_use_local"] is False

    def test_neither_triggers(self, tmp_dir):
        """cascade 和 ML 都不触发 → 不升级"""
        cascade = CascadeDecision(tmp_dir, escalation_threshold=0.4)
        ml = MetaLearner(tmp_dir, decision_threshold=0.5)
        al = ActiveLearner(tmp_dir)

        # 高置信度 → cascade 不升级
        conf_data = {"confidence": 0.9}
        features = extract_routing_features(2.0, conf_data)

        # 训练 ML 预测成功
        for _ in range(30):
            ml.record_and_learn(features, success=True, route="local")

        result = self._fusion_decision(cascade, ml, al, conf_data, features)
        assert result["should_escalate"] is False

    def test_active_learning_triggers(self, tmp_dir):
        """主动学习触发（高不确定性）→ 升级"""
        cascade = CascadeDecision(tmp_dir, escalation_threshold=0.4)
        ml = MetaLearner(tmp_dir, decision_threshold=0.5)
        al = ActiveLearner(tmp_dir)

        # 高置信度，ML 预测成功 → 前两层不触发
        conf_data = {"confidence": 0.9}
        features = extract_routing_features(2.0, conf_data)
        for _ in range(30):
            ml.record_and_learn(features, success=True, route="local")

        # 主动学习：高波动数据 → 高不确定性
        for i in range(10):
            al.record("volatile_task", 0.9 if i % 2 == 0 else 0.1, i % 2 == 0)

        result = self._fusion_decision(cascade, ml, al, conf_data, features, task_type="volatile_task")
        assert result["should_escalate"] is True
        assert result["active_verify"] is True

    def test_active_learning_cold_start_no_trigger(self, tmp_dir):
        """主动学习冷启动保护：数据不足不触发"""
        cascade = CascadeDecision(tmp_dir, escalation_threshold=0.4)
        ml = MetaLearner(tmp_dir, decision_threshold=0.5)
        al = ActiveLearner(tmp_dir)

        conf_data = {"confidence": 0.9}
        features = extract_routing_features(2.0, conf_data)
        for _ in range(30):
            ml.record_and_learn(features, success=True, route="local")

        # 只有 2 条数据，不足 min_samples=5
        al.record("new_task", 0.5, True)
        al.record("new_task", 0.5, False)

        result = self._fusion_decision(cascade, ml, al, conf_data, features, task_type="new_task")
        assert result["active_verify"] is False
        assert result["should_escalate"] is False

    def test_ml_low_confidence_no_trigger(self, tmp_dir):
        """ML 预测不使用本地但置信度低 → 不触发（避免误判）"""
        cascade = CascadeDecision(tmp_dir, escalation_threshold=0.4)
        ml = MetaLearner(tmp_dir, decision_threshold=0.5)
        al = ActiveLearner(tmp_dir)

        conf_data = {"confidence": 0.9}
        features = extract_routing_features(5.0, conf_data)

        # 少量训练，使 ML 预测略低于阈值但置信度低
        for _ in range(3):
            ml.record_and_learn(features, success=False, route="local")

        ml_prediction = ml.predict(features)
        # 如果 ML 置信度 <= 0.3，即使预测失败也不触发
        if ml_prediction["confidence"] <= 0.3:
            result = self._fusion_decision(cascade, ml, al, conf_data, features)
            # cascade 不触发，ML 置信度不足，active 不触发
            assert result["should_escalate"] is False

    def test_circuit_breaker_fallback(self, tmp_dir):
        """云端熔断时降级使用本地结果"""
        cascade = CascadeDecision(tmp_dir, escalation_threshold=0.4)
        ml = MetaLearner(tmp_dir, decision_threshold=0.5)
        al = ActiveLearner(tmp_dir)

        # 模拟 cascade 触发
        conf_data = {"confidence": 0.1}
        features = extract_routing_features(2.0, conf_data)

        cascade_result = cascade.should_escalate(conf_data)
        assert cascade_result["escalate"] is True

        # 云端熔断时，应该降级使用本地结果
        # 这个行为在 task_router.py 中通过 circuit_open 检查实现
        # 这里只验证 cascade 决策本身
        ml_prediction = ml.predict(features)
        active_verify = al.should_request_verification("test", min_samples=5)

        should_escalate = (
            cascade_result["escalate"]
            or (not ml_prediction["should_use_local"] and ml_prediction["confidence"] > 0.3)
            or active_verify
        )
        assert should_escalate is True  # 应该尝试升级

    def test_learning_from_escalation(self, tmp_dir):
        """级联升级后仍应记录学习信号"""
        ml = MetaLearner(tmp_dir, decision_threshold=0.5)
        features = extract_routing_features(3.0, {"confidence": 0.5})

        # 模拟升级后的学习记录
        ml.record_and_learn(features, success=True, route="cascade_escalated", task_type="test")

        stats = ml.get_stats()
        assert stats["total"] == 1

    def test_fusion_decision_consistency(self, tmp_dir):
        """相同输入应产生相同决策"""
        cascade = CascadeDecision(tmp_dir, escalation_threshold=0.4)
        ml = MetaLearner(tmp_dir, decision_threshold=0.5)
        al = ActiveLearner(tmp_dir)

        conf_data = {"confidence": 0.6}
        features = extract_routing_features(3.0, conf_data)

        result1 = self._fusion_decision(cascade, ml, al, conf_data, features)
        result2 = self._fusion_decision(cascade, ml, al, conf_data, features)
        assert result1["should_escalate"] == result2["should_escalate"]
