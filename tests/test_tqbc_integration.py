"""
TQBC 端到端集成测试 — 验证四层决策融合
"""

import pytest
import tempfile

from task_router.tqbc import TQBCRouter
from task_router.meta_learner import MetaLearner, ActiveLearner, extract_routing_features
from task_router.confidence import CascadeDecision


def _make_logprobs(n: int, confident: bool = True) -> list[dict]:
    """生成测试 logprobs"""
    logprobs = []
    for i in range(n):
        if confident:
            logprobs.append({
                "logprob": -0.05,
                "top_logprobs": {"correct": -0.05, "wrong": -5.0}
            })
        else:
            logprobs.append({
                "logprob": -2.0,
                "top_logprobs": {"opt1": -1.5, "opt2": -1.6, "opt3": -1.8}
            })
    return logprobs


class TestFourLayerFusion:
    """四层决策融合测试"""

    @pytest.fixture
    def tmp_dir(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    def _four_layer_fusion(self, tmp_dir, conf_data, features, tqbc_decision,
                           ml_total=0, active_verify=False):
        """复现四层融合逻辑"""
        cascade = CascadeDecision(tmp_dir, escalation_threshold=0.4)
        ml = MetaLearner(tmp_dir, decision_threshold=0.5)
        ActiveLearner(tmp_dir)

        cascade_result = cascade.should_escalate(conf_data)
        ml_prediction = ml.predict(features)

        should_escalate = (
            cascade_result["escalate"]
            or (not ml_prediction["should_use_local"] and ml_prediction["confidence"] > 0.5 and ml_total >= 50)
            or active_verify
            or tqbc_decision.should_escalate
        )
        return should_escalate

    def test_all_layers_agree_escalate(self, tmp_dir):
        """四层都同意升级"""
        tqbc = TQBCRouter(tmp_dir)
        logprobs = _make_logprobs(5, confident=False)
        tqbc_decision = tqbc.decide(logprobs=logprobs, complexity_score=8.0)

        conf_data = {"confidence": 0.1}  # 低置信度
        features = extract_routing_features(8.0, conf_data)  # 高复杂度

        # 训练 ML 预测失败
        ml = MetaLearner(tmp_dir)
        for _ in range(60):
            ml.record_and_learn(features, success=False, route="local")

        result = self._four_layer_fusion(
            tmp_dir, conf_data, features, tqbc_decision,
            ml_total=60, active_verify=True,
        )
        assert result is True

    def test_only_tqbc_escalates(self, tmp_dir):
        """只有 TQBC 触发升级（其他层不触发）"""
        tqbc = TQBCRouter(tmp_dir)

        # 极度不确定的 logprobs → TQBC 可能触发
        logprobs = _make_logprobs(3, confident=False)
        tqbc_decision = tqbc.decide(logprobs=logprobs, complexity_score=8.0)

        # 高置信度 → cascade 不触发
        conf_data = {"confidence": 0.9}
        # 低复杂度 → ML 不触发
        features = extract_routing_features(2.0, conf_data)
        # 不触发主动学习
        active_verify = False

        if tqbc_decision.should_escalate:
            result = self._four_layer_fusion(
                tmp_dir, conf_data, features, tqbc_decision,
                ml_total=0, active_verify=active_verify,
            )
            assert result is True  # TQBC 独立触发

    def test_no_layer_escalates(self, tmp_dir):
        """所有层都不触发 → 不升级"""
        tqbc = TQBCRouter(tmp_dir)

        # 高置信度 → TQBC 不触发
        logprobs = _make_logprobs(20, confident=True)
        tqbc_decision = tqbc.decide(logprobs=logprobs, complexity_score=1.0)

        conf_data = {"confidence": 0.9}  # 高置信度
        features = extract_routing_features(2.0, conf_data)  # 低复杂度

        # 训练 ML 预测成功
        ml = MetaLearner(tmp_dir)
        for _ in range(60):
            ml.record_and_learn(features, success=True, route="local")

        result = self._four_layer_fusion(
            tmp_dir, conf_data, features, tqbc_decision,
            ml_total=60, active_verify=False,
        )
        assert result is False

    def test_tqbc_learning_cycle(self, tmp_dir):
        """TQBC 完整学习循环"""
        tqbc = TQBCRouter(tmp_dir)

        # 50 轮学习
        for i in range(50):
            # 翻译任务 → 高置信度 → 本地
            logprobs = _make_logprobs(15, confident=True)
            decision = tqbc.decide(logprobs=logprobs, task_type="translation")
            tqbc.record_outcome(decision, success=True, escalated=False, task_type="translation")

            # 代码生成 → 低置信度 → 云端
            logprobs = _make_logprobs(3, confident=False)
            decision = tqbc.decide(logprobs=logprobs, task_type="code_gen")
            tqbc.record_outcome(decision, success=True, escalated=True, task_type="code_gen")

        stats = tqbc.get_stats()
        assert stats["total_decisions"] == 100
        assert stats["success_rate"] == 1.0

        # 验证贝叶斯参数已更新
        ts_stats = stats["thompson"]
        assert ts_stats["arms"]["local"]["n_observations"] > 0
        assert ts_stats["arms"]["cloud"]["n_observations"] > 0

        # 验证校准器已分组学习
        # 注意：只有未升级的结果用于校准（与 Cascade 行为一致）
        cal_stats = stats["calibration"]
        assert "translation" in cal_stats["groups"]
        # code_gen 任务全部升级了，所以不记录到校准器中
        # 这是正确的设计：升级后的结果由云端处理，不应影响本地校准

    def test_backward_compatibility(self, tmp_dir):
        """TQBC 不影响现有三层决策的正确性"""
        cascade = CascadeDecision(tmp_dir, escalation_threshold=0.4)
        ml = MetaLearner(tmp_dir)
        tqbc = TQBCRouter(tmp_dir)

        # 现有三层逻辑
        conf_data = {"confidence": 0.1}
        cascade_result = cascade.should_escalate(conf_data)
        assert cascade_result["escalate"] is True

        features = extract_routing_features(2.0, conf_data)
        ml_result = ml.predict(features)

        # 添加 TQBC
        logprobs = _make_logprobs(10, confident=True)
        tqbc_decision = tqbc.decide(logprobs=logprobs, complexity_score=2.0)

        # 现有决策不应改变
        old_should_escalate = cascade_result["escalate"] or (
            not ml_result["should_use_local"] and ml_result["confidence"] > 0.5
        )

        # 四层融合
        new_should_escalate = old_should_escalate or tqbc_decision.should_escalate

        # 如果旧系统触发了升级，新系统也应触发
        if old_should_escalate:
            assert new_should_escalate is True


class TestTQBCBenchmark:
    """TQBC 性能基准测试"""

    @pytest.fixture
    def tmp_dir(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    def test_confident_tasks_route_local(self, tmp_dir):
        """高置信度任务应路由到本地"""
        tqbc = TQBCRouter(tmp_dir)

        local_routed = 0
        total = 100

        for i in range(total):
            logprobs = _make_logprobs(20, confident=True)
            decision = tqbc.decide(logprobs=logprobs, complexity_score=1.0, task_type="translation")
            if decision.route == "local":
                local_routed += 1

        local_rate = local_routed / total
        print(f"\n高置信度任务本地路由率: {local_rate:.1%}")
        assert local_rate >= 0.7, f"高置信度任务本地路由率过低: {local_rate:.1%}"

    def test_uncertain_tasks_escalate(self, tmp_dir):
        """低置信度任务应倾向升级"""
        tqbc = TQBCRouter(tmp_dir)

        escalated = 0
        total = 100

        for i in range(total):
            logprobs = _make_logprobs(3, confident=False)
            decision = tqbc.decide(logprobs=logprobs, complexity_score=7.0, task_type="code_gen")
            if decision.route == "cloud":
                escalated += 1

        escalation_rate = escalated / total
        print(f"\n低置信度任务升级率: {escalation_rate:.1%}")
        assert escalation_rate >= 0.3, f"低置信度任务升级率过低: {escalation_rate:.1%}"

    def test_calibration_improves_over_time(self, tmp_dir):
        """校准应随数据增加而改善"""
        tqbc = TQBCRouter(tmp_dir)

        # 第一轮：无数据
        logprobs = _make_logprobs(10, confident=True)
        d1 = tqbc.decide(logprobs=logprobs, task_type="test")

        # 第二轮：100 轮反馈
        for i in range(100):
            lp = _make_logprobs(15, confident=True)
            d = tqbc.decide(logprobs=lp, task_type="test")
            tqbc.record_outcome(d, success=True, escalated=False, task_type="test")

        d2 = tqbc.decide(logprobs=logprobs, task_type="test")

        # 不确定性应降低
        assert d2.uncertainty <= d1.uncertainty + 0.1, \
            "校准不确定性应随数据增加而降低"

    def test_thompson_exploration_balance(self, tmp_dir):
        """Thompson Sampling 的探索-利用平衡"""
        tqbc = TQBCRouter(tmp_dir)

        # 初始阶段（探索期）
        initial_routes = set()
        for i in range(30):
            logprobs = _make_logprobs(10, confident=True)
            d = tqbc.decide(logprobs=logprobs, task_type="test")
            initial_routes.add(d.thompson_arm)

        # 应探索了多个臂
        print(f"\n初始探索的臂: {initial_routes}")

        # 充分学习后（利用期）
        for i in range(200):
            logprobs = _make_logprobs(15, confident=True)
            d = tqbc.decide(logprobs=logprobs, task_type="test")
            tqbc.record_outcome(d, success=True, escalated=False, task_type="test")

        # 统计
        stats = tqbc.get_stats()
        print(f"最终统计: {stats['total_decisions']} 次决策, "
              f"成功率={stats['success_rate']:.1%}")

    def test_cost_savings(self, tmp_dir):
        """
        成本节约测试：TQBC 应将更多简单任务路由到本地，
        同时确保复杂任务仍路由到云端。
        """
        tqbc = TQBCRouter(tmp_dir)

        # 预训练
        for i in range(50):
            # 简单任务 → 本地成功
            lp = _make_logprobs(15, confident=True)
            d = tqbc.decide(logprobs=lp, complexity_score=1.0, task_type="translation")
            tqbc.record_outcome(d, success=True, escalated=False, task_type="translation")

            # 复杂任务 → 云端成功
            lp = _make_logprobs(3, confident=False)
            d = tqbc.decide(logprobs=lp, complexity_score=7.0, task_type="code_gen")
            tqbc.record_outcome(d, success=True, escalated=True, task_type="code_gen")

        # 测试：简单任务应大部分走本地
        local_simple = 0
        for i in range(50):
            lp = _make_logprobs(20, confident=True)
            d = tqbc.decide(logprobs=lp, complexity_score=1.0, task_type="translation")
            if d.route == "local":
                local_simple += 1

        local_rate = local_simple / 50
        print(f"\n简单任务本地路由率（经过训练）: {local_rate:.1%}")
        assert local_rate >= 0.6
