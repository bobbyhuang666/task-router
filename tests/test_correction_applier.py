"""
参数修正应用器测试

测试覆盖:
- 阈值修正 (a3m_weights.json)
- 策略权重修正 (strategy_params.json)
- 路由策略修正 (routing_policies.json)
- 置信度过滤（自动应用 / 待审批 / 忽略）
- 审批流程
- 回滚
- 安全限制
"""

import json
import os
import pytest
import tempfile

from task_router.correction_applier import (
    CorrectionApplier,
    AUTO_APPLY_CONFIDENCE,
    MIN_CONFIDENCE,
    MAX_THRESHOLD_CHANGE,
    STATUS_APPLIED,
    STATUS_PENDING,
    STATUS_SKIPPED,
    STATUS_ROLLED_BACK,
)
from task_router.reflection import Correction


# ─── 辅助函数 ──────────────────────────────────────────────


def _make_correction(
    target="threshold",
    parameter="base_threshold",
    new_value=0.3,
    old_value=None,
    confidence=0.85,
    evidence_count=5,
    **overrides,
):
    """创建测试用 Correction"""
    c = Correction(
        target=target,
        parameter=parameter,
        new_value=new_value,
        old_value=old_value,
        confidence=confidence,
        evidence_count=evidence_count,
        reason="测试修正",
    )
    c.__dict__.update(overrides)
    return c


# ─── 阈值修正测试 ──────────────────────────────────────────────


class TestThresholdCorrection:
    """阈值修正 (a3m_weights.json) 测试"""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.applier = CorrectionApplier(cache_dir=self.tmpdir)

    def test_apply_positive_threshold_change(self):
        """正增量 → 提高阈值"""
        # 初始权重
        weights_file = os.path.join(self.tmpdir, "a3m_weights.json")
        with open(weights_file, "w") as f:
            json.dump({"base_threshold": 3.0}, f)

        correction = _make_correction(new_value=0.3)
        result = self.applier.apply_corrections([correction])

        assert result["applied"] == 1
        assert result["skipped"] == 0

        with open(weights_file) as f:
            data = json.load(f)
        assert data["base_threshold"] == pytest.approx(3.3, abs=0.01)

    def test_apply_negative_threshold_change(self):
        """负增量 → 降低阈值"""
        weights_file = os.path.join(self.tmpdir, "a3m_weights.json")
        with open(weights_file, "w") as f:
            json.dump({"base_threshold": 3.0}, f)

        correction = _make_correction(new_value=-0.2)
        result = self.applier.apply_corrections([correction])

        assert result["applied"] == 1
        with open(weights_file) as f:
            data = json.load(f)
        assert data["base_threshold"] == pytest.approx(2.8, abs=0.01)

    def test_threshold_clamp_max(self):
        """阈值修正幅度限制（不超过 ±0.5）"""
        weights_file = os.path.join(self.tmpdir, "a3m_weights.json")
        with open(weights_file, "w") as f:
            json.dump({"base_threshold": 3.0}, f)

        correction = _make_correction(new_value=10.0)  # 超大增量
        self.applier.apply_corrections([correction])

        with open(weights_file) as f:
            data = json.load(f)
        # 最多 +0.5
        assert data["base_threshold"] == pytest.approx(3.5, abs=0.01)

    def test_threshold_range_limits(self):
        """阈值不超过 [1.0, 8.0] 范围"""
        weights_file = os.path.join(self.tmpdir, "a3m_weights.json")
        with open(weights_file, "w") as f:
            json.dump({"base_threshold": 7.8}, f)

        correction = _make_correction(new_value=0.5)
        self.applier.apply_corrections([correction])

        with open(weights_file) as f:
            data = json.load(f)
        assert data["base_threshold"] <= 8.0

    def test_threshold_creates_file_if_missing(self):
        """a3m_weights.json 不存在时自动创建"""
        correction = _make_correction(new_value=0.2)
        result = self.applier.apply_corrections([correction])
        assert result["applied"] == 1

        weights_file = os.path.join(self.tmpdir, "a3m_weights.json")
        assert os.path.exists(weights_file)


# ─── 策略权重修正测试 ──────────────────────────────────────────────


class TestStrategyWeightCorrection:
    """策略权重修正 (strategy_params.json) 测试"""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.applier = CorrectionApplier(cache_dir=self.tmpdir)

    def test_apply_strategy_weight(self):
        """应用策略权重修正"""
        correction = _make_correction(
            target="strategy_weight",
            parameter="prefer_lightweight_strategy",
            new_value=True,
        )
        result = self.applier.apply_corrections([correction])
        assert result["applied"] == 1

        params_file = os.path.join(self.tmpdir, "strategy_params.json")
        with open(params_file) as f:
            data = json.load(f)
        assert data["prefer_lightweight_strategy"] is True
        assert "_last_updated" in data

    def test_apply_per_task_type_strategy(self):
        """按任务类型设置策略偏好"""
        correction = _make_correction(
            target="strategy_weight",
            parameter="task_type.classify.preferred_strategy",
            new_value="cod",
        )
        result = self.applier.apply_corrections([correction])
        assert result["applied"] == 1

        params_file = os.path.join(self.tmpdir, "strategy_params.json")
        with open(params_file) as f:
            data = json.load(f)
        assert data["task_type.classify.preferred_strategy"] == "cod"


# ─── 路由策略修正测试 ──────────────────────────────────────────────


class TestRoutingPolicyCorrection:
    """路由策略修正 (routing_policies.json) 测试"""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.applier = CorrectionApplier(cache_dir=self.tmpdir)

    def test_apply_routing_policy(self):
        """应用路由策略修正"""
        correction = _make_correction(
            target="routing_policy",
            parameter="task_type.法律.prefer_route",
            new_value="cloud",
        )
        result = self.applier.apply_corrections([correction])
        assert result["applied"] == 1

        policies_file = os.path.join(self.tmpdir, "routing_policies.json")
        with open(policies_file) as f:
            data = json.load(f)
        assert data["task_type.法律.prefer_route"] == "cloud"


# ─── 置信度过滤测试 ──────────────────────────────────────────────


class TestConfidenceFiltering:
    """置信度过滤机制测试"""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.applier = CorrectionApplier(cache_dir=self.tmpdir)

    def test_high_confidence_auto_applied(self):
        """高置信度 → 自动应用"""
        correction = _make_correction(confidence=0.85)
        result = self.applier.apply_corrections([correction])
        assert result["applied"] == 1
        assert result["pending"] == 0
        assert result["skipped"] == 0

    def test_medium_confidence_pending(self):
        """中等置信度 → 待审批"""
        correction = _make_correction(confidence=0.7)
        result = self.applier.apply_corrections([correction])
        assert result["applied"] == 0
        assert result["pending"] == 1
        assert result["skipped"] == 0

    def test_low_confidence_skipped(self):
        """低置信度 → 忽略"""
        correction = _make_correction(confidence=0.5)
        result = self.applier.apply_corrections([correction])
        assert result["applied"] == 0
        assert result["pending"] == 0
        assert result["skipped"] == 1

    def test_mixed_confidence_corrections(self):
        """混合置信度修正"""
        corrections = [
            _make_correction(confidence=0.9, parameter="p1"),
            _make_correction(confidence=0.7, parameter="p2"),
            _make_correction(confidence=0.4, parameter="p3"),
        ]
        result = self.applier.apply_corrections(corrections)
        assert result["applied"] == 1
        assert result["pending"] == 1
        assert result["skipped"] == 1

    def test_details_contain_reasons(self):
        """details 包含跳过/待审批原因"""
        corrections = [
            _make_correction(confidence=0.5, parameter="p1"),
        ]
        result = self.applier.apply_corrections(corrections)
        assert "置信度" in result["details"][0]["reason"]


# ─── 审批流程测试 ──────────────────────────────────────────────


class TestApprovalWorkflow:
    """审批流程测试"""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.applier = CorrectionApplier(cache_dir=self.tmpdir)

    def test_pending_queue(self):
        """待审批队列管理"""
        correction = _make_correction(confidence=0.7)
        self.applier.apply_corrections([correction])

        pending = self.applier.get_pending()
        assert len(pending) == 1
        assert pending[0]["confidence"] == 0.7

    def test_approve_pending(self):
        """审批通过"""
        weights_file = os.path.join(self.tmpdir, "a3m_weights.json")
        with open(weights_file, "w") as f:
            json.dump({"base_threshold": 3.0}, f)

        correction = _make_correction(confidence=0.7)
        self.applier.apply_corrections([correction])

        pending = self.applier.get_pending()
        assert len(pending) == 1
        cid = pending[0]["correction_id"]

        success = self.applier.approve_pending(cid)
        assert success is True

        # pending 队列应为空
        assert len(self.applier.get_pending()) == 0

    def test_reject_pending(self):
        """拒绝待审批修正"""
        correction = _make_correction(confidence=0.7)
        self.applier.apply_corrections([correction])

        pending = self.applier.get_pending()
        cid = pending[0]["correction_id"]

        success = self.applier.reject_pending(cid)
        assert success is True
        assert len(self.applier.get_pending()) == 0

    def test_approve_nonexistent(self):
        """审批不存在的修正"""
        success = self.applier.approve_pending("nonexistent")
        assert success is False


# ─── 回滚测试 ──────────────────────────────────────────────


class TestRollback:
    """回滚功能测试"""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.applier = CorrectionApplier(cache_dir=self.tmpdir)

    def test_rollback_threshold(self):
        """回滚阈值修正"""
        weights_file = os.path.join(self.tmpdir, "a3m_weights.json")
        with open(weights_file, "w") as f:
            json.dump({"base_threshold": 3.0}, f)

        # 应用修正
        correction = _make_correction(new_value=0.3, confidence=0.9)
        result = self.applier.apply_corrections([correction])
        cid = result["details"][0]["correction_id"]

        with open(weights_file) as f:
            data = json.load(f)
        assert data["base_threshold"] == pytest.approx(3.3, abs=0.01)

        # 回滚
        success = self.applier.rollback(cid)
        assert success is True

    def test_rollback_nonexistent(self):
        """回滚不存在的修正"""
        success = self.applier.rollback("nonexistent_id")
        assert success is False


# ─── 安全限制测试 ──────────────────────────────────────────────


class TestSafetyLimits:
    """安全限制测试"""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.applier = CorrectionApplier(cache_dir=self.tmpdir)

    def test_max_threshold_change_enforced(self):
        """单次阈值修正不超过 ±0.5"""
        weights_file = os.path.join(self.tmpdir, "a3m_weights.json")
        with open(weights_file, "w") as f:
            json.dump({"base_threshold": 3.0}, f)

        # 试图修改 +5.0
        correction = _make_correction(new_value=5.0, confidence=0.9)
        self.applier.apply_corrections([correction])

        with open(weights_file) as f:
            data = json.load(f)
        # 被限制为 +0.5
        assert data["base_threshold"] == pytest.approx(3.5, abs=0.01)

    def test_unknown_target_type_skipped(self):
        """未知修正类型被跳过"""
        correction = _make_correction(target="unknown_type", confidence=0.9)
        result = self.applier.apply_corrections([correction])
        assert result["applied"] == 0
        assert result["skipped"] == 1


# ─── 统计和历史测试 ──────────────────────────────────────────────


class TestStatsAndHistory:
    """统计和历史查询测试"""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.applier = CorrectionApplier(cache_dir=self.tmpdir)

    def test_history_recording(self):
        """修正历史记录"""
        correction = _make_correction(confidence=0.9)
        self.applier.apply_corrections([correction])

        history = self.applier.get_history()
        assert len(history) == 1
        assert history[0]["status"] == STATUS_APPLIED

    def test_stats(self):
        """修正统计"""
        corrections = [
            _make_correction(confidence=0.9, parameter="p1"),
            _make_correction(confidence=0.7, parameter="p2"),
            _make_correction(confidence=0.4, parameter="p3"),
        ]
        self.applier.apply_corrections(corrections)

        stats = self.applier.get_stats()
        assert stats["total"] == 3
        assert stats["by_status"][STATUS_APPLIED] == 1
        assert stats["by_status"][STATUS_PENDING] == 1
        assert stats["by_status"][STATUS_SKIPPED] == 1
        assert stats["pending"] == 1

    def test_correction_history_detail(self):
        """修正历史包含详细信息"""
        correction = _make_correction(
            target="threshold",
            parameter="base_threshold",
            confidence=0.9,
            reason="过度升级率高",
        )
        self.applier.apply_corrections([correction])

        history = self.applier.get_history()
        assert history[0]["target"] == "threshold"
        assert history[0]["reason"] == "过度升级率高"
        assert history[0]["applied_at"] != ""
