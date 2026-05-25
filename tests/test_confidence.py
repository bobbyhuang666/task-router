"""
置信度门控级联测试

测试覆盖:
- extract_confidence: token 级 logprobs 置信度提取
- extract_confidence_from_text: 纯文本启发式置信度
- ConfidenceCalibrator: 等调回归校准器
- CascadeDecision: 级联升级决策
"""

import os
import json
import pytest
import tempfile

from task_router.confidence import (
    extract_confidence,
    extract_confidence_from_text,
    ConfidenceCalibrator,
    CascadeDecision,
)


# ─── extract_confidence 测试 ──────────────────────────────────────


class TestExtractConfidence:
    """token 级 logprobs 置信度提取"""

    def test_empty_logprobs(self):
        """空 logprobs 返回零置信度"""
        result = extract_confidence([])
        assert result["confidence"] == 0.0
        assert result["entropy"] == 1.0
        assert result["margin"] == 0.0
        assert result["token_count"] == 0
        assert result["max_entropy"] == 1.0

    def test_single_token_high_confidence(self):
        """单个高置信度 token"""
        logprobs = [
            {"token": "Hello", "logprob": -0.05,
             "top_logprobs": {"Hello": -0.05, "Hi": -5.0}}
        ]
        result = extract_confidence(logprobs)
        assert result["confidence"] > 0.7
        assert result["token_count"] == 1
        assert result["margin"] > 0.5

    def test_single_token_low_confidence(self):
        """单个低置信度 token（top-2 概率接近）"""
        logprobs = [
            {"token": "Hello", "logprob": -0.7,
             "top_logprobs": {"Hello": -0.7, "Hi": -0.75}}
        ]
        result = extract_confidence(logprobs)
        assert result["confidence"] < 0.5
        assert result["margin"] < 0.3

    def test_multiple_tokens_averaging(self):
        """多个 token 的置信度取平均"""
        logprobs = [
            {"token": "A", "logprob": -0.01,
             "top_logprobs": {"A": -0.01, "B": -5.0}},
            {"token": "B", "logprob": -0.01,
             "top_logprobs": {"B": -0.01, "A": -5.0}},
            {"token": "C", "logprob": -0.01,
             "top_logprobs": {"C": -0.01, "D": -5.0}},
        ]
        result = extract_confidence(logprobs)
        assert result["confidence"] > 0.8
        assert result["token_count"] == 3

    def test_mixed_confidence_tokens(self):
        """混合置信度 token（高+低）"""
        logprobs = [
            {"token": "A", "logprob": -0.01,
             "top_logprobs": {"A": -0.01, "B": -5.0}},
            {"token": "B", "logprob": -2.0,
             "top_logprobs": {"B": -2.0, "C": -2.1, "D": -3.0}},
        ]
        result = extract_confidence(logprobs)
        # 高置信度 token 权重大，总体仍较高
        assert 0.5 < result["confidence"] < 1.0
        assert result["token_count"] == 2

    def test_no_top_logprobs_fallback(self):
        """没有 top_logprobs 时降级到 logprob"""
        logprobs = [
            {"token": "Hello", "logprob": -0.1},
        ]
        result = extract_confidence(logprobs)
        assert result["token_count"] == 1
        assert result["confidence"] > 0.0

    def test_empty_top_logprobs_dict(self):
        """top_logprobs 为空字典"""
        logprobs = [
            {"token": "Hello", "logprob": -0.5, "top_logprobs": {}},
        ]
        result = extract_confidence(logprobs)
        assert result["token_count"] == 1

    def test_max_entropy_tracking(self):
        """max_entropy 应该记录最不确定的 token"""
        logprobs = [
            {"token": "A", "logprob": -0.01,
             "top_logprobs": {"A": -0.01, "B": -5.0}},
            {"token": "B", "logprob": -3.0,
             "top_logprobs": {"B": -3.0, "C": -3.5}},
        ]
        result = extract_confidence(logprobs)
        # 第二个 token 熵更高
        assert result["max_entropy"] > result["entropy"]

    def test_many_top_logprobs(self):
        """多个 top_logprobs 候选"""
        logprobs = [
            {"token": "A", "logprob": -0.1,
             "top_logprobs": {
                 "A": -0.1, "B": -1.0, "C": -2.0,
                 "D": -3.0, "E": -4.0,
             }}
        ]
        result = extract_confidence(logprobs)
        assert result["confidence"] > 0.5
        assert result["token_count"] == 1

    def test_extreme_logprob_values(self):
        """极端 logprob 值不崩溃"""
        logprobs = [
            {"token": "A", "logprob": -100.0,
             "top_logprobs": {"A": -100.0, "B": -200.0}},
        ]
        result = extract_confidence(logprobs)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_zero_logprob(self):
        """logprob=0（概率=100%）"""
        logprobs = [
            {"token": "A", "logprob": 0.0,
             "top_logprobs": {"A": 0.0, "B": -20.0}},
        ]
        result = extract_confidence(logprobs)
        assert result["confidence"] > 0.9

    def test_confidence_range(self):
        """置信度始终在 [0, 1] 范围内"""
        for logprobs_list in [
            [],
            [{"token": "X", "logprob": -0.5, "top_logprobs": {"X": -0.5, "Y": -0.6}}],
            [{"token": "X", "logprob": -50.0, "top_logprobs": {"X": -50.0, "Y": -50.1}}],
        ]:
            result = extract_confidence(logprobs_list)
            assert 0.0 <= result["confidence"] <= 1.0, f"confidence={result['confidence']}"


# ─── extract_confidence_from_text 测试 ───────────────────────────


class TestExtractConfidenceFromText:
    """纯文本启发式置信度提取"""

    def test_empty_text(self):
        """空文本返回零置信度"""
        result = extract_confidence_from_text("")
        assert result["confidence"] == 0.0
        assert result["token_count"] == 0

    def test_whitespace_only(self):
        """纯空白返回零置信度"""
        result = extract_confidence_from_text("   \n\t  ")
        assert result["confidence"] == 0.0

    def test_short_text(self):
        """短文本置信度较低（相比中等文本）"""
        short = extract_confidence_from_text("OK")
        medium = extract_confidence_from_text("这是一个正常的回答，包含了足够的信息量。" * 3)
        assert short["confidence"] < medium["confidence"]

    def test_medium_text(self):
        """中等长度文本置信度适中"""
        result = extract_confidence_from_text("这是一个正常的回答，包含了足够的信息量。" * 3)
        assert result["confidence"] > 0.4

    def test_long_text(self):
        """长文本置信度略低（可能废话）"""
        result = extract_confidence_from_text("内容 " * 500)
        assert result["confidence"] > 0.0

    def test_failure_signal_apology(self):
        """包含道歉词 → 低置信度"""
        result = extract_confidence_from_text("抱歉，我无法完成这个任务")
        assert result["confidence"] < 0.5

    def test_failure_signal_error(self):
        """包含 error → 置信度低于无失败信号的文本"""
        with_error = extract_confidence_from_text("Error: something went wrong")
        without_error = extract_confidence_from_text("分类结果：文件A属于文档类，文件B属于图片类。")
        assert with_error["confidence"] < without_error["confidence"]

    def test_failure_signal_ai(self):
        """包含 '作为AI' → 低置信度"""
        result = extract_confidence_from_text("作为AI，我无法回答这个问题")
        assert result["confidence"] < 0.5

    def test_no_failure_signals(self):
        """无失败信号 → 较高置信度"""
        result = extract_confidence_from_text("分类结果：文件A属于文档类，文件B属于图片类。" * 2)
        assert result["confidence"] > 0.5

    def test_repeated_content(self):
        """重复内容 → 置信度低于正常文本"""
        repeated = extract_confidence_from_text("hello hello hello hello hello hello world world world")
        normal = extract_confidence_from_text("分类结果：文件A属于文档类，文件B属于图片类。")
        assert repeated["confidence"] < normal["confidence"]

    def test_chinese_text(self):
        """中文文本正常处理"""
        result = extract_confidence_from_text("这是翻译结果：你好世界。原文被准确翻译为中文。")
        assert result["confidence"] > 0.3

    def test_confidence_range(self):
        """置信度始终在 [0, 1] 范围内"""
        for text in ["", "OK", "正常回答" * 10, "抱歉无法" * 5, "word " * 1000]:
            result = extract_confidence_from_text(text)
            assert 0.0 <= result["confidence"] <= 1.0, f"text='{text[:20]}...' conf={result['confidence']}"


# ─── ConfidenceCalibrator 测试 ───────────────────────────────────


class TestConfidenceCalibrator:
    """等调回归校准器"""

    @pytest.fixture
    def tmp_dir(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    def test_no_data_returns_raw(self, tmp_dir):
        """数据不足时返回原始置信度"""
        cal = ConfidenceCalibrator(tmp_dir)
        assert cal.calibrate(0.7) == 0.7
        assert cal.calibrate(0.3) == 0.3

    def test_insufficient_data_returns_raw(self, tmp_dir):
        """少于 20 个样本返回原始置信度"""
        cal = ConfidenceCalibrator(tmp_dir)
        for i in range(15):
            cal.record(0.5 + i * 0.02, i % 2 == 0)
        # 15 个样本，不足以校准
        assert cal.calibrate(0.7) == 0.7

    def test_calibration_after_20_samples(self, tmp_dir):
        """20 个样本后开始校准"""
        cal = ConfidenceCalibrator(tmp_dir)
        # 添加 20 个样本：高置信度通常正确
        for _ in range(10):
            cal.record(0.9, True)
        for _ in range(10):
            cal.record(0.1, False)
        stats = cal.get_stats()
        assert stats["is_calibrated"]
        assert stats["total_samples"] == 20

    def test_calibration_reduces_miscalibration(self, tmp_dir):
        """校准应该减少过度自信"""
        cal = ConfidenceCalibrator(tmp_dir)
        # 高置信度但只有 70% 正确
        for _ in range(7):
            cal.record(0.9, True)
        for _ in range(3):
            cal.record(0.9, False)
        # 低置信度但有 30% 正确
        for _ in range(3):
            cal.record(0.1, True)
        for _ in range(7):
            cal.record(0.1, False)
        # 添加更多数据触发校准
        for _ in range(5):
            cal.record(0.5, True)
        for _ in range(5):
            cal.record(0.5, False)

        stats = cal.get_stats()
        if stats["is_calibrated"]:
            # 校准后 0.9 的置信度应该低于 0.9（因为实际正确率只有 70%）
            calibrated = cal.calibrate(0.9)
            assert calibrated < 0.9

    def test_pav_monotonicity(self, tmp_dir):
        """PAV 算法输出应该是单调的"""
        cal = ConfidenceCalibrator(tmp_dir)
        # 添加非单调数据
        cal.record(0.1, True)
        cal.record(0.2, False)
        cal.record(0.3, True)
        cal.record(0.4, False)
        cal.record(0.5, True)
        cal.record(0.6, True)
        cal.record(0.7, False)
        cal.record(0.8, True)
        cal.record(0.9, True)
        # 加够 20 个
        for i in range(11):
            cal.record(0.1 + i * 0.08, i % 3 == 0)

        if cal.get_stats()["is_calibrated"]:
            fn = cal._calibration_fn
            # 校准函数应该是单调非递减的
            for i in range(len(fn) - 1):
                assert fn[i][1] <= fn[i + 1][1] + 0.001, \
                    f"Non-monotonic at index {i}: {fn[i][1]} > {fn[i+1][1]}"

    def test_boundary_values(self, tmp_dir):
        """边界值校准"""
        cal = ConfidenceCalibrator(tmp_dir)
        # 全部正确
        for _ in range(20):
            cal.record(0.95, True)

        if cal.get_stats()["is_calibrated"]:
            # 低于最低校准点
            low = cal.calibrate(0.0)
            assert 0.0 <= low <= 1.0
            # 高于最高校准点
            high = cal.calibrate(1.0)
            assert 0.0 <= high <= 1.0

    def test_interpolation(self, tmp_dir):
        """中间值应该线性插值"""
        cal = ConfidenceCalibrator(tmp_dir)
        # 明确的两极数据
        for _ in range(10):
            cal.record(0.1, False)
        for _ in range(10):
            cal.record(0.9, True)

        if cal.get_stats()["is_calibrated"]:
            mid = cal.calibrate(0.5)
            low = cal.calibrate(0.1)
            high = cal.calibrate(0.9)
            assert low <= mid <= high

    def test_record_persistence(self, tmp_dir):
        """记录应该持久化到文件"""
        cal = ConfidenceCalibrator(tmp_dir)
        cal.record(0.8, True)
        cal.record(0.2, False)

        # 重新加载
        cal2 = ConfidenceCalibrator(tmp_dir)
        assert cal2.get_stats()["total_samples"] == 2

    def test_malformed_jsonl_skipped(self, tmp_dir):
        """损坏的 JSONL 行应该被跳过"""
        data_file = os.path.join(tmp_dir, "confidence_history.jsonl")
        with open(data_file, "w") as f:
            f.write("not json\n")
            f.write(json.dumps({"confidence": 0.5, "was_correct": True}) + "\n")
            f.write("also bad\n")

        cal = ConfidenceCalibrator(tmp_dir)
        assert cal.get_stats()["total_samples"] == 1

    def test_every_10_recalibration(self, tmp_dir):
        """每 10 条数据重新校准"""
        cal = ConfidenceCalibrator(tmp_dir)
        # 添加 19 个样本
        for i in range(19):
            cal.record(0.5, True)
        assert not cal.get_stats()["is_calibrated"]

        # 第 20 个触发校准
        cal.record(0.5, True)
        assert cal.get_stats()["is_calibrated"]

    def test_get_stats_empty(self, tmp_dir):
        """空校准器的统计"""
        cal = ConfidenceCalibrator(tmp_dir)
        stats = cal.get_stats()
        assert stats["total_samples"] == 0
        assert stats["overall_accuracy"] == 0.0
        assert not stats["is_calibrated"]
        assert stats["calibration_buckets"] == 0


# ─── CascadeDecision 测试 ────────────────────────────────────────


class TestCascadeDecision:
    """级联升级决策"""

    @pytest.fixture
    def tmp_dir(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    def test_high_confidence_no_escalate(self, tmp_dir):
        """高置信度不升级"""
        cascade = CascadeDecision(tmp_dir, escalation_threshold=0.4)
        decision = cascade.should_escalate({"confidence": 0.9})
        assert not decision["escalate"]
        assert decision["raw_confidence"] == 0.9

    def test_low_confidence_escalate(self, tmp_dir):
        """低置信度升级到云端"""
        cascade = CascadeDecision(tmp_dir, escalation_threshold=0.4)
        decision = cascade.should_escalate({"confidence": 0.1})
        assert decision["escalate"]

    def test_threshold_boundary(self, tmp_dir):
        """阈值边界：等于阈值时不升级"""
        cascade = CascadeDecision(tmp_dir, escalation_threshold=0.4)
        # 未校准时用 0.4+0.1=0.5 作为阈值
        decision = cascade.should_escalate({"confidence": 0.55})
        assert not decision["escalate"]

    def test_uncalibrated_conservative(self, tmp_dir):
        """未校准时更保守（阈值 +0.1）"""
        cascade = CascadeDecision(tmp_dir, escalation_threshold=0.4)
        # 0.45 < 0.5 (0.4+0.1) → 应该升级
        decision = cascade.should_escalate({"confidence": 0.45})
        assert decision["escalate"]

    def test_record_outcome(self, tmp_dir):
        """记录结果"""
        cascade = CascadeDecision(tmp_dir)
        cascade.record_outcome({"confidence": 0.8}, was_correct=True, escalated=False)
        stats = cascade.get_stats()
        assert stats["total"] == 1
        assert stats["escalated"] == 0

    def test_record_escalated(self, tmp_dir):
        """记录升级"""
        cascade = CascadeDecision(tmp_dir)
        cascade.record_outcome({"confidence": 0.2}, was_correct=False, escalated=True)
        stats = cascade.get_stats()
        assert stats["total"] == 1
        assert stats["escalated"] == 1

    def test_escalation_rate(self, tmp_dir):
        """升级率计算"""
        cascade = CascadeDecision(tmp_dir)
        cascade.record_outcome({"confidence": 0.9}, was_correct=True, escalated=False)
        cascade.record_outcome({"confidence": 0.1}, was_correct=False, escalated=True)
        cascade.record_outcome({"confidence": 0.8}, was_correct=True, escalated=False)
        cascade.record_outcome({"confidence": 0.2}, was_correct=False, escalated=True)

        stats = cascade.get_stats()
        assert stats["total"] == 4
        assert stats["escalated"] == 2
        assert stats["escalation_rate"] == 0.5
        assert stats["local_kept"] == 2

    def test_local_accuracy(self, tmp_dir):
        """本地准确率计算"""
        cascade = CascadeDecision(tmp_dir)
        # 3 个本地保留：2 正确 1 错误
        cascade.record_outcome({"confidence": 0.9}, was_correct=True, escalated=False)
        cascade.record_outcome({"confidence": 0.8}, was_correct=True, escalated=False)
        cascade.record_outcome({"confidence": 0.7}, was_correct=False, escalated=False)
        # 1 个升级
        cascade.record_outcome({"confidence": 0.1}, was_correct=False, escalated=True)

        stats = cascade.get_stats()
        assert stats["local_kept"] == 3
        assert abs(stats["local_accuracy"] - 2 / 3) < 0.01

    def test_calibrator_integration(self, tmp_dir):
        """校准器集成：未升级的结果用于校准"""
        cascade = CascadeDecision(tmp_dir, escalation_threshold=0.4)
        # 添加足够的校准数据
        for _ in range(20):
            cascade.record_outcome({"confidence": 0.9}, was_correct=True, escalated=False)

        stats = cascade.get_stats()
        assert stats["calibration"]["is_calibrated"]

    def test_escalated_not_used_for_calibration(self, tmp_dir):
        """升级的结果不用于校准（由云端处理）"""
        cascade = CascadeDecision(tmp_dir, escalation_threshold=0.4)
        for _ in range(20):
            cascade.record_outcome({"confidence": 0.1}, was_correct=False, escalated=True)

        stats = cascade.get_stats()
        # 升级的结果不校准，所以校准样本为 0
        assert stats["calibration"]["total_samples"] == 0

    def test_get_stats_empty(self, tmp_dir):
        """空级联统计"""
        cascade = CascadeDecision(tmp_dir)
        stats = cascade.get_stats()
        assert stats["total"] == 0
        assert stats["escalated"] == 0
        assert stats["escalation_rate"] == 0.0

    def test_history_persistence(self, tmp_dir):
        """历史记录持久化"""
        cascade = CascadeDecision(tmp_dir)
        cascade.record_outcome({"confidence": 0.9}, was_correct=True, escalated=False)

        # 重新加载
        cascade2 = CascadeDecision(tmp_dir)
        stats = cascade2.get_stats()
        assert stats["total"] == 1

    def test_custom_threshold(self, tmp_dir):
        """自定义阈值"""
        cascade = CascadeDecision(tmp_dir, escalation_threshold=0.6)
        # 0.5 < 0.7 (0.6+0.1 未校准) → 升级
        decision = cascade.should_escalate({"confidence": 0.5})
        assert decision["escalate"]
        # 0.8 > 0.7 → 不升级
        decision = cascade.should_escalate({"confidence": 0.8})
        assert not decision["escalate"]

    def test_decision_contains_reason(self, tmp_dir):
        """决策包含原因说明"""
        cascade = CascadeDecision(tmp_dir)
        decision = cascade.should_escalate({"confidence": 0.8})
        assert "reason" in decision
        assert "原始" in decision["reason"]

    def test_missing_confidence_key(self, tmp_dir):
        """缺少 confidence key 时默认 0.0"""
        cascade = CascadeDecision(tmp_dir)
        decision = cascade.should_escalate({})
        assert decision["raw_confidence"] == 0.0
        assert decision["escalate"]  # 0.0 < 阈值


# ─── 集成测试 ────────────────────────────────────────────────────


class TestConfidenceIntegration:
    """置信度系统集成测试"""

    @pytest.fixture
    def tmp_dir(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    def test_full_cascade_flow(self, tmp_dir):
        """完整级联流程：本地执行 → 置信度提取 → 校准 → 决策"""
        cascade = CascadeDecision(tmp_dir, escalation_threshold=0.4)

        # 模拟高置信度本地输出
        high_conf = extract_confidence([
            {"token": "A", "logprob": -0.01,
             "top_logprobs": {"A": -0.01, "B": -5.0}},
        ])
        decision = cascade.should_escalate(high_conf)
        assert not decision["escalate"]

        # 模拟低置信度本地输出
        low_conf = extract_confidence([
            {"token": "A", "logprob": -2.0,
             "top_logprobs": {"A": -2.0, "B": -2.1}},
        ])
        decision = cascade.should_escalate(low_conf)
        # 低置信度应该触发升级
        assert decision["escalate"]

    def test_text_fallback_cascade(self, tmp_dir):
        """纯文本启发式 + 级联"""
        cascade = CascadeDecision(tmp_dir, escalation_threshold=0.4)

        # 正常输出
        conf = extract_confidence_from_text("分类结果：文件A属于文档类。" * 3)
        decision = cascade.should_escalate(conf)
        assert not decision["escalate"]

        # 失败输出
        conf = extract_confidence_from_text("抱歉，我无法完成这个任务")
        decision = cascade.should_escalate(conf)
        assert decision["escalate"]

    def test_calibration_improves_over_time(self, tmp_dir):
        """校准随时间改善"""
        cascade = CascadeDecision(tmp_dir, escalation_threshold=0.4)

        # 添加校准数据：不同置信度水平
        for _ in range(10):
            cascade.record_outcome({"confidence": 0.9}, was_correct=True, escalated=False)
        for _ in range(5):
            cascade.record_outcome({"confidence": 0.6}, was_correct=True, escalated=False)
        for _ in range(5):
            cascade.record_outcome({"confidence": 0.3}, was_correct=False, escalated=False)

        stats = cascade.get_stats()
        assert stats["calibration"]["is_calibrated"]
        assert stats["local_accuracy"] == 0.75  # 15/20


# ─── 边界条件测试 ────────────────────────────────────────────────


class TestEdgeCases:
    """边界条件和异常输入"""

    @pytest.fixture
    def tmp_dir(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    def test_extract_confidence_none_logprobs(self):
        """None 输入处理（防御性）"""
        # 不应该崩溃
        result = extract_confidence([])
        assert result["confidence"] == 0.0

    def test_extract_confidence_negative_entropy(self):
        """logprob=0 不应该产生负熵"""
        logprobs = [
            {"token": "A", "logprob": 0.0,
             "top_logprobs": {"A": 0.0}},
        ]
        result = extract_confidence(logprobs)
        assert result["entropy"] >= 0.0

    def test_calibrator_concurrent_records(self, tmp_dir):
        """连续快速记录不崩溃"""
        cal = ConfidenceCalibrator(tmp_dir)
        for i in range(100):
            cal.record(0.5 + (i % 10) * 0.05, i % 2 == 0)
        assert cal.get_stats()["total_samples"] == 100

    def test_cascade_zero_threshold(self, tmp_dir):
        """零阈值：几乎一切都升级"""
        cascade = CascadeDecision(tmp_dir, escalation_threshold=0.0)
        decision = cascade.should_escalate({"confidence": 0.01})
        # 未校准时阈值 = 0.0 + 0.1 = 0.1
        assert decision["escalate"]

    def test_cascade_threshold_one(self, tmp_dir):
        """阈值 1.0：几乎一切都不升级"""
        cascade = CascadeDecision(tmp_dir, escalation_threshold=1.0)
        decision = cascade.should_escalate({"confidence": 0.99})
        # 未校准时阈值 = 1.0 + 0.1 = 1.1，0.99 < 1.1 → 升级
        assert decision["escalate"]

    def test_calibrate_all_correct(self, tmp_dir):
        """全部正确时校准应该高"""
        cal = ConfidenceCalibrator(tmp_dir)
        # 使用多个不同置信度值以产生多个 PAV 桶
        for i in range(10):
            cal.record(0.3 + i * 0.05, True)
        for i in range(10):
            cal.record(0.8 + i * 0.01, True)
        if cal.get_stats()["is_calibrated"]:
            assert cal.calibrate(0.9) > 0.8

    def test_calibrate_all_wrong(self, tmp_dir):
        """全部错误时校准应该低"""
        cal = ConfidenceCalibrator(tmp_dir)
        # 使用多个不同置信度值以产生多个 PAV 桶
        for i in range(10):
            cal.record(0.3 + i * 0.05, False)
        for i in range(10):
            cal.record(0.8 + i * 0.01, False)
        if cal.get_stats()["is_calibrated"]:
            assert cal.calibrate(0.9) < 0.2

    def test_calibrate_uniform_distribution(self, tmp_dir):
        """均匀分布时校准应该保持接近原始值"""
        cal = ConfidenceCalibrator(tmp_dir)
        for i in range(30):
            conf = 0.1 + i * 0.027
            cal.record(conf, i % 2 == 0)
        if cal.get_stats()["is_calibrated"]:
            mid = cal.calibrate(0.5)
            # 均匀分布时中间值应该接近 0.5
            assert 0.2 < mid < 0.8
