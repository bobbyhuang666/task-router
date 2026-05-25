"""
推理路径路由测试

测试覆盖:
- select_reasoning_strategy: 策略选择逻辑
- enhance_prompt_with_strategy: prompt 增强
- StrategyTracker: 策略性能追踪
"""

import os
import json
import pytest
import tempfile

from task_router.reasoning import (
    select_reasoning_strategy,
    enhance_prompt_with_strategy,
    StrategyTracker,
    STRATEGY_DIRECT,
    STRATEGY_COT,
    STRATEGY_FEW_SHOT,
    STRATEGY_STRUCTURED,
    STRATEGY_TOKEN_MULTIPLIER,
    COT_TRIGGERS,
    STRUCTURED_TRIGGERS,
)


# ─── 策略选择测试 ──────────────────────────────────────────────


class TestSelectReasoningStrategy:
    """推理策略选择"""

    def test_simple_task_direct(self):
        """简单任务 → direct"""
        decision = select_reasoning_strategy("分类文件", "a.pdf, b.jpg", 1.0)
        assert decision.strategy == STRATEGY_DIRECT

    def test_medium_task_cot(self):
        """中等复杂度 → cot"""
        decision = select_reasoning_strategy("分析数据趋势", "数据如下", 3.0)
        assert decision.strategy == STRATEGY_COT

    def test_cot_trigger_keyword(self):
        """包含 CoT 触发词 → cot"""
        decision = select_reasoning_strategy("分析原因", "", 2.5)
        assert decision.strategy == STRATEGY_COT

    def test_structured_trigger_keyword(self):
        """包含结构化触发词 + 高复杂度 → structured"""
        decision = select_reasoning_strategy("列出实施步骤", "", 4.0)
        assert decision.strategy == STRATEGY_STRUCTURED

    def test_high_complexity_structured(self):
        """高复杂度无触发词 → structured"""
        decision = select_reasoning_strategy("完成这个任务", "", 6.0)
        assert decision.strategy == STRATEGY_STRUCTURED

    def test_low_complexity_ignores_triggers(self):
        """低复杂度忽略触发词"""
        decision = select_reasoning_strategy("分析", "", 0.5)
        assert decision.strategy == STRATEGY_DIRECT

    def test_boundary_2_0(self):
        """边界值 2.0：刚好进入中等复杂度"""
        decision = select_reasoning_strategy("普通任务", "", 2.0)
        assert decision.strategy in [STRATEGY_COT, STRATEGY_STRUCTURED]

    def test_reason_contains_score(self):
        """原因包含评分信息"""
        decision = select_reasoning_strategy("测试", "", 1.5)
        assert "score" in decision.reason or "简单" in decision.reason

    def test_complexity_score_preserved(self):
        """复杂度评分被保留"""
        decision = select_reasoning_strategy("测试", "", 3.5)
        assert decision.complexity_score == 3.5

    def test_multiple_cot_triggers(self):
        """多个 CoT 触发词"""
        decision = select_reasoning_strategy("分析原因并比较优缺点", "", 3.0)
        assert decision.strategy == STRATEGY_COT

    def test_multiple_structured_triggers(self):
        """多个结构化触发词"""
        decision = select_reasoning_strategy("列出步骤和流程", "", 4.0)
        assert decision.strategy == STRATEGY_STRUCTURED


# ─── Prompt 增强测试 ────────────────────────────────────────────


class TestEnhancePromptWithStrategy:
    """prompt 增强"""

    def test_direct_no_change(self):
        """direct 策略不改变 prompt"""
        prompt = "原始 prompt"
        result = enhance_prompt_with_strategy(prompt, STRATEGY_DIRECT)
        assert result == prompt

    def test_cot_adds_thinking(self):
        """cot 添加思考步骤"""
        prompt = "分析问题"
        result = enhance_prompt_with_strategy(prompt, STRATEGY_COT)
        assert "一步一步" in result
        assert prompt in result

    def test_structured_adds_format(self):
        """structured 添加输出格式"""
        prompt = "列出方案"
        result = enhance_prompt_with_strategy(prompt, STRATEGY_STRUCTURED)
        assert "要点" in result
        assert prompt in result

    def test_few_shot_with_examples(self):
        """few_shot 注入示例"""
        prompt = "分类文件"
        examples = "示例1: a.pdf → 文档\n示例2: b.jpg → 图片"
        result = enhance_prompt_with_strategy(prompt, STRATEGY_FEW_SHOT, examples)
        assert "参考示例" in result
        assert examples in result

    def test_few_shot_empty_examples(self):
        """few_shot 无示例时不改变 prompt"""
        prompt = "分类文件"
        result = enhance_prompt_with_strategy(prompt, STRATEGY_FEW_SHOT, "")
        assert result == prompt

    def test_unknown_strategy_no_change(self):
        """未知策略不改变 prompt"""
        prompt = "测试"
        result = enhance_prompt_with_strategy(prompt, "unknown_strategy")
        assert result == prompt

    def test_preserves_original_content(self):
        """增强保留原始 prompt 内容"""
        prompt = "将以下文本翻译成中文：Hello world"
        result = enhance_prompt_with_strategy(prompt, STRATEGY_COT)
        assert "翻译成中文" in result
        assert "Hello world" in result


# ─── 策略性能追踪测试 ──────────────────────────────────────────


class TestStrategyTracker:
    """策略性能追踪"""

    @pytest.fixture
    def tmp_dir(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    def test_record_and_stats(self, tmp_dir):
        """记录和统计"""
        tracker = StrategyTracker(tmp_dir)
        tracker.record(STRATEGY_DIRECT, "classification", True, 50, 100)
        tracker.record(STRATEGY_COT, "classification", True, 120, 200)

        stats = tracker.get_stats()
        assert stats["total"] == 2
        assert stats["by_strategy"][STRATEGY_DIRECT]["total"] == 1
        assert stats["by_strategy"][STRATEGY_COT]["total"] == 1

    def test_best_strategy_insufficient_data(self, tmp_dir):
        """数据不足时返回 None"""
        tracker = StrategyTracker(tmp_dir)
        tracker.record(STRATEGY_DIRECT, "classification", True, 50, 100)
        assert tracker.get_best_strategy("classification") is None

    def test_best_strategy_with_data(self, tmp_dir):
        """足够数据时返回最佳策略"""
        tracker = StrategyTracker(tmp_dir)
        # direct: 4/5 成功
        for _ in range(4):
            tracker.record(STRATEGY_DIRECT, "classification", True, 50, 100)
        tracker.record(STRATEGY_DIRECT, "classification", False, 50, 100)
        # cot: 5/5 成功但 token 更多
        for _ in range(5):
            tracker.record(STRATEGY_COT, "classification", True, 200, 300)

        best = tracker.get_best_strategy("classification")
        assert best in [STRATEGY_DIRECT, STRATEGY_COT]

    def test_best_strategy_per_task_type(self, tmp_dir):
        """不同任务类型独立追踪"""
        tracker = StrategyTracker(tmp_dir)
        for _ in range(5):
            tracker.record(STRATEGY_DIRECT, "classification", True, 50, 100)
        for _ in range(5):
            tracker.record(STRATEGY_COT, "translation", True, 120, 200)

        assert tracker.get_best_strategy("classification") == STRATEGY_DIRECT
        assert tracker.get_best_strategy("translation") == STRATEGY_COT

    def test_persistence(self, tmp_dir):
        """数据持久化"""
        tracker = StrategyTracker(tmp_dir)
        tracker.record(STRATEGY_DIRECT, "test", True, 50, 100)

        tracker2 = StrategyTracker(tmp_dir)
        assert tracker2.get_stats()["total"] == 1

    def test_empty_stats(self, tmp_dir):
        """空统计"""
        tracker = StrategyTracker(tmp_dir)
        stats = tracker.get_stats()
        assert stats["total"] == 0
        assert stats["by_strategy"] == {}

    def test_malformed_jsonl_skipped(self, tmp_dir):
        """损坏的 JSONL 被跳过"""
        data_file = os.path.join(tmp_dir, "strategy_history.jsonl")
        with open(data_file, "w") as f:
            f.write("not json\n")
            f.write(json.dumps({"strategy": "direct", "task_type": "test", "success": True, "tokens": 50, "latency_ms": 100}) + "\n")

        tracker = StrategyTracker(tmp_dir)
        assert tracker.get_stats()["total"] == 1


# ─── Token 乘数测试 ─────────────────────────────────────────────


class TestTokenMultiplier:
    """策略 token 乘数"""

    def test_direct_is_baseline(self):
        """direct 是基线（1.0x）"""
        assert STRATEGY_TOKEN_MULTIPLIER[STRATEGY_DIRECT] == 1.0

    def test_cot_uses_more_tokens(self):
        """cot 消耗更多 token"""
        assert STRATEGY_TOKEN_MULTIPLIER[STRATEGY_COT] > 1.0

    def test_all_strategies_have_multiplier(self):
        """所有策略都有乘数"""
        for s in [STRATEGY_DIRECT, STRATEGY_COT, STRATEGY_FEW_SHOT, STRATEGY_STRUCTURED]:
            assert s in STRATEGY_TOKEN_MULTIPLIER
            assert STRATEGY_TOKEN_MULTIPLIER[s] >= 1.0

    def test_cot_is_most_expensive(self):
        """cot 是最贵的策略"""
        assert STRATEGY_TOKEN_MULTIPLIER[STRATEGY_COT] == max(STRATEGY_TOKEN_MULTIPLIER.values())


# ─── 触发词测试 ─────────────────────────────────────────────────


class TestTriggers:
    """触发词覆盖"""

    def test_cot_triggers_not_empty(self):
        assert len(COT_TRIGGERS) > 0

    def test_structured_triggers_not_empty(self):
        assert len(STRUCTURED_TRIGGERS) > 0

    def test_no_overlap(self):
        """两类触发词不重叠"""
        overlap = set(COT_TRIGGERS) & set(STRUCTURED_TRIGGERS)
        assert len(overlap) == 0, f"重叠触发词: {overlap}"
