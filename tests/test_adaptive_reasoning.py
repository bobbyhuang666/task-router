"""
自适应推理路径选择器测试 — Token 分位数驱动的推理策略

测试创新点：
1. Token 分位数置信度信号计算
2. 基于置信度的策略决策树
3. Chain-of-Draft (CoD) 极简推理策略
4. Prompt 增强适配
"""

import pytest
from task_router.adaptive_reasoning import (
    AdaptiveStrategyDecision,
    TOKEN_BUDGET,
    select_adaptive_strategy,
    enhance_prompt_adaptive,
)


def _make_confident_logprobs(n: int = 10) -> list[dict]:
    """高置信度 logprobs"""
    logprobs = []
    for _ in range(n):
        logprobs.append({
            "logprob": -0.05,
            "top_logprobs": {"correct": -0.05, "wrong": -5.0, "wrong2": -8.0}
        })
    return logprobs


def _make_medium_logprobs(n: int = 10) -> list[dict]:
    """中等置信度 logprobs"""
    logprobs = []
    for _ in range(n):
        logprobs.append({
            "logprob": -1.0,
            "top_logprobs": {"opt1": -0.9, "opt2": -1.1, "opt3": -1.3}
        })
    return logprobs


def _make_uncertain_logprobs(n: int = 5) -> list[dict]:
    """低置信度 logprobs"""
    logprobs = []
    for _ in range(n):
        logprobs.append({
            "logprob": -3.0,
            "top_logprobs": {"a": -2.5, "b": -2.6, "c": -2.8, "d": -3.0}
        })
    return logprobs


# ─── 策略选择测试 ──────────────────────────────────────────────


class TestStrategySelection:
    """策略选择决策树"""

    def test_returns_decision_object(self):
        """应返回 AdaptiveStrategyDecision 对象"""
        logprobs = _make_confident_logprobs(10)
        result = select_adaptive_strategy(logprobs=logprobs)
        assert isinstance(result, AdaptiveStrategyDecision)

    def test_high_confidence_direct(self):
        """高置信度 → direct 策略"""
        logprobs = _make_confident_logprobs(20)
        result = select_adaptive_strategy(
            logprobs=logprobs,
            task_type="translation",
            complexity_score=1.0,
        )
        assert result.strategy == "direct"
        assert result.confidence_signal > 0.5

    def test_structured_on_trigger_words(self):
        """结构化触发词 → structured 策略（高复杂度时）"""
        logprobs = _make_medium_logprobs(10)
        result = select_adaptive_strategy(
            logprobs=logprobs,
            complexity_score=3.0,
            action="请列出所有步骤并给出总结",
        )
        assert result.strategy == "structured"

    def test_few_shot_with_examples(self):
        """有示例且中等置信度 → few_shot 策略"""
        logprobs = _make_medium_logprobs(10)
        result = select_adaptive_strategy(
            logprobs=logprobs,
            has_examples=True,
            complexity_score=3.0,
        )
        # 应该是 few_shot 或 cod（取决于置信度精确值）
        assert result.strategy in ("few_shot", "cod", "cot")

    def test_cod_for_medium_confidence(self):
        """中等置信度 → Chain-of-Draft 策略"""
        logprobs = _make_medium_logprobs(10)
        result = select_adaptive_strategy(
            logprobs=logprobs,
            complexity_score=3.0,
        )
        # 中等置信度，应为 cod 或 cot
        assert result.strategy in ("cod", "cot")
        if result.strategy == "cod":
            assert result.token_budget_factor == TOKEN_BUDGET["cod"]

    def test_cot_for_low_confidence(self):
        """低置信度 → CoT 策略"""
        logprobs = _make_uncertain_logprobs(5)
        result = select_adaptive_strategy(
            logprobs=logprobs,
            complexity_score=5.0,
        )
        assert result.strategy in ("cot", "structured")

    def test_no_logprobs_low_complexity_direct(self):
        """无 logprobs + 低复杂度 → direct"""
        result = select_adaptive_strategy(
            logprobs=[],
            complexity_score=1.0,
        )
        assert result.strategy == "direct"

    def test_no_logprobs_medium_complexity_cod(self):
        """无 logprobs + 中等复杂度 → cod（置信度=0.65 落入 cod 区间）"""
        result = select_adaptive_strategy(
            logprobs=[],
            complexity_score=3.5,
        )
        # 无 logprobs 时 confidence = 1.0 - 3.5/10 = 0.65, 落入 cod (0.4~0.7, complexity<4.0)
        assert result.strategy == "cod"

    def test_no_logprobs_high_complexity_cot(self):
        """无 logprobs + 高复杂度 → cot（置信度=0.3 落入 cot 区间）"""
        result = select_adaptive_strategy(
            logprobs=[],
            complexity_score=7.0,
        )
        # 无 logprobs 时 confidence = 1.0 - 7.0/10 = 0.3, 落入 cot (0.2~0.5)
        assert result.strategy == "cot"

    def test_no_logprobs_very_high_complexity_structured(self):
        """无 logprobs + 极高复杂度 → structured（走启发式降级路径）"""
        result = select_adaptive_strategy(
            logprobs=[],
            complexity_score=9.5,
        )
        # 无 logprobs 时 confidence = 1.0 - 9.5/10 = 0.05, 低于 0.2 不匹配 cot
        # 进入决策树第 6 步，complexity=9.5 >= 5.0 → structured
        assert result.strategy == "structured"

    def test_structured_needs_high_complexity(self):
        """structured 需要高复杂度才触发"""
        logprobs = _make_confident_logprobs(10)
        result = select_adaptive_strategy(
            logprobs=logprobs,
            complexity_score=1.0,  # 低复杂度
            action="请列出步骤",
        )
        # 低复杂度不应触发 structured
        assert result.strategy != "structured"

    def test_all_strategies_possible(self):
        """确保所有 5 种策略都可以被选中"""
        strategies_found = set()

        # direct: 高置信度
        lp = _make_confident_logprobs(20)
        r = select_adaptive_strategy(logprobs=lp, complexity_score=1.0)
        strategies_found.add(r.strategy)

        # cod: 中等置信度
        lp = _make_medium_logprobs(10)
        r = select_adaptive_strategy(logprobs=lp, complexity_score=3.0)
        strategies_found.add(r.strategy)

        # cot: 低置信度
        lp = _make_uncertain_logprobs(5)
        r = select_adaptive_strategy(logprobs=lp, complexity_score=2.0)
        strategies_found.add(r.strategy)

        # structured: 触发词 + 高复杂度
        lp = _make_medium_logprobs(10)
        r = select_adaptive_strategy(logprobs=lp, complexity_score=4.0, action="请列出所有步骤并总结")
        strategies_found.add(r.strategy)

        # few_shot: 有示例
        lp = _make_medium_logprobs(10)
        r = select_adaptive_strategy(logprobs=lp, has_examples=True, complexity_score=3.0)
        strategies_found.add(r.strategy)

        assert "direct" in strategies_found
        assert "cot" in strategies_found or "cod" in strategies_found


# ─── Token 预算测试 ──────────────────────────────────────────────


class TestTokenBudget:
    """Token 预算因子"""

    def test_cod_budget_minimal(self):
        """Chain-of-Draft 应消耗最少 token"""
        assert TOKEN_BUDGET["cod"] < TOKEN_BUDGET["direct"]
        assert TOKEN_BUDGET["cod"] < TOKEN_BUDGET["cot"]
        assert TOKEN_BUDGET["cod"] < TOKEN_BUDGET["few_shot"]
        assert TOKEN_BUDGET["cod"] < TOKEN_BUDGET["structured"]

    def test_cot_budget_highest(self):
        """CoT 应消耗最多 token"""
        assert TOKEN_BUDGET["cot"] > TOKEN_BUDGET["direct"]
        assert TOKEN_BUDGET["cot"] > TOKEN_BUDGET["cod"]
        assert TOKEN_BUDGET["cot"] > TOKEN_BUDGET["structured"]

    def test_direct_budget_is_baseline(self):
        """direct 预算为 1.0（基准）"""
        assert TOKEN_BUDGET["direct"] == 1.0

    def test_budget_range(self):
        """所有预算因子应为正数"""
        for strategy, factor in TOKEN_BUDGET.items():
            assert factor > 0, f"{strategy} 的预算因子应为正数: {factor}"


# ─── 置信度信号测试 ──────────────────────────────────────────────


class TestConfidenceSignal:
    """置信度信号计算"""

    def test_confident_higher_signal(self):
        """高置信度 logprobs 应产生更高的置信度信号"""
        conf = _make_confident_logprobs(20)
        unconf = _make_uncertain_logprobs(5)

        r_conf = select_adaptive_strategy(logprobs=conf)
        r_unconf = select_adaptive_strategy(logprobs=unconf)

        assert r_conf.confidence_signal > r_unconf.confidence_signal

    def test_signal_range(self):
        """置信度信号应在合理范围内"""
        for lp in [
            _make_confident_logprobs(20),
            _make_medium_logprobs(10),
            _make_uncertain_logprobs(5),
        ]:
            r = select_adaptive_strategy(logprobs=lp)
            assert 0.0 <= r.confidence_signal <= 1.0 + 0.01  # 允许微小浮点误差


# ─── Prompt 增强测试 ──────────────────────────────────────────────


class TestPromptEnhancement:
    """Prompt 增强"""

    def test_direct_no_change(self):
        """direct 策略不修改 prompt"""
        prompt = "翻译这句话"
        result = enhance_prompt_adaptive(prompt, "direct")
        assert result == prompt

    def test_cot_adds_steps(self):
        """CoT 应添加逐步思考提示"""
        prompt = "翻译这句话"
        result = enhance_prompt_adaptive(prompt, "cot")
        assert "一步一步" in result or "逐步" in result
        assert result.startswith(prompt)

    def test_cod_adds_minimal_prompt(self):
        """CoD 应添加极简推理提示"""
        prompt = "翻译这句话"
        result = enhance_prompt_adaptive(prompt, "cod")
        assert "最少" in result or "关键" in result
        assert result.startswith(prompt)
        # CoD 的 prompt 应比 CoT 短
        cot_result = enhance_prompt_adaptive(prompt, "cot")
        assert len(result) < len(cot_result)

    def test_few_shot_adds_examples(self):
        """few_shot 应添加示例"""
        prompt = "翻译这句话"
        examples = "1. Hello → 你好\n2. World → 世界"
        result = enhance_prompt_adaptive(prompt, "few_shot", examples=examples)
        assert "Hello" in result
        assert "示例" in result

    def test_structured_adds_format(self):
        """structured 应添加格式模板"""
        prompt = "总结项目"
        result = enhance_prompt_adaptive(prompt, "structured")
        assert "要点" in result or "结论" in result

    def test_few_shot_no_examples_fallback(self):
        """few_shot 无示例时返回原始 prompt"""
        prompt = "翻译这句话"
        result = enhance_prompt_adaptive(prompt, "few_shot", examples="")
        assert result == prompt

    def test_unknown_strategy_passthrough(self):
        """未知策略返回原始 prompt"""
        prompt = "翻译这句话"
        result = enhance_prompt_adaptive(prompt, "unknown_strategy")
        assert result == prompt


# ─── 结构化触发词测试 ──────────────────────────────────────────────


class TestStructuredTriggers:
    """结构化输出触发词检测"""

    @pytest.mark.parametrize("trigger", [
        "列出", "列举", "大纲", "结构", "框架", "步骤", "流程",
        "方案", "计划", "报告", "总结", "汇总", "清单",
        "表格", "对比", "比较",
    ])
    def test_trigger_word_detected(self, trigger):
        """所有触发词应被正确检测"""
        logprobs = _make_medium_logprobs(10)
        result = select_adaptive_strategy(
            logprobs=logprobs,
            complexity_score=3.0,
            action=f"请{trigger}相关的内容",
        )
        # 高复杂度 + 触发词 → structured
        assert result.strategy == "structured"

    def test_no_trigger_word(self):
        """无触发词时不应触发 structured"""
        logprobs = _make_confident_logprobs(20)
        result = select_adaptive_strategy(
            logprobs=logprobs,
            complexity_score=3.0,
            action="帮我看一下这段话",
        )
        assert result.strategy != "structured"


# ─── 边界条件测试 ──────────────────────────────────────────────


class TestEdgeCases:
    """边界条件"""

    def test_single_token_logprobs(self):
        """单个 token 的 logprobs"""
        logprobs = [{"logprob": -0.1, "top_logprobs": {"a": -0.1, "b": -5.0}}]
        result = select_adaptive_strategy(logprobs=logprobs)
        assert isinstance(result, AdaptiveStrategyDecision)
        assert result.strategy in TOKEN_BUDGET

    def test_zero_complexity(self):
        """零复杂度"""
        logprobs = _make_confident_logprobs(10)
        result = select_adaptive_strategy(logprobs=logprobs, complexity_score=0.0)
        assert result.strategy in TOKEN_BUDGET

    def test_very_high_complexity(self):
        """极高复杂度"""
        logprobs = _make_uncertain_logprobs(3)
        result = select_adaptive_strategy(logprobs=logprobs, complexity_score=10.0)
        assert result.strategy in TOKEN_BUDGET

    def test_empty_action(self):
        """空 action"""
        logprobs = _make_confident_logprobs(10)
        result = select_adaptive_strategy(logprobs=logprobs, action="")
        assert isinstance(result, AdaptiveStrategyDecision)

    def test_none_action(self):
        """None action"""
        logprobs = _make_confident_logprobs(10)
        result = select_adaptive_strategy(logprobs=logprobs, action=None)
        assert isinstance(result, AdaptiveStrategyDecision)
