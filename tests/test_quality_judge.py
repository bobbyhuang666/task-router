"""
LLM-as-Judge 多维度质量评估器测试

测试覆盖:
- QualityScores 数据结构
- JSON 解析容错
- 单条评估（mock LLM）
- 批量评估
- Fallback 评估（无 LLM）
- 去重机制
- 综合评分计算
"""

import json
import os
import pytest
import tempfile

from task_router.quality_judge import (
    QualityJudge,
    QualityScores,
    _parse_single_judgment,
    _parse_batch_judgment,
    _fallback_judge,
    _clamp_score,
    _scores_from_judgment,
    reset_quality_judge,
    get_quality_judge,
    QUALITY_DIMENSIONS,
)


# ─── 辅助函数 ──────────────────────────────────────────────


def _sample_episode(**overrides):
    """创建测试用 episode dict"""
    ep = {
        "episode_id": "test_ep_001",
        "action": "翻译成中文",
        "text": "Hello world, this is a test.",
        "task_type": "translate_en2zh",
        "capability": "translation",
        "complexity_score": 1.5,
        "confidence_data": {"confidence": 0.85, "entropy": 0.2, "margin": 0.6},
        "strategy": "direct",
        "strategy_reason": "简单任务",
        "route": "local",
        "model_used": "qwen-tool",
        "routing_signals": {},
        "output": "你好世界，这是一个测试。",
        "tokens_input": 45,
        "tokens_output": 12,
        "time_ms": 1200,
        "cost_saved": 0.00042,
    }
    ep.update(overrides)
    return ep


def _sample_llm_response():
    """标准 LLM 评分响应"""
    return json.dumps({
        "relevance": 9,
        "completeness": 8,
        "accuracy": 9,
        "efficiency": 8,
        "correctness": 9,
        "optimal_route": "local",
        "optimal_strategy": "direct",
        "routing_error": "none",
        "notes": "翻译准确，路由正确",
    })


def _mock_llm_factory(responses=None):
    """创建 mock LLM 调用者（可选预设响应列表）"""
    call_count = [0]

    def mock_caller(prompt):
        if responses and call_count[0] < len(responses):
            resp = responses[call_count[0]]
            call_count[0] += 1
            return resp
        return _sample_llm_response()

    return mock_caller


# ─── QualityScores 测试 ──────────────────────────────────────


class TestQualityScores:
    """QualityScores 数据结构测试"""

    def test_overall_score(self):
        """综合评分计算"""
        qs = QualityScores(
            relevance=8.0,
            completeness=7.0,
            accuracy=9.0,
            efficiency=6.0,
            correctness=8.0,
        )
        # 加权: 0.20*8 + 0.20*7 + 0.25*9 + 0.15*6 + 0.20*8 = 1.6+1.4+2.25+0.9+1.6 = 7.75
        expected = 7.75 / 10.0
        assert abs(qs.overall - expected) < 0.01

    def test_route_correct_high(self):
        """高正确性且无错误 → route_correct = True"""
        qs = QualityScores(correctness=8.0, routing_error="none")
        assert qs.route_correct is True

    def test_route_correct_low(self):
        """低正确性 → route_correct = False"""
        qs = QualityScores(correctness=4.0, routing_error="none")
        assert qs.route_correct is False

    def test_route_correct_with_error(self):
        """有路由错误 → route_correct = False"""
        qs = QualityScores(correctness=8.0, routing_error="over_escalated")
        assert qs.route_correct is False

    def test_to_dict(self):
        """to_dict 包含所有字段"""
        qs = QualityScores(episode_id="test", relevance=7.5)
        d = qs.to_dict()
        assert d["episode_id"] == "test"
        assert d["relevance"] == 7.5
        assert "timestamp" in d
        assert "judge_method" in d


# ─── 解析函数测试 ──────────────────────────────────────────────


class TestParsing:
    """JSON 解析容错测试"""

    def test_parse_clean_json(self):
        """标准 JSON 解析"""
        response = _sample_llm_response()
        data = _parse_single_judgment(response, "ep001")
        assert data["relevance"] == 9
        assert data["episode_id"] == "ep001"

    def test_parse_markdown_code_block(self):
        """markdown 代码块包裹的 JSON"""
        response = '```json\n{"relevance": 8, "completeness": 7}\n```'
        data = _parse_single_judgment(response)
        assert data["relevance"] == 8
        assert data["completeness"] == 7

    def test_parse_json_with_surrounding_text(self):
        """JSON 前后有杂文"""
        response = '以下是评分结果：\n{"relevance": 7, "notes": "不错"}\n希望对你有帮助。'
        data = _parse_single_judgment(response)
        assert data["relevance"] == 7

    def test_parse_invalid_json(self):
        """完全无法解析时返回空 dict"""
        data = _parse_single_judgment("这不是 JSON")
        assert data == {}

    def test_parse_empty_response(self):
        """空响应"""
        data = _parse_single_judgment("")
        assert data == {}

    def test_parse_batch_array(self):
        """批量 JSON 数组解析"""
        response = json.dumps([
            {"episode_id": "ep1", "relevance": 8},
            {"episode_id": "ep2", "relevance": 7},
        ])
        data = _parse_batch_judgment(response)
        assert len(data) == 2
        assert data[0]["episode_id"] == "ep1"

    def test_parse_batch_markdown(self):
        """批量解析 markdown 代码块"""
        response = '```json\n[{"episode_id": "ep1", "relevance": 9}]\n```'
        data = _parse_batch_judgment(response)
        assert len(data) == 1

    def test_parse_batch_invalid(self):
        """批量解析失败返回空列表"""
        data = _parse_batch_judgment("not valid")
        assert data == []

    def test_clamp_score(self):
        """评分范围限制"""
        assert _clamp_score(5) == 5.0
        assert _clamp_score(15) == 10.0
        assert _clamp_score(-3) == 0.0
        assert _clamp_score("abc") == 0.0
        assert _clamp_score(None) == 0.0

    def test_scores_from_judgment(self):
        """从解析结果构造 QualityScores"""
        data = {
            "episode_id": "ep001",
            "relevance": 8,
            "completeness": 7,
            "accuracy": 9,
            "efficiency": 6,
            "correctness": 8,
            "optimal_route": "local",
            "optimal_strategy": "direct",
            "routing_error": "none",
            "notes": "好",
        }
        qs = _scores_from_judgment(data, "ep001")
        assert qs.episode_id == "ep001"
        assert qs.relevance == 8.0
        assert qs.routing_error == "none"


# ─── Fallback 评估测试 ──────────────────────────────────────


class TestFallbackJudge:
    """无 LLM 时的降级评估测试"""

    def test_good_local_output(self):
        """高质量本地输出 → 高分"""
        ep = _sample_episode(
            output="你好世界，这是一个测试。",
            route="local",
            confidence_data={"confidence": 0.9},
        )
        qs = _fallback_judge(ep)
        assert qs.judge_method == "fallback"
        assert qs.relevance >= 5.0
        assert qs.correctness >= 5.0
        assert qs.efficiency >= 7.0  # 本地高效

    def test_failure_output(self):
        """含失败信号的输出 → 低分"""
        ep = _sample_episode(
            output="抱歉，我无法完成此任务。",
            route="local",
            confidence_data={"confidence": 0.2},
        )
        qs = _fallback_judge(ep)
        assert qs.relevance <= 3.0
        assert qs.routing_error == "under_escalated"

    def test_cache_route_high_efficiency(self):
        """缓存路由 → 最高效率分"""
        ep = _sample_episode(route="cache(exact)")
        qs = _fallback_judge(ep)
        assert qs.efficiency == 10.0
        assert qs.correctness == 9.0

    def test_cloud_route_with_high_confidence(self):
        """高置信度但走了云端 → over_escalated"""
        ep = _sample_episode(
            route="cloud",
            confidence_data={"confidence": 0.9},
            output="翻译结果很好",
        )
        qs = _fallback_judge(ep)
        assert qs.routing_error == "over_escalated"


# ─── QualityJudge 单条评估测试 ──────────────────────────────────────


class TestQualityJudgeSingle:
    """QualityJudge 单条评估测试"""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.judge = QualityJudge(cache_dir=self.tmpdir, use_llm=True)

    def test_judge_with_mock_llm(self):
        """用 mock LLM 评估单条 episode"""
        self.judge.set_llm_caller(_mock_llm_factory())
        ep = _sample_episode()
        scores = self.judge.judge_episode(ep)

        assert scores.episode_id == "test_ep_001"
        assert scores.relevance == 9.0
        assert scores.completeness == 8.0
        assert scores.accuracy == 9.0
        assert scores.efficiency == 8.0
        assert scores.correctness == 9.0
        assert scores.judge_method == "llm"
        assert scores.routing_error == "none"

    def test_judge_saves_to_disk(self):
        """评估结果保存到磁盘"""
        self.judge.set_llm_caller(_mock_llm_factory())
        ep = _sample_episode()
        self.judge.judge_episode(ep)

        judgments = self.judge.get_all_judgments()
        assert len(judgments) == 1
        assert judgments[0]["episode_id"] == "test_ep_001"

    def test_dedup_skip_already_judged(self):
        """已评估的 episode 不重复评估"""
        call_count = [0]

        def counting_caller(prompt):
            call_count[0] += 1
            return _sample_llm_response()

        self.judge.set_llm_caller(counting_caller)
        ep = _sample_episode()

        self.judge.judge_episode(ep)
        self.judge.judge_episode(ep)  # 第二次应该跳过

        assert call_count[0] == 1  # 只调用了一次 LLM
        assert self.judge.count() == 1

    def test_judge_without_llm(self):
        """不使用 LLM 时用 fallback"""
        judge = QualityJudge(cache_dir=self.tmpdir, use_llm=False)
        ep = _sample_episode()
        scores = judge.judge_episode(ep)

        assert scores.judge_method == "fallback"
        assert 0 <= scores.relevance <= 10
        assert 0 <= scores.correctness <= 10

    def test_llm_failure_fallback(self):
        """LLM 调用失败时降级到 fallback"""
        def failing_caller(prompt):
            raise RuntimeError("API 限流")

        self.judge.set_llm_caller(failing_caller)
        ep = _sample_episode()
        scores = self.judge.judge_episode(ep)

        assert scores.judge_method == "fallback"
        assert scores.episode_id == "test_ep_001"


# ─── QualityJudge 批量评估测试 ──────────────────────────────────────


class TestQualityJudgeBatch:
    """QualityJudge 批量评估测试"""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.judge = QualityJudge(cache_dir=self.tmpdir, use_llm=True)

    def test_batch_with_mock_llm(self):
        """批量评估（每批合并到一个 prompt）"""
        batch_response = json.dumps([
            {"episode_id": "ep_001", "relevance": 8, "completeness": 7, "accuracy": 9,
             "efficiency": 6, "correctness": 8, "routing_error": "none", "notes": ""},
            {"episode_id": "ep_002", "relevance": 5, "completeness": 4, "accuracy": 3,
             "efficiency": 7, "correctness": 4, "routing_error": "under_escalated", "notes": ""},
        ])

        self.judge.set_llm_caller(_mock_llm_factory([batch_response]))

        episodes = [
            _sample_episode(episode_id="ep_001"),
            _sample_episode(episode_id="ep_002", output="抱歉，无法翻译。"),
        ]

        results = self.judge.judge_batch(episodes, batch_size=5)
        assert len(results) == 2
        assert results[0].relevance == 8.0
        assert results[1].routing_error == "under_escalated"

    def test_batch_skip_already_judged(self):
        """批量评估跳过已评估的 episode"""
        self.judge.set_llm_caller(_mock_llm_factory())
        ep1 = _sample_episode(episode_id="ep_001")

        # 先评估 ep1
        self.judge.judge_episode(ep1)

        # 批量评估 ep1 + ep2
        batch_response = json.dumps([
            {"episode_id": "ep_002", "relevance": 7, "completeness": 6, "accuracy": 8,
             "efficiency": 5, "correctness": 7, "routing_error": "none", "notes": ""},
        ])
        self.judge.set_llm_caller(_mock_llm_factory([batch_response]))
        ep2 = _sample_episode(episode_id="ep_002")

        results = self.judge.judge_batch([ep1, ep2], batch_size=5)
        assert len(results) == 2

    def test_batch_fallback_on_llm_failure(self):
        """批量 LLM 失败时逐条降级"""
        def failing_caller(prompt):
            raise RuntimeError("API 失败")

        self.judge.set_llm_caller(failing_caller)
        episodes = [
            _sample_episode(episode_id="ep_001"),
            _sample_episode(episode_id="ep_002"),
        ]

        results = self.judge.judge_batch(episodes, batch_size=5)
        assert len(results) == 2
        assert all(r.judge_method == "fallback" for r in results)


# ─── 全局实例管理测试 ──────────────────────────────────────────────


class TestGlobalJudge:
    """全局 QualityJudge 实例管理"""

    def setup_method(self):
        reset_quality_judge()

    def teardown_method(self):
        reset_quality_judge()

    def test_singleton(self):
        """get_quality_judge 返回单例"""
        tmpdir = tempfile.mkdtemp()
        j1 = get_quality_judge(cache_dir=tmpdir, use_llm=False)
        j2 = get_quality_judge()
        assert j1 is j2

    def test_reset(self):
        """reset_quality_judge 重置单例"""
        tmpdir = tempfile.mkdtemp()
        j1 = get_quality_judge(cache_dir=tmpdir, use_llm=False)
        reset_quality_judge()
        j2 = get_quality_judge(cache_dir=tmpdir, use_llm=False)
        assert j1 is not j2


# ─── 边界条件测试 ──────────────────────────────────────────────


class TestEdgeCases:
    """边界条件和异常场景"""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.judge = QualityJudge(cache_dir=self.tmpdir, use_llm=False)

    def test_empty_episode(self):
        """空 episode 不崩溃"""
        ep = {"episode_id": "empty"}
        scores = self.judge.judge_episode(ep)
        assert 0 <= scores.relevance <= 10

    def test_missing_output(self):
        """输出为空"""
        ep = _sample_episode(output="")
        scores = self.judge.judge_episode(ep)
        assert scores.completeness < 8.0

    def test_no_episode_id(self):
        """episode 没有 ID"""
        ep = _sample_episode()
        del ep["episode_id"]
        scores = self.judge.judge_episode(ep)
        assert scores is not None

    def test_very_long_output(self):
        """超长输出不崩溃"""
        ep = _sample_episode(output="很长的输出。" * 1000)
        scores = self.judge.judge_episode(ep)
        assert scores is not None

    def test_judge_count(self):
        """count 正确统计"""
        assert self.judge.count() == 0
        ep1 = _sample_episode(episode_id="ep1")
        ep2 = _sample_episode(episode_id="ep2")
        self.judge.judge_episode(ep1)
        self.judge.judge_episode(ep2)
        assert self.judge.count() == 2

    def test_clear_judgments(self):
        """clear 清空所有评分"""
        ep = _sample_episode()
        self.judge.judge_episode(ep)
        assert self.judge.count() == 1
        self.judge.clear()
        assert self.judge.count() == 0
