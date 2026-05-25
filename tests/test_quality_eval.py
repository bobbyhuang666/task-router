"""
QualityEvaluator 测试

测试覆盖:
- score_output: 评分逻辑（精确/包含/反向/部分/失败）
- generate_report: 报告生成
- detect_regression: 回归检测
- save_results / load_history: 持久化
- EvalCase / EvalResult: 数据结构
"""

import tempfile

import pytest

from task_router.quality_eval import (
    EvalCase,
    EvalResult,
    QualityEvaluator,
    DEFAULT_EVAL_SET,
)


# ─── fixtures ──────────────────────────────────────────────────


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def evaluator(tmp_dir):
    return QualityEvaluator(cache_dir=tmp_dir)


# ─── EvalCase / EvalResult ─────────────────────────────────────


class TestDataStructures:
    def test_eval_case_defaults(self):
        case = EvalCase(id="test", action="翻译", text="hello", expected="你好")
        assert case.task_type == ""
        assert case.difficulty == "easy"
        assert case.tags == []

    def test_eval_result_fields(self):
        r = EvalResult(
            case_id="test", model="qwen", output="你好",
            score=0.9, match_type="contains", latency_ms=100,
        )
        assert r.score == 0.9
        assert r.tokens_input == 0  # default

    def test_default_eval_set_not_empty(self):
        assert len(DEFAULT_EVAL_SET) >= 15
        # 每个用例都有必要字段
        for case in DEFAULT_EVAL_SET:
            assert case.id
            assert case.action
            assert case.expected


# ─── score_output ──────────────────────────────────────────────


class TestScoreOutput:
    def test_exact_match(self, evaluator):
        score, match_type = evaluator.score_output("你好", "你好")
        assert score == 1.0
        assert match_type == "exact"

    def test_exact_match_case_insensitive(self, evaluator):
        score, match_type = evaluator.score_output("Hello", "hello")
        assert score == 1.0
        assert match_type == "exact"

    def test_contains_match(self, evaluator):
        score, match_type = evaluator.score_output("你好世界", "你好")
        assert score == 0.9
        assert match_type == "contains"

    def test_contains_match_longer_output(self, evaluator):
        score, match_type = evaluator.score_output(
            "翻译结果：机器学习是人工智能的一个分支", "机器学习"
        )
        assert score == 0.9
        assert match_type == "contains"

    def test_reverse_contains(self, evaluator):
        score, match_type = evaluator.score_output("你好", "你好世界欢迎你")
        assert score == 0.7
        assert match_type == "reverse_contains"

    def test_partial_match(self, evaluator):
        # "机器学习" 中大部分字符在输出中
        score, match_type = evaluator.score_output(
            "关于机器和学习的关系", "机器学习"
        )
        assert 0 < score < 0.9
        assert match_type == "partial"

    def test_fail_match(self, evaluator):
        score, match_type = evaluator.score_output("完全不同的内容", "你好")
        assert score == 0.0
        assert match_type == "fail"

    def test_empty_output(self, evaluator):
        # 空字符串是任何字符串的子串 → reverse_contains
        score, match_type = evaluator.score_output("", "你好")
        assert score == 0.7
        assert match_type == "reverse_contains"

    def test_empty_expected(self, evaluator):
        # 空字符串是任何字符串的子串 → contains
        score, match_type = evaluator.score_output("你好", "")
        assert score == 0.9
        assert match_type == "contains"

    def test_whitespace_handling(self, evaluator):
        score, match_type = evaluator.score_output("  你好  ", "你好")
        assert score == 1.0
        assert match_type == "exact"

    def test_short_expected_no_partial(self, evaluator):
        # 期望只有 1 个字符，不应触发 partial 匹配
        score, match_type = evaluator.score_output("xyz", "a")
        assert score == 0.0
        assert match_type == "fail"


# ─── generate_report ───────────────────────────────────────────


class TestGenerateReport:
    def test_empty_results(self, evaluator):
        report = evaluator.generate_report([])
        assert report == "无评估结果"

    def test_all_pass(self, evaluator):
        results = [
            EvalResult("t1", "qwen", "你好", 1.0, "exact", 50),
            EvalResult("t2", "qwen", "早上好世界", 0.9, "contains", 60),
        ]
        report = evaluator.generate_report(results)
        assert "2/2" in report
        assert "qwen" in report
        assert "通过率" in report

    def test_some_fail(self, evaluator):
        results = [
            EvalResult("t1", "qwen", "你好", 1.0, "exact", 50),
            EvalResult("t2", "qwen", "错误输出", 0.0, "fail", 60),
        ]
        report = evaluator.generate_report(results)
        assert "1/2" in report
        assert "失败用例" in report
        assert "t2" in report

    def test_match_type_distribution(self, evaluator):
        results = [
            EvalResult("t1", "m", "a", 1.0, "exact", 10),
            EvalResult("t2", "m", "b", 0.9, "contains", 10),
            EvalResult("t3", "m", "c", 0.9, "contains", 10),
        ]
        report = evaluator.generate_report(results)
        assert "contains: 2" in report
        assert "exact: 1" in report


# ─── save_results / load_history ───────────────────────────────


class TestPersistence:
    def test_save_and_load(self, evaluator, tmp_dir):
        results = [
            EvalResult("t1", "qwen", "你好", 1.0, "exact", 50,
                       timestamp="2024-01-01T00:00:00"),
            EvalResult("t2", "qwen", "世界", 0.9, "contains", 60,
                       timestamp="2024-01-01T00:00:01"),
        ]
        evaluator.save_results(results)

        loaded = evaluator.load_history()
        assert len(loaded) == 2
        assert loaded[0].case_id == "t1"
        assert loaded[1].score == 0.9

    def test_load_filtered_by_model(self, evaluator):
        results = [
            EvalResult("t1", "model_a", "out", 0.8, "exact", 10),
            EvalResult("t2", "model_b", "out", 0.7, "contains", 10),
        ]
        evaluator.save_results(results)

        loaded = evaluator.load_history(model="model_a")
        assert len(loaded) == 1
        assert loaded[0].model == "model_a"

    def test_load_limit(self, evaluator):
        for i in range(10):
            evaluator.save_results([
                EvalResult(f"t{i}", "m", "out", 0.5, "fail", 10)
            ])
        loaded = evaluator.load_history(limit=3)
        assert len(loaded) == 3

    def test_load_no_file(self, evaluator):
        loaded = evaluator.load_history()
        assert loaded == []


# ─── detect_regression ─────────────────────────────────────────


class TestDetectRegression:
    def test_insufficient_data(self, evaluator):
        result = evaluator.detect_regression("model_a")
        assert result["regression_detected"] is False
        assert "数据不足" in result["details"]

    def test_no_regression(self, evaluator):
        # 写入稳定分数的历史数据
        for i in range(20):
            evaluator.save_results([
                EvalResult(f"t{i}", "model_a", "out", 0.8, "exact", 10)
            ])
        result = evaluator.detect_regression("model_a", window=5)
        assert result["regression_detected"] is False

    def test_regression_detected(self, evaluator):
        # 先写入高分历史
        for i in range(15):
            evaluator.save_results([
                EvalResult(f"t{i}", "model_a", "out", 0.9, "exact", 10)
            ])
        # 再写入低分（最近 5 次）
        for i in range(5):
            evaluator.save_results([
                EvalResult(f"r{i}", "model_a", "out", 0.3, "fail", 10)
            ])
        result = evaluator.detect_regression("model_a", window=5)
        assert result["regression_detected"] is True
        assert result["delta"] < -0.1

    def test_regression_different_model_ignored(self, evaluator):
        # model_a 高分，model_b 低分
        for i in range(15):
            evaluator.save_results([
                EvalResult(f"t{i}", "model_a", "out", 0.9, "exact", 10)
            ])
        for i in range(5):
            evaluator.save_results([
                EvalResult(f"r{i}", "model_b", "out", 0.1, "fail", 10)
            ])
        result = evaluator.detect_regression("model_a", window=5)
        # model_a 没有低分数据，不应检测到回归
        assert result["regression_detected"] is False
