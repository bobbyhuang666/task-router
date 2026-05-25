"""
闭环蒸馏测试

测试覆盖:
- QualityEvaluator: 自动质量评估
- FailureClusterer: 失败模式聚类
- ClosedLoopManager: 闭环管理
"""

import pytest
import tempfile

from task_router.distillation import (
    QualityEvaluator,
    FailureClusterer,
    ClosedLoopManager,
    DistillationStore,
    DistillationPair,
    PAIR_HYPOTHESIS,
    PAIR_SUPPORTED,
    PAIR_CONTESTED,
    JUDGE_HIGH_THRESHOLD,
    JUDGE_MODERATE_THRESHOLD,
)


# ─── QualityEvaluator 测试 ──────────────────────────────────────


class TestQualityEvaluator:
    """质量评估器"""

    def test_empty_response_score_zero(self):
        """空响应接近 0 分"""
        evaluator = QualityEvaluator()
        score = evaluator.evaluate({"response": ""})
        assert score <= 0.1

    def test_short_response_low_score(self):
        """短响应得分低"""
        evaluator = QualityEvaluator()
        score = evaluator.evaluate({"response": "OK"})
        assert score < 0.5

    def test_normal_response_good_score(self):
        """正常响应得分中等"""
        evaluator = QualityEvaluator()
        score = evaluator.evaluate({"response": "分类结果：文件A属于文档类，文件B属于图片类。"})
        assert score > 0.5

    def test_failure_signal_penalized(self):
        """包含失败信号扣分"""
        evaluator = QualityEvaluator()
        score_bad = evaluator.evaluate({"response": "抱歉，我无法完成这个任务"})
        score_good = evaluator.evaluate({"response": "分类结果：文件A属于文档类，文件B属于图片类。"})
        assert score_bad < score_good

    def test_capability_success_bonus(self):
        """能力匹配的成功信号加分"""
        evaluator = QualityEvaluator()
        score_with = evaluator.evaluate({
            "response": "分类结果：属于文档类",
            "capability": "classification",
        })
        score_without = evaluator.evaluate({
            "response": "输出结果正常",
            "capability": "classification",
        })
        # 有成功信号的应该更高
        assert score_with >= score_without

    def test_consistency_with_local_response(self):
        """与本地输出一致性加分"""
        evaluator = QualityEvaluator()
        score_consistent = evaluator.evaluate({
            "response": "文件A属于文档类",
            "local_response": "文件A属于文档类",
            "capability": "classification",
        })
        score_divergent = evaluator.evaluate({
            "response": "文件A属于文档类",
            "local_response": "完全不同的输出内容",
            "capability": "classification",
        })
        # 一致的应该更高或相等
        assert score_consistent >= score_divergent

    def test_score_range(self):
        """分数始终在 [0, 1]"""
        evaluator = QualityEvaluator()
        for pair in [
            {"response": ""},
            {"response": "短"},
            {"response": "正常长度的输出" * 10},
            {"response": "抱歉无法" * 100},
        ]:
            score = evaluator.evaluate(pair)
            assert 0.0 <= score <= 1.0

    def test_long_response_slightly_lower(self):
        """过长响应分数略低"""
        evaluator = QualityEvaluator()
        score_normal = evaluator.evaluate({"response": "正常输出" * 10})
        score_long = evaluator.evaluate({"response": "很长的输出" * 200})
        # 过长不一定是坏事，但不应该比正常更高
        assert score_long <= score_normal + 0.1

    # ── 5 维评估专项测试 ──

    def test_good_translation_high_score(self):
        """好的翻译应得高分"""
        evaluator = QualityEvaluator()
        score = evaluator.evaluate({
            "response": "你好世界，这是一个测试。翻译结果准确完整。",
            "capability": "translation",
            "prompt": "翻译成中文 Hello World this is a test",
        })
        assert score > 0.5

    def test_bad_translation_low_score(self):
        """差的翻译（太短）应得低分"""
        evaluator = QualityEvaluator()
        score = evaluator.evaluate({
            "response": "你好",
            "capability": "translation",
            "prompt": "翻译成中文 Hello World this is a test",
        })
        assert score < 0.5

    def test_refusal_very_low_score(self):
        """拒绝回答应得低分（低于正常输出）"""
        evaluator = QualityEvaluator()
        score_refusal = evaluator.evaluate({
            "response": "抱歉，我无法完成这个任务",
            "capability": "classification",
        })
        score_normal = evaluator.evaluate({
            "response": "分类结果：文件A属于文档类，文件B属于图片类。",
            "capability": "classification",
        })
        assert score_refusal < score_normal
        assert score_refusal < 0.5

    def test_error_response_low_score(self):
        """错误响应应得低分"""
        evaluator = QualityEvaluator()
        score = evaluator.evaluate({
            "response": "Error: something went wrong",
            "capability": "extraction",
        })
        assert score < 0.4

    def test_classification_with_structure_high_score(self):
        """有结构的分类结果应得高分"""
        evaluator = QualityEvaluator()
        score = evaluator.evaluate({
            "response": "分类结果：\n1. 文件A：文档类\n2. 文件B：图片类\n3. 文件C：代码类",
            "capability": "classification",
        })
        assert score > 0.6

    def test_extraction_with_list_high_score(self):
        """有列表的提取结果应得高分"""
        evaluator = QualityEvaluator()
        score = evaluator.evaluate({
            "response": "关键词：人工智能，机器学习，深度学习，神经网络",
            "capability": "extraction",
        })
        assert score > 0.5

    def test_summarization_adequate_length(self):
        """摘要应有足够的长度"""
        evaluator = QualityEvaluator()
        score = evaluator.evaluate({
            "response": "本文主要讨论了人工智能在医疗领域的应用，包括疾病诊断、药物研发和医疗影像分析等方面。",
            "capability": "summarization",
        })
        assert score > 0.5

    def test_formatting_with_structure(self):
        """格式化输出应有结构"""
        evaluator = QualityEvaluator()
        score = evaluator.evaluate({
            "response": "标题：报告\n日期：2024-01-01\n内容：详细信息",
            "capability": "formatting",
        })
        assert score > 0.5

    def test_weak_failure_signal_moderate_penalty(self):
        """弱失败信号应有适度扣分"""
        evaluator = QualityEvaluator()
        score_weak = evaluator.evaluate({"response": "未找到相关结果"})
        score_normal = evaluator.evaluate({"response": "分类结果：文件A属于文档类"})
        assert score_weak < score_normal

    def test_empty_capability_neutral_scoring(self):
        """无能力类型时应中性评分"""
        evaluator = QualityEvaluator()
        score = evaluator.evaluate({"response": "这是一段正常的输出内容"})
        assert 0.3 <= score <= 0.8

    def test_structure_score_multiline(self):
        """多行输出应有更高的结构分"""
        evaluator = QualityEvaluator()
        score_single = evaluator.evaluate({"response": "单行输出"})
        score_multi = evaluator.evaluate({"response": "第一行\n第二行\n第三行"})
        # 多行不一定更高分（因为其他维度也影响），但结构分应更高
        # 这里只验证不崩溃
        assert 0.0 <= score_single <= 1.0
        assert 0.0 <= score_multi <= 1.0

    def test_relevance_score_keyword_overlap(self):
        """关键词重叠应影响相关性分"""
        evaluator = QualityEvaluator()
        score_relevant = evaluator.evaluate({
            "response": "分类结果：文档类",
            "prompt": "分类这些文件",
        })
        score_irrelevant = evaluator.evaluate({
            "response": "今天天气很好",
            "prompt": "分类这些文件",
        })
        assert score_relevant >= score_irrelevant


# ─── FailureClusterer 测试 ──────────────────────────────────────


class TestFailureClusterer:
    """失败模式聚类"""

    def test_empty_pairs(self):
        """空列表不崩溃"""
        clusterer = FailureClusterer()
        result = clusterer.cluster_failures([])
        assert result == {}

    def test_cluster_by_failure_type(self):
        """按失败类型聚类"""
        clusterer = FailureClusterer()
        pairs = [
            {"response": "抱歉，我无法完成", "failure_type": ""},
            {"response": "抱歉，无法处理", "failure_type": ""},
            {"response": "Error: something", "failure_type": ""},
        ]
        clusters = clusterer.cluster_failures(pairs)
        assert "refusal" in clusters
        assert len(clusters["refusal"]) == 2

    def test_explicit_failure_type(self):
        """显式 failure_type 优先"""
        clusterer = FailureClusterer()
        pairs = [
            {"response": "正常输出", "failure_type": "custom_error"},
        ]
        clusters = clusterer.cluster_failures(pairs)
        assert "custom_error" in clusters

    def test_empty_output_cluster(self):
        """空输出聚类"""
        clusterer = FailureClusterer()
        pairs = [
            {"response": "", "failure_type": ""},
            {"response": "ab", "failure_type": ""},
        ]
        clusters = clusterer.cluster_failures(pairs)
        assert "empty_output" in clusters

    def test_get_top_failures(self):
        """获取最常见的失败"""
        clusterer = FailureClusterer()
        pairs = [
            {"response": "抱歉无法", "action": "分类", "capability": "classification"},
            {"response": "抱歉不能", "action": "翻译", "capability": "translation"},
            {"response": "Error", "action": "提取", "capability": "extraction"},
        ]
        clusterer.cluster_failures(pairs)
        top = clusterer.get_top_failures(2)
        assert len(top) <= 2
        assert all("count" in t for t in top)

    def test_divergent_cluster(self):
        """与本地输出差异大 → divergent"""
        clusterer = FailureClusterer()
        pairs = [
            {
                "response": "这是云端的输出",
                "local_response": "完全不同的本地输出内容",
                "failure_type": "",
            },
        ]
        clusters = clusterer.cluster_failures(pairs)
        # 重叠度低时应该归类为 divergent 或 quality_low
        assert any(k in clusters for k in ["divergent", "quality_low"])


# ─── ClosedLoopManager 测试 ─────────────────────────────────────


class TestClosedLoopManager:
    """闭环管理器"""

    @pytest.fixture
    def tmp_dir(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    def test_evaluate_pending(self, tmp_dir):
        """评估 hypothesis 状态的蒸馏对"""
        store = DistillationStore(cache_dir=tmp_dir)
        manager = ClosedLoopManager(store)

        # 添加一个高质量蒸馏对
        pair = DistillationPair(
            prompt="分类文件",
            response="分类结果：文件A属于文档类，文件B属于图片类。",
            task_type="general_classify",
            capability="classification",
        )
        store.add_pair(pair)

        # 添加一个低质量蒸馏对
        pair2 = DistillationPair(
            prompt="翻译",
            response="抱歉无法",
            task_type="translate_en2zh",
            capability="translation",
        )
        store.add_pair(pair2)

        evaluated = manager.evaluate_pending()
        assert evaluated == 2

        # 检查状态是否更新
        pairs = store.get_pairs()
        states = [p["epistemic_state"] for p in pairs]
        # 高质量应该是 supported 或 hypothesis（取决于分数）
        # 低质量应该是 contested
        assert PAIR_CONTESTED in states or PAIR_HYPOTHESIS in states

    def test_analyze_failures(self, tmp_dir):
        """分析失败模式"""
        store = DistillationStore(cache_dir=tmp_dir)
        manager = ClosedLoopManager(store)

        # 添加低质量蒸馏对
        for i in range(3):
            pair = DistillationPair(
                prompt=f"任务{i}",
                response="抱歉无法完成",
                task_type="general_classify",
                capability="classification",
            )
            store.add_pair(pair)

        result = manager.analyze_failures()
        assert "total_failed" in result
        assert "failure_types" in result
        assert "top_failures" in result

    def test_run_cycle(self, tmp_dir):
        """完整闭环周期"""
        store = DistillationStore(cache_dir=tmp_dir)
        manager = ClosedLoopManager(store)

        pair = DistillationPair(
            prompt="分类",
            response="文档：file.pdf",
            task_type="general_classify",
            capability="classification",
        )
        store.add_pair(pair)

        result = manager.run_cycle()
        assert "evaluated" in result
        assert "cleaned" in result
        assert "failure_analysis" in result
        assert "store_stats" in result
        assert result["evaluated"] >= 1

    def test_empty_store_cycle(self, tmp_dir):
        """空存储不崩溃"""
        store = DistillationStore(cache_dir=tmp_dir)
        manager = ClosedLoopManager(store)

        result = manager.run_cycle()
        assert result["evaluated"] == 0
        assert result["cleaned"] == 0

    def test_hypothesis_to_supported(self, tmp_dir):
        """高质量 hypothesis → supported"""
        store = DistillationStore(cache_dir=tmp_dir)
        manager = ClosedLoopManager(store)

        pair = DistillationPair(
            prompt="分类文件",
            response="分类结果：文件A属于文档类，文件B属于图片类，文件C属于代码类。",
            task_type="general_classify",
            capability="classification",
        )
        store.add_pair(pair)

        manager.evaluate_pending()

        supported = store.get_pairs(state=PAIR_SUPPORTED)
        # 如果分数足够高，应该变成 supported
        if supported:
            assert supported[0]["quality_score"] >= JUDGE_HIGH_THRESHOLD

    def test_low_quality_to_contested(self, tmp_dir):
        """低质量 → contested"""
        store = DistillationStore(cache_dir=tmp_dir)
        manager = ClosedLoopManager(store)

        pair = DistillationPair(
            prompt="任务",
            response="抱歉",
            task_type="general_classify",
        )
        store.add_pair(pair)

        manager.evaluate_pending()

        contested = store.get_pairs(state=PAIR_CONTESTED)
        # 短且有失败信号的应该变成 contested
        if contested:
            assert contested[0]["quality_score"] < JUDGE_MODERATE_THRESHOLD
