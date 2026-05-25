"""
自适应 Prompt 压缩器测试
"""

from task_router.adaptive_compression import (
    estimate_importance,
    compress_adaptive,
    get_token_savings,
    CompressionResult,
)


class TestImportanceEstimation:
    """语义重要性评估"""

    def test_short_sentence_low_importance(self):
        """过短句子应有低重要性"""
        score = estimate_importance("是")
        assert score < 0.3

    def test_instruction_sentence_high_importance(self):
        """指令性句子应有高重要性"""
        score = estimate_importance("请确保输出格式为 JSON")
        assert score > 0.6

    def test_number_sentence_bonus(self):
        """包含数字的句子应有额外重要性"""
        score_with = estimate_importance("共有 42 个测试用例")
        score_without = estimate_importance("共有多个测试用例")
        assert score_with >= score_without

    def test_format_sentence_bonus(self):
        """包含格式要求的句子应有额外重要性"""
        score = estimate_importance("请以表格形式输出结果")
        assert score > 0.6

    def test_separator_high_importance(self):
        """分隔符应有高重要性（结构标记）"""
        score = estimate_importance("========")
        assert score > 0.5

    def test_example_sentence_bonus(self):
        """示例标记应有额外重要性"""
        score = estimate_importance("例如：输入 hello 输出 world")
        assert score > 0.5

    def test_normal_sentence_medium(self):
        """普通句子应有中等重要性"""
        score = estimate_importance("这是一个正常的句子，描述了某个概念。")
        assert 0.3 < score < 0.8


class TestAdaptiveCompression:
    """自适应压缩"""

    def test_short_prompt_no_compression(self):
        """短 prompt 不应被压缩"""
        prompt = "翻译这句话"
        result = compress_adaptive(prompt, confidence=0.9)
        assert result.level == "none"
        assert result.compression_ratio == 1.0

    def test_high_confidence_aggressive(self):
        """高置信度应使用激进压缩"""
        prompt = "\n".join([f"这是第 {i} 行内容，包含一些描述信息。" for i in range(30)])
        result = compress_adaptive(prompt, confidence=0.9)
        assert result.level == "aggressive"
        assert result.compression_ratio < 0.7

    def test_medium_confidence_moderate(self):
        """中等置信度应使用中度压缩"""
        prompt = "\n".join([f"这是第 {i} 行内容，包含一些描述信息。" for i in range(30)])
        result = compress_adaptive(prompt, confidence=0.7)
        assert result.level == "moderate"
        assert result.compression_ratio < 0.9

    def test_low_confidence_no_compression(self):
        """低置信度不应压缩"""
        prompt = "\n".join([f"这是第 {i} 行内容，包含一些描述信息。" for i in range(30)])
        result = compress_adaptive(prompt, confidence=0.3)
        assert result.level == "none"
        assert result.compression_ratio == 1.0

    def test_target_ratio_override(self):
        """target_ratio 应覆盖自动选择"""
        prompt = "\n".join([f"这是第 {i} 行内容，包含一些描述信息。" for i in range(30)])
        result = compress_adaptive(prompt, confidence=0.3, target_ratio=0.5)
        assert result.compression_ratio <= 0.7  # 允许一定误差

    def test_returns_result_object(self):
        """应返回 CompressionResult 对象"""
        prompt = "\n".join([f"这是第 {i} 行内容。" for i in range(20)])
        result = compress_adaptive(prompt, confidence=0.5)
        assert isinstance(result, CompressionResult)
        assert result.original_length > 0
        assert result.compressed_length > 0
        assert 0 < result.compression_ratio <= 1.0

    def test_preserves_important_lines(self):
        """应保留重要行（指令、格式要求）"""
        lines = [
            "请以 JSON 格式输出结果，确保包含以下字段：",  # 重要
            "这是一段普通的描述文本。",  # 不重要
            "另一个普通的段落。",  # 不重要
            "返回格式必须包含 id, name, value 三个字段。",  # 重要
        ]
        prompt = "\n".join(lines * 5)  # 重复以达到压缩阈值
        result = compress_adaptive(prompt, confidence=0.8)
        # 重要行应被保留
        assert "JSON" in result.compressed_prompt or "格式" in result.compressed_prompt


class TestTokenSavings:
    """Token 节省估算"""

    def test_high_confidence_savings(self):
        """高置信度应节省最多 token"""
        savings = get_token_savings(0.9, 1000)
        assert savings == 600

    def test_medium_confidence_savings(self):
        """中等置信度应节省中等 token"""
        savings = get_token_savings(0.7, 1000)
        assert savings == 400

    def test_low_confidence_no_savings(self):
        """低置信度不应节省 token"""
        savings = get_token_savings(0.3, 1000)
        assert savings == 0

    def test_zero_tokens(self):
        """零 token 输入"""
        savings = get_token_savings(0.9, 0)
        assert savings == 0
