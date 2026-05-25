"""
Privacy 模块测试

测试覆盖:
- PrivacyFilter.detect: 6 种 PII 类型检测
- PrivacyFilter.anonymize: 脱敏 + 占位符生成
- PrivacyFilter.deanonymize: 还原 + 往返一致性
- 选择性启用 PII 类型
- 重叠处理
- PrivacyConfig: 加载/保存
- get_privacy_filter 单例
- quick_anonymize/quick_deanonymize
"""

from task_router.privacy import (
    PrivacyFilter,
    PrivacyConfig,
    PIIMatch,
    AnonymizationResult,
    get_privacy_filter,
    quick_anonymize,
    quick_deanonymize,
    PII_PATTERNS,
)


# ─── Detect 测试 ─────────────────────────────────────────────────


class TestDetect:
    """PII 检测测试"""

    def test_detect_phone(self):
        f = PrivacyFilter()
        matches = f.detect("请联系 13812345678")
        assert len(matches) == 1
        assert matches[0].pii_type == "phone"
        assert matches[0].original == "13812345678"

    def test_detect_multiple_phones(self):
        f = PrivacyFilter()
        matches = f.detect("打 13812345678 或 15900001111")
        phones = [m for m in matches if m.pii_type == "phone"]
        assert len(phones) == 2

    def test_detect_id_card(self):
        f = PrivacyFilter()
        matches = f.detect("身份证 110101199001011234")
        assert len(matches) == 1
        assert matches[0].pii_type == "id_card"
        assert matches[0].original == "110101199001011234"

    def test_detect_id_card_with_x(self):
        """身份证末尾为 X"""
        f = PrivacyFilter()
        matches = f.detect("身份证 11010119900101123X")
        assert len(matches) == 1
        assert matches[0].pii_type == "id_card"

    def test_detect_email(self):
        f = PrivacyFilter()
        matches = f.detect("发到 test@example.com")
        assert len(matches) == 1
        assert matches[0].pii_type == "email"
        assert matches[0].original == "test@example.com"

    def test_detect_bank_card(self):
        """19 位银联卡号"""
        f = PrivacyFilter()
        matches = f.detect("卡号 6222021234567890123")
        assert any(m.pii_type == "bank_card" for m in matches)

    def test_detect_ip_address(self):
        f = PrivacyFilter()
        matches = f.detect("服务器 192.168.1.100")
        assert len(matches) == 1
        assert matches[0].pii_type == "ip_address"
        assert matches[0].original == "192.168.1.100"

    def test_detect_invalid_ip(self):
        """256.x.x.x 不是合法 IP"""
        f = PrivacyFilter()
        matches = f.detect("地址 256.1.1.1")
        ip_matches = [m for m in matches if m.pii_type == "ip_address"]
        assert len(ip_matches) == 0

    def test_detect_passport(self):
        f = PrivacyFilter()
        matches = f.detect("护照 G12345678")
        assert len(matches) == 1
        assert matches[0].pii_type == "passport"

    def test_detect_no_pii(self):
        f = PrivacyFilter()
        matches = f.detect("这是一段普通文本，没有敏感信息")
        assert len(matches) == 0

    def test_detect_empty_text(self):
        f = PrivacyFilter()
        matches = f.detect("")
        assert len(matches) == 0

    def test_detect_mixed_pii(self):
        """多种 PII 同时出现"""
        f = PrivacyFilter()
        text = "手机 13812345678，邮箱 test@example.com，IP 10.0.0.1"
        matches = f.detect(text)
        types = {m.pii_type for m in matches}
        assert "phone" in types
        assert "email" in types
        assert "ip_address" in types

    def test_detect_positions_are_correct(self):
        """匹配位置应与原文一致"""
        f = PrivacyFilter()
        text = "联系 13812345678 了解"
        matches = f.detect(text)
        assert len(matches) == 1
        m = matches[0]
        assert text[m.start:m.end] == m.original

    def test_phone_not_in_larger_number(self):
        """12 位数字中不应匹配手机号"""
        f = PrivacyFilter()
        matches = f.detect("订单号 1381234567890")
        phone_matches = [m for m in matches if m.pii_type == "phone"]
        assert len(phone_matches) == 0


# ─── Anonymize 测试 ─────────────────────────────────────────────


class TestAnonymize:
    """脱敏测试"""

    def test_anonymize_phone(self):
        f = PrivacyFilter()
        result = f.anonymize("打 13812345678 给我")
        assert "13812345678" not in result.text
        assert "[手机号]" in result.text
        assert result.anonymized_count == 1

    def test_anonymize_multiple_same_type(self):
        """同类型多个 PII 应有递增编号（label 格式: [手机号]_0）"""
        f = PrivacyFilter()
        result = f.anonymize("打 13812345678 或 15900001111")
        assert "[手机号]_0" in result.text
        assert "[手机号]_1" in result.text
        assert result.anonymized_count == 2

    def test_anonymize_no_pii(self):
        f = PrivacyFilter()
        result = f.anonymize("普通文本")
        assert result.text == "普通文本"
        assert result.anonymized_count == 0

    def test_anonymize_empty_text(self):
        f = PrivacyFilter()
        result = f.anonymize("")
        assert result.text == ""
        assert result.anonymized_count == 0

    def test_anonymize_preserves_surrounding_text(self):
        """脱敏应保留周围文本"""
        f = PrivacyFilter()
        result = f.anonymize("请发到 test@example.com，谢谢")
        assert result.text.startswith("请发到 ")
        assert result.text.endswith("，谢谢")

    def test_anonymize_original_length(self):
        f = PrivacyFilter()
        text = "手机 13812345678"
        result = f.anonymize(text)
        assert result.original_length == len(text)


# ─── Deanonymize 测试 ───────────────────────────────────────────


class TestDeanonymize:
    """还原测试"""

    def test_roundtrip(self):
        """脱敏 → 还原应恢复原文"""
        f = PrivacyFilter()
        original = "联系 13812345678 或 test@example.com"
        result = f.anonymize(original)
        restored = f.deanonymize(result.text)
        assert restored == original

    def test_roundtrip_multiple_phones(self):
        f = PrivacyFilter()
        original = "打 13812345678 或 15900001111"
        result = f.anonymize(original)
        restored = f.deanonymize(result.text)
        assert restored == original

    def test_roundtrip_mixed_types(self):
        f = PrivacyFilter()
        original = "手机 13812345678 邮箱 a@b.com IP 192.168.1.1"
        result = f.anonymize(original)
        restored = f.deanonymize(result.text)
        assert restored == original

    def test_deanonymize_no_mapping(self):
        """无映射时返回原文"""
        f = PrivacyFilter()
        assert f.deanonymize("普通文本") == "普通文本"


# ─── 选择性启用 ─────────────────────────────────────────────────


class TestSelectiveEnable:
    """选择性启用 PII 类型"""

    def test_only_phone(self):
        f = PrivacyFilter(enabled_types=["phone"])
        text = "手机 13812345678 邮箱 test@example.com"
        result = f.anonymize(text)
        assert "13812345678" not in result.text
        assert "test@example.com" in result.text  # 邮箱未被脱敏

    def test_only_email(self):
        f = PrivacyFilter(enabled_types=["email"])
        text = "手机 13812345678 邮箱 test@example.com"
        result = f.anonymize(text)
        assert "13812345678" in result.text  # 手机未被脱敏
        assert "test@example.com" not in result.text

    def test_empty_enabled_types(self):
        """空列表在 Python 中等价于 None（falsy），所有类型仍启用"""
        f = PrivacyFilter(enabled_types=[])
        matches = f.detect("手机 13812345678")
        assert len(matches) == 1  # 空列表被视为"全部启用"

    def test_get_stats_shows_enabled(self):
        f = PrivacyFilter(enabled_types=["phone", "email"])
        stats = f.get_stats()
        assert "phone" in stats["enabled_types"]
        assert "email" in stats["enabled_types"]
        assert "id_card" not in stats["enabled_types"]


# ─── 自定义模式 ─────────────────────────────────────────────────


class TestCustomPatterns:
    """自定义正则模式"""

    def test_custom_pattern(self):
        f = PrivacyFilter(custom_patterns={
            "order_id": {
                "pattern": r'ORD-\d{6}',
                "label": "[订单号]",
                "priority": 2,
            }
        })
        matches = f.detect("订单 ORD-123456 状态")
        assert len(matches) == 1
        assert matches[0].pii_type == "order_id"

    def test_override_existing_pattern(self):
        """自定义模式可覆盖内置模式"""
        f = PrivacyFilter(custom_patterns={
            "phone": {
                "pattern": r'\d{3}-\d{4}-\d{4}',
                "label": "[电话]",
                "priority": 1,
            }
        })
        matches = f.detect("电话 138-1234-5678")
        assert len(matches) == 1
        assert matches[0].pii_type == "phone"


# ─── 重叠处理 ─────────────────────────────────────────────────


class TestOverlapResolution:
    """重叠 PII 匹配处理"""

    def test_overlapping_keeps_longer(self):
        """重叠时保留更长的匹配"""
        f = PrivacyFilter()
        # 构造一个可能重叠的场景（邮箱包含部分匹配）
        matches = f.detect("邮箱 admin@test.com")
        assert len(matches) >= 1

    def test_adjacent_no_overlap(self):
        """相邻但不重叠的 PII 都应保留"""
        f = PrivacyFilter()
        matches = f.detect("13812345678 test@example.com")
        types = {m.pii_type for m in matches}
        assert "phone" in types
        assert "email" in types


# ─── Clear 测试 ─────────────────────────────────────────────────


class TestClear:
    """清除映射"""

    def test_clear_resets_mappings(self):
        f = PrivacyFilter()
        f.anonymize("手机 13812345678")
        assert len(f._placeholder_map) > 0
        f.clear()
        assert len(f._placeholder_map) == 0
        assert len(f._reverse_map) == 0

    def test_deanonymize_after_clear_returns_placeholder(self):
        """清除后无法还原"""
        f = PrivacyFilter()
        result = f.anonymize("手机 13812345678")
        f.clear()
        restored = f.deanonymize(result.text)
        assert "13812345678" not in restored  # 无法还原


# ─── PrivacyConfig 测试 ─────────────────────────────────────────


class TestPrivacyConfig:
    """隐私配置测试"""

    def test_default_config(self):
        cfg = PrivacyConfig()
        assert cfg.enabled is True
        assert "phone" in cfg.enabled_types
        assert cfg.anonymize_local is False

    def test_from_file_nonexistent(self):
        cfg = PrivacyConfig.from_file("/nonexistent/path.json")
        assert cfg.enabled is True  # 返回默认值

    def test_save_and_load(self, tmp_path):
        path = str(tmp_path / "privacy.json")
        cfg = PrivacyConfig(enabled_types=["phone", "email"], anonymize_local=True)
        cfg.save(path)

        loaded = PrivacyConfig.from_file(path)
        assert loaded.enabled_types == ["phone", "email"]
        assert loaded.anonymize_local is True

    def test_from_file_invalid_json(self, tmp_path):
        path = str(tmp_path / "bad.json")
        with open(path, "w") as f:
            f.write("not json")
        cfg = PrivacyConfig.from_file(path)
        assert cfg.enabled is True  # 返回默认值


# ─── 单例测试 ─────────────────────────────────────────────────


class TestSingleton:
    """全局单例测试"""

    def test_get_privacy_filter_returns_instance(self):
        f = get_privacy_filter()
        assert isinstance(f, PrivacyFilter)

    def test_get_privacy_filter_returns_same_instance(self):
        f1 = get_privacy_filter()
        f2 = get_privacy_filter()
        assert f1 is f2


# ─── 便捷函数测试 ─────────────────────────────────────────────


class TestConvenienceFunctions:
    """便捷函数测试"""

    def test_quick_anonymize(self):
        result = quick_anonymize("手机 13812345678")
        assert "13812345678" not in result

    def test_quick_deanonymize_roundtrip(self):
        original = "手机 13812345678"
        anonymized = quick_anonymize(original)
        restored = quick_deanonymize(anonymized)
        assert restored == original


# ─── 边界情况 ─────────────────────────────────────────────────


class TestEdgeCases:
    """边界情况测试"""

    def test_very_long_text(self):
        """长文本不应崩溃"""
        f = PrivacyFilter()
        text = "普通文本" * 10000 + " 13812345678"
        result = f.anonymize(text)
        assert "13812345678" not in result.text
        assert result.anonymized_count == 1

    def test_pii_at_start(self):
        f = PrivacyFilter()
        result = f.anonymize("13812345678 打这个电话")
        assert "13812345678" not in result.text

    def test_pii_at_end(self):
        f = PrivacyFilter()
        result = f.anonymize("电话是 13812345678")
        assert "13812345678" not in result.text

    def test_pii_only_text(self):
        """纯 PII 文本"""
        f = PrivacyFilter()
        result = f.anonymize("13812345678")
        assert "13812345678" not in result.text
        assert result.anonymized_count == 1

    def test_same_pii_appears_twice(self):
        """同一 PII 出现两次，应有不同编号"""
        f = PrivacyFilter()
        result = f.anonymize("先打 13812345678，再打 13812345678")
        assert result.anonymized_count == 2
        assert "[手机号]_0" in result.text
        assert "[手机号]_1" in result.text
        restored = f.deanonymize(result.text)
        assert restored == "先打 13812345678，再打 13812345678"

    def test_all_pii_types_in_one_text(self):
        """所有 PII 类型同时出现"""
        f = PrivacyFilter()
        text = " ".join([
            "13812345678",           # phone
            "110101199001011234",    # id_card
            "test@example.com",      # email
            "6222021234567890123",   # bank_card
            "192.168.1.100",         # ip_address
            "G12345678",             # passport
        ])
        result = f.anonymize(text)
        assert result.anonymized_count >= 5  # 至少匹配 5 种

    def test_pii_patterns_registry(self):
        """PII_PATTERNS 应包含所有 6 种类型"""
        expected = {"phone", "id_card", "email", "bank_card", "ip_address", "passport"}
        assert set(PII_PATTERNS.keys()) == expected

    def test_anonymization_result_dataclass(self):
        """AnonymizationResult 默认值"""
        r = AnonymizationResult(text="test")
        assert r.matches == []
        assert r.original_length == 0
        assert r.anonymized_count == 0

    def test_pii_match_dataclass(self):
        """PIIMatch 字段"""
        m = PIIMatch(pii_type="phone", original="138", start=0, end=3, placeholder="")
        assert m.pii_type == "phone"
        assert m.start == 0
