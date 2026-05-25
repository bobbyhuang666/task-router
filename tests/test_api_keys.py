"""
多 API Key 管理 + 用量告警测试

测试覆盖:
- ApiKeyConfig: API Key 配置
- ApiKeyManager: 多 Key 管理
- QuotaManager: 用量告警
"""

import pytest
import tempfile

from task_router.audit import ApiKeyConfig, ApiKeyManager, QuotaManager


# ─── ApiKeyConfig 测试 ──────────────────────────────────────────────


class TestApiKeyConfig:
    """API Key 配置"""

    def test_default_values(self):
        """默认值正确"""
        config = ApiKeyConfig(key="test-key")
        assert config.team == "default"
        assert config.enabled is True
        assert config.monthly_task_limit == 10000
        assert config.monthly_token_limit == 1000000
        assert config.monthly_cost_limit == 100.0
        assert config.allowed_models == []

    def test_custom_values(self):
        """自定义值"""
        config = ApiKeyConfig(
            key="tk-market-001",
            team="市场部",
            description="市场团队专用",
            monthly_task_limit=5000,
            allowed_models=["gpt-4o-mini"],
        )
        assert config.team == "市场部"
        assert config.monthly_task_limit == 5000
        assert "gpt-4o-mini" in config.allowed_models


# ─── ApiKeyManager 测试 ──────────────────────────────────────────────


class TestApiKeyManager:
    """多 Key 管理器"""

    @pytest.fixture
    def tmp_dir(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    @pytest.fixture
    def manager(self, tmp_dir):
        return ApiKeyManager(cache_dir=tmp_dir)

    def test_add_key(self, manager):
        """添加 Key"""
        config = manager.add_key(key="tk-test-001", team="测试组")
        assert config.key == "tk-test-001"
        assert config.team == "测试组"
        assert len(manager.keys) == 1

    def test_remove_key(self, manager):
        """删除 Key"""
        manager.add_key(key="tk-test-001", team="测试组")
        assert manager.remove_key("tk-test-001") is True
        assert len(manager.keys) == 0

    def test_remove_nonexistent_key(self, manager):
        """删除不存在的 Key"""
        assert manager.remove_key("nonexistent") is False

    def test_authenticate_valid_key(self, manager):
        """验证有效 Key"""
        manager.add_key(key="tk-test-001", team="测试组")
        config = manager.authenticate("tk-test-001")
        assert config is not None
        assert config.team == "测试组"

    def test_authenticate_invalid_key(self, manager):
        """验证无效 Key"""
        manager.add_key(key="tk-test-001", team="测试组")
        config = manager.authenticate("wrong-key")
        assert config is None

    def test_authenticate_disabled_key(self, manager):
        """验证已禁用的 Key"""
        manager.add_key(key="tk-test-001", team="测试组")
        manager.enable_key("tk-test-001", enabled=False)
        config = manager.authenticate("tk-test-001")
        assert config is None

    def test_enable_disable_key(self, manager):
        """启用/禁用 Key"""
        manager.add_key(key="tk-test-001", team="测试组")
        assert manager.enable_key("tk-test-001", enabled=False) is True
        assert manager.keys["tk-test-001"].enabled is False
        assert manager.enable_key("tk-test-001", enabled=True) is True
        assert manager.keys["tk-test-001"].enabled is True

    def test_check_model_access_no_restriction(self, manager):
        """无模型限制时允许所有模型"""
        manager.add_key(key="tk-test-001", team="测试组")
        assert manager.check_model_access("tk-test-001", "gpt-4o") is True
        assert manager.check_model_access("tk-test-001", "claude-3") is True

    def test_check_model_access_with_restriction(self, manager):
        """有模型限制时只允许指定模型"""
        manager.add_key(key="tk-test-001", team="测试组", allowed_models=["gpt-4o-mini"])
        assert manager.check_model_access("tk-test-001", "gpt-4o-mini") is True
        assert manager.check_model_access("tk-test-001", "gpt-4o") is False

    def test_check_model_access_invalid_key(self, manager):
        """无效 Key 无权限"""
        assert manager.check_model_access("invalid", "gpt-4o") is False

    def test_get_key_info_masked(self, manager):
        """Key 信息脱敏"""
        manager.add_key(key="tk-test-001-very-long-key", team="测试组")
        info = manager.get_key_info("tk-test-001-very-long-key")
        assert info is not None
        assert "****" in info["key_prefix"]
        assert info["key_prefix"].startswith("tk-t")

    def test_list_keys(self, manager):
        """列出所有 Key"""
        manager.add_key(key="tk-001", team="A组")
        manager.add_key(key="tk-002", team="B组")
        keys = manager.list_keys()
        assert len(keys) == 2

    def test_get_all_teams(self, manager):
        """获取所有团队"""
        manager.add_key(key="tk-001", team="市场部")
        manager.add_key(key="tk-002", team="研发部")
        manager.add_key(key="tk-003", team="市场部")
        teams = manager.get_all_teams()
        assert len(teams) == 2
        assert "市场部" in teams
        assert "研发部" in teams

    def test_persistence(self, tmp_dir):
        """数据持久化"""
        manager1 = ApiKeyManager(cache_dir=tmp_dir)
        manager1.add_key(key="tk-001", team="测试组")

        manager2 = ApiKeyManager(cache_dir=tmp_dir)
        assert len(manager2.keys) == 1
        assert "tk-001" in manager2.keys

    def test_last_used_updated(self, manager):
        """使用时更新最后使用时间"""
        manager.add_key(key="tk-001", team="测试组")
        config = manager.authenticate("tk-001")
        assert config.last_used_at != ""


# ─── 用量告警测试 ──────────────────────────────────────────────


class TestQuotaAlerts:
    """用量告警"""

    @pytest.fixture
    def tmp_dir(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    @pytest.fixture
    def quota_manager(self, tmp_dir):
        qm = QuotaManager(cache_dir=tmp_dir)
        qm.set_quota("test-user", daily_task_limit=100, daily_token_limit=10000, daily_cost_limit=5.0)
        return qm

    def test_no_alerts_when_low_usage(self, quota_manager):
        """低用量无告警"""
        alerts = quota_manager.check_alerts("test-user")
        assert len(alerts) == 0

    def test_alert_at_80_percent(self, quota_manager):
        """80% 用量触发告警"""
        # 记录 80 个任务（80%）
        for _ in range(80):
            quota_manager.record_usage("test-user", tokens=100, cost=0.05)
        alerts = quota_manager.check_alerts("test-user")
        assert len(alerts) > 0
        assert any("接近上限" in a for a in alerts)

    def test_alert_at_100_percent(self, quota_manager):
        """100% 用量触发严重告警"""
        # 记录 100 个任务（100%）
        for _ in range(100):
            quota_manager.record_usage("test-user", tokens=100, cost=0.05)
        alerts = quota_manager.check_alerts("test-user")
        assert any("已达上限" in a for a in alerts)

    def test_custom_threshold(self, quota_manager):
        """自定义告警阈值"""
        # 记录 50 个任务（50%）
        for _ in range(50):
            quota_manager.record_usage("test-user", tokens=100, cost=0.05)

        # 50% 不触发 80% 阈值
        alerts_80 = quota_manager.check_alerts("test-user", alert_threshold=80.0)
        assert len(alerts_80) == 0

        # 50% 触发 40% 阈值
        alerts_40 = quota_manager.check_alerts("test-user", alert_threshold=40.0)
        assert len(alerts_40) > 0

    def test_get_usage_summary(self, quota_manager):
        """获取使用量摘要"""
        for _ in range(30):
            quota_manager.record_usage("test-user", tokens=100, cost=0.05)
        summary = quota_manager.get_usage_summary("test-user")
        assert summary["daily_tasks"]["used"] == 30
        assert summary["daily_tasks"]["limit"] == 100
        assert summary["daily_tasks"]["pct"] == 30.0

    def test_check_all_alerts(self, quota_manager):
        """检查所有用户告警"""
        quota_manager.set_quota("user-a", daily_task_limit=100)
        quota_manager.set_quota("user-b", daily_task_limit=100)

        # user-a 高用量
        for _ in range(90):
            quota_manager.record_usage("user-a", tokens=100)
        # user-b 低用量
        for _ in range(10):
            quota_manager.record_usage("user-b", tokens=100)

        alerts = quota_manager.check_all_alerts()
        # 只有 user-a 应该有告警
        assert any("user-a" in a for a in alerts)
        assert not any("user-b" in a for a in alerts)

    def test_no_alerts_for_empty_usage(self, quota_manager):
        """无使用记录时无告警"""
        alerts = quota_manager.check_all_alerts()
        assert len(alerts) == 0
