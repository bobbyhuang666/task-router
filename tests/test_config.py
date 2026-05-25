"""
RouterConfig 测试

测试覆盖:
- 默认值
- 环境变量回退
- JSON 文件加载
- get_cache_ttl
- TASK_TO_CAPABILITY 映射
- get_config / set_config 单例
"""

import json


from task_router.config import (
    RouterConfig,
    TASK_TO_CAPABILITY,
    get_config,
    set_config,
)


# ─── RouterConfig 默认值 ───────────────────────────────────────


class TestRouterConfigDefaults:
    def test_default_local_model(self):
        cfg = RouterConfig()
        assert cfg.local_model == "qwen-tool"

    def test_default_threshold(self):
        cfg = RouterConfig()
        assert cfg.base_threshold == 3.0

    def test_default_cache_ttl_has_keys(self):
        cfg = RouterConfig()
        assert "translation" in cfg.cache_ttl_hours
        assert "default" in cfg.cache_ttl_hours

    def test_default_max_recurse_depth(self):
        cfg = RouterConfig()
        assert cfg.max_recurse_depth == 3


# ─── 环境变量回退 ─────────────────────────────────────────────


class TestEnvFallback:
    def test_ollama_base_from_env(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", "http://custom:9999")
        cfg = RouterConfig(ollama_base="")
        assert cfg.ollama_base == "http://custom:9999"

    def test_ollama_base_explicit_overrides_env(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", "http://custom:9999")
        cfg = RouterConfig(ollama_base="http://explicit:1111")
        assert cfg.ollama_base == "http://explicit:1111"

    def test_cloud_api_from_env(self, monkeypatch):
        monkeypatch.setenv("CLOUD_API_URL", "https://api.example.com")
        monkeypatch.setenv("CLOUD_API_KEY", "sk-test")
        monkeypatch.setenv("CLOUD_MODEL", "gpt-4")
        cfg = RouterConfig(cloud_api_url="", cloud_api_key="", cloud_model="")
        assert cfg.cloud_api_url == "https://api.example.com"
        assert cfg.cloud_api_key == "sk-test"
        assert cfg.cloud_model == "gpt-4"

    def test_cache_dir_from_env(self, monkeypatch):
        monkeypatch.setenv("TASK_ROUTER_CACHE", "/tmp/test_cache")
        cfg = RouterConfig(cache_dir="")
        assert cfg.cache_dir == "/tmp/test_cache"


# ─── JSON 加载 ─────────────────────────────────────────────────


class TestJsonLoading:
    def test_from_json_valid(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "local_model": "custom-model",
            "base_threshold": 5.0,
        }))
        cfg = RouterConfig.from_json(str(config_file))
        assert cfg.local_model == "custom-model"
        assert cfg.base_threshold == 5.0

    def test_from_json_missing_file(self):
        cfg = RouterConfig.from_json("/nonexistent/config.json")
        # 应返回默认配置
        assert cfg.local_model == "qwen-tool"

    def test_from_json_invalid_json(self, tmp_path):
        config_file = tmp_path / "bad.json"
        config_file.write_text("not valid json {{{")
        cfg = RouterConfig.from_json(str(config_file))
        assert cfg.local_model == "qwen-tool"

    def test_from_json_ignores_unknown_keys(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "local_model": "test",
            "unknown_key": "value",
        }))
        cfg = RouterConfig.from_json(str(config_file))
        assert cfg.local_model == "test"
        assert not hasattr(cfg, "unknown_key")


# ─── get_cache_ttl ─────────────────────────────────────────────


class TestGetCacheTtl:
    def test_known_task_type(self):
        cfg = RouterConfig()
        ttl = cfg.get_cache_ttl("translate_en2zh")
        assert ttl == 168  # translation → 168 hours

    def test_unknown_task_type_falls_back(self):
        cfg = RouterConfig()
        ttl = cfg.get_cache_ttl("unknown_task")
        assert ttl == 48  # default

    def test_sentiment_maps_to_classification(self):
        cfg = RouterConfig()
        ttl = cfg.get_cache_ttl("sentiment")
        assert ttl == 168  # classification → 168 hours


# ─── TASK_TO_CAPABILITY ────────────────────────────────────────


class TestTaskToCapability:
    def test_translation_tasks(self):
        assert TASK_TO_CAPABILITY["translate_en2zh"] == "translation"
        assert TASK_TO_CAPABILITY["translate_zh2en"] == "translation"

    def test_classification_tasks(self):
        assert TASK_TO_CAPABILITY["sentiment"] == "classification"
        assert TASK_TO_CAPABILITY["general_classify"] == "classification"

    def test_extraction_tasks(self):
        assert TASK_TO_CAPABILITY["extract_keywords"] == "extraction"

    def test_all_values_are_strings(self):
        for k, v in TASK_TO_CAPABILITY.items():
            assert isinstance(v, str), f"{k} maps to non-string: {v}"


# ─── get_config / set_config ───────────────────────────────────


class TestGlobalConfig:
    def setup_method(self):
        """每个测试前重置全局配置"""
        set_config(None)

    def teardown_method(self):
        """每个测试后重置全局配置"""
        set_config(None)

    def test_get_config_creates_instance(self):
        cfg = get_config()
        assert isinstance(cfg, RouterConfig)

    def test_get_config_returns_same_instance(self):
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_set_config_overrides(self):
        custom = RouterConfig(local_model="custom")
        set_config(custom)
        cfg = get_config()
        assert cfg.local_model == "custom"
        assert cfg is custom
