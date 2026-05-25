"""
配置管理 — 集中管理所有配置项，消除模块级全局状态
"""

import os
import json
import threading
from pathlib import Path
from dataclasses import dataclass, field, fields
from typing import Optional


@dataclass
class RouterConfig:
    """路由器配置"""
    # Ollama 设置
    ollama_base: str = ""
    local_model: str = "qwen-tool"
    local_max_tokens: int = 2048

    # 云端 API 设置
    cloud_api_url: str = ""
    cloud_api_key: str = ""
    cloud_model: str = ""

    # 成本计算
    cost_per_1k_input: float = 0.003
    cost_per_1k_output: float = 0.015

    # 缓存
    cache_dir: str = ""
    cache_max_entries: int = 1000
    cache_fuzzy_threshold: float = 0.85

    # 缓存 TTL（小时）
    cache_ttl_hours: dict = field(default_factory=lambda: {
        "translation": 168,
        "classification": 168,
        "extraction": 72,
        "formatting": 168,
        "summarization": 24,
        "default": 48,
    })

    # 路由阈值
    base_threshold: float = 3.0
    recurse_score_min: float = 2.5
    recurse_score_max: float = 7.0
    max_recurse_depth: int = 3

    def __post_init__(self) -> None:
        if not self.ollama_base:
            self.ollama_base = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        if not self.cloud_api_url:
            self.cloud_api_url = os.environ.get("CLOUD_API_URL", "")
        if not self.cloud_api_key:
            self.cloud_api_key = os.environ.get("CLOUD_API_KEY", "")
        if not self.cloud_model:
            self.cloud_model = os.environ.get("CLOUD_MODEL", "")
        if not self.cache_dir:
            self.cache_dir = os.environ.get(
                "TASK_ROUTER_CACHE", str(Path.home() / ".cache" / "task_router")
            )

    @classmethod
    def from_yaml(cls, path: str) -> "RouterConfig":
        """从 YAML 配置文件加载"""
        if not os.path.exists(path):
            return cls()
        try:
            import yaml
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            valid = {f.name for f in fields(cls)}
            return cls(**{k: v for k, v in data.items() if k in valid})
        except Exception:
            return cls()

    @classmethod
    def from_json(cls, path: str) -> "RouterConfig":
        """从 JSON 配置文件加载"""
        if not os.path.exists(path):
            return cls()
        try:
            with open(path) as f:
                data = json.load(f)
            valid = {f.name for f in fields(cls)}
            return cls(**{k: v for k, v in data.items() if k in valid})
        except Exception:
            return cls()

    @classmethod
    def from_env(cls) -> "RouterConfig":
        """从配置文件或环境变量创建配置（优先级：YAML > JSON > 环境变量）"""
        # 尝试从配置文件加载
        for candidate in ["config.yaml", "config.yml", "config.json"]:
            config_path = os.path.join(os.path.dirname(__file__), candidate)
            if os.path.exists(config_path):
                if candidate.endswith((".yaml", ".yml")):
                    cfg = cls.from_yaml(config_path)
                else:
                    cfg = cls.from_json(config_path)
                # 如果配置文件中有值，覆盖环境变量默认值
                return cfg
        return cls()

    def get_cache_ttl(self, task_type: str) -> int:
        """获取指定任务类型的缓存 TTL"""
        capability = TASK_TO_CAPABILITY.get(task_type, "default")
        return self.cache_ttl_hours.get(capability, self.cache_ttl_hours.get("default", 48))


# ─── 能力映射 ──────────────────────────────────────────────────

TASK_TO_CAPABILITY: dict[str, str] = {
    "general_classify": "classification",
    "file_classify": "classification",
    "sentiment": "classification",
    "tag": "classification",
    "feedback_classify": "classification",
    "translate_en2zh": "translation",
    "translate_zh2en": "translation",
    "extract_keywords": "extraction",
    "extract_info": "extraction",
    "contract_clause": "extraction",
    "invoice_parse": "extraction",
    "summarize_short": "summarization",
    "meeting_minutes": "summarization",
    "data_report": "summarization",
    "format_json": "formatting",
    "clean_data": "formatting",
    "dedup": "formatting",
    "sort_numbers": "formatting",
    "sort_alpha": "formatting",
    "rename_suggest": "formatting",
    "qa_short": "qa",
    "_count": "formatting",
}


# ─── 全局配置实例（延迟初始化，线程安全）────────────────────────

_config: Optional[RouterConfig] = None
_config_lock = threading.Lock()


def get_config() -> RouterConfig:
    """获取全局配置实例"""
    global _config
    if _config is None:
        with _config_lock:
            if _config is None:
                _config = RouterConfig.from_env()
    return _config


def set_config(config: RouterConfig) -> None:
    """设置全局配置（用于测试或自定义）"""
    global _config
    _config = config
