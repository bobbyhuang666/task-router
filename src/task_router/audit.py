"""
审计日志系统 — 企业级操作追踪

核心功能:
- 记录所有任务执行（谁、什么时候、做了什么、结果如何）
- 支持 JSON 格式输出（便于 ELK/Splunk 集成）
- 支持按时间/用户/操作类型查询
- 支持配额管理和速率限制
- 支持多 API Key 按团队管理
- 支持用量告警（80% 配额通知）
"""

import os
import json
import time
import threading
import logging
from dataclasses import dataclass, asdict, field
from collections import defaultdict

from task_router.config import get_config
from task_router.io_utils import read_jsonl, append_jsonl

log = logging.getLogger("audit")


# ─── 审计事件 ──────────────────────────────────────────────────

@dataclass
class AuditEvent:
    """审计事件"""
    timestamp: str
    event_type: str          # task_execution / model_change / config_change / api_call
    user_id: str = "system"  # 用户标识（API 场景中来自 API Key）
    action: str = ""         # 具体操作
    details: dict = None     # 详细信息
    ip_address: str = ""     # 客户端 IP
    status: str = "success"  # success / failure / warning
    duration_ms: int = 0     # 执行耗时

    def __post_init__(self):
        if self.details is None:
            self.details = {}


# ─── 多 API Key 管理 ──────────────────────────────────────────────

@dataclass
class ApiKeyConfig:
    """API Key 配置"""
    key: str                          # API Key 值
    team: str = "default"             # 团队名称
    description: str = ""             # 描述
    role: str = "user"                # 角色: admin / user / readonly
    enabled: bool = True              # 是否启用
    monthly_task_limit: int = 10000   # 每月任务数上限
    monthly_token_limit: int = 1000000  # 每月 token 上限
    monthly_cost_limit: float = 100.0   # 每月成本上限（美元）
    allowed_models: list = field(default_factory=list)  # 允许的模型（空=全部）
    created_at: str = ""              # 创建时间
    last_used_at: str = ""            # 最后使用时间


class ApiKeyManager:
    """多 API Key 管理器"""

    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or get_config().cache_dir
        self.keys_file = os.path.join(self.cache_dir, "api_keys.json")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.keys: dict[str, ApiKeyConfig] = {}
        self._lock = threading.RLock()
        self._dirty = False
        self._load_keys()

    def _load_keys(self):
        if not os.path.exists(self.keys_file):
            return
        try:
            with open(self.keys_file) as f:
                data = json.load(f)
            for key_val, config in data.items():
                self.keys[key_val] = ApiKeyConfig(**config)
        except (json.JSONDecodeError, TypeError):
            pass

    def _save_keys(self):
        with self._lock:
            data = {k: asdict(v) for k, v in self.keys.items()}
            with open(self.keys_file, "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self._dirty = False

    def flush(self):
        """将内存中的变更写入磁盘（定期调用）"""
        if self._dirty:
            self._save_keys()

    def add_key(self, key: str, team: str, description: str = "",
                role: str = "user",
                monthly_task_limit: int = 10000,
                monthly_token_limit: int = 1000000,
                monthly_cost_limit: float = 100.0,
                allowed_models: list = None) -> ApiKeyConfig:
        """添加新的 API Key"""
        config = ApiKeyConfig(
            key=key,
            team=team,
            description=description,
            role=role,
            monthly_task_limit=monthly_task_limit,
            monthly_token_limit=monthly_token_limit,
            monthly_cost_limit=monthly_cost_limit,
            allowed_models=allowed_models or [],
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        with self._lock:
            self.keys[key] = config
        self._save_keys()
        return config

    def remove_key(self, key: str) -> bool:
        """删除 API Key"""
        with self._lock:
            if key in self.keys:
                del self.keys[key]
                self._save_keys()
                return True
        return False

    def enable_key(self, key: str, enabled: bool = True) -> bool:
        """启用/禁用 API Key"""
        with self._lock:
            if key in self.keys:
                self.keys[key].enabled = enabled
                self._save_keys()
                return True
        return False

    def authenticate(self, key: str) -> ApiKeyConfig | None:
        """验证 API Key，返回配置或 None（只更新内存，不写磁盘）"""
        config = self.keys.get(key)
        if config and config.enabled:
            config.last_used_at = time.strftime("%Y-%m-%dT%H:%M:%S")
            self._dirty = True
            return config
        return None

    def is_admin(self, key: str) -> bool:
        """检查 Key 是否为管理员角色"""
        config = self.keys.get(key)
        return config is not None and config.enabled and config.role == "admin"

    def check_model_access(self, key: str, model: str) -> bool:
        """检查 Key 是否有权限使用指定模型"""
        config = self.keys.get(key)
        if not config or not config.enabled:
            return False
        if not config.allowed_models:
            return True
        return model in config.allowed_models

    def get_key_info(self, key: str) -> dict | None:
        """获取 Key 信息（脱敏）"""
        config = self.keys.get(key)
        if not config:
            return None
        return {
            "key_prefix": key[:4] + "****" if len(key) > 4 else "****",
            "team": config.team,
            "description": config.description,
            "role": config.role,
            "enabled": config.enabled,
            "monthly_task_limit": config.monthly_task_limit,
            "monthly_token_limit": config.monthly_token_limit,
            "monthly_cost_limit": config.monthly_cost_limit,
            "allowed_models": config.allowed_models,
            "created_at": config.created_at,
            "last_used_at": config.last_used_at,
        }

    def list_keys(self) -> list[dict]:
        """列出所有 Key（脱敏）"""
        return [self.get_key_info(k) for k in self.keys]

    def get_all_teams(self) -> list[str]:
        """获取所有团队名称"""
        return list(set(c.team for c in self.keys.values()))


# ─── 配额管理 ──────────────────────────────────────────────────

@dataclass
class QuotaConfig:
    """配额配置"""
    user_id: str
    # 每日限额
    daily_task_limit: int = 1000       # 每日任务数
    daily_token_limit: int = 100000    # 每日 token 数
    daily_cost_limit: float = 10.0     # 每日成本上限（美元）
    # 速率限制
    requests_per_minute: int = 60      # 每分钟请求数
    requests_per_hour: int = 1000      # 每小时请求数


class QuotaManager:
    """配额管理器（线程安全 + 内存计数器优化）"""

    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or get_config().cache_dir
        self.quotas_file = os.path.join(self.cache_dir, "quotas.json")
        self.usage_file = os.path.join(self.cache_dir, "quota_usage.jsonl")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.quotas: dict[str, QuotaConfig] = {}
        self._lock = threading.Lock()
        # 内存计数器：{user_id: {date: {tasks, tokens, cost}}}
        self._counters: dict[str, dict[str, dict]] = {}
        self._load_quotas()
        self._load_today_counters()

    def _load_quotas(self):
        if not os.path.exists(self.quotas_file):
            return
        try:
            with open(self.quotas_file) as f:
                data = json.load(f)
            for uid, config in data.items():
                self.quotas[uid] = QuotaConfig(**config)
        except (json.JSONDecodeError, TypeError):
            pass

    def _save_quotas(self):
        with self._lock:
            data = {uid: asdict(q) for uid, q in self.quotas.items()}
            with open(self.quotas_file, "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_today_counters(self):
        """启动时加载今日计数器（避免全表扫描）"""
        today = time.strftime("%Y-%m-%d")
        if not os.path.exists(self.usage_file):
            return
        try:
            with open(self.usage_file) as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if entry.get("date") == today:
                            uid = entry.get("user_id", "")
                            if uid:
                                if uid not in self._counters:
                                    self._counters[uid] = {}
                                if today not in self._counters[uid]:
                                    self._counters[uid][today] = {"tasks": 0, "tokens": 0, "cost": 0.0}
                                self._counters[uid][today]["tasks"] += 1
                                self._counters[uid][today]["tokens"] += entry.get("tokens", 0)
                                self._counters[uid][today]["cost"] += entry.get("cost", 0.0)
                    except (json.JSONDecodeError, KeyError):
                        continue
        except OSError:
            pass

    def _get_today_usage(self, user_id: str) -> dict:
        """获取今日使用量（从内存计数器）"""
        today = time.strftime("%Y-%m-%d")
        if user_id in self._counters and today in self._counters[user_id]:
            return self._counters[user_id][today]
        return {"tasks": 0, "tokens": 0, "cost": 0.0}

    def set_quota(self, user_id: str, **kwargs):
        """设置用户配额"""
        with self._lock:
            if user_id in self.quotas:
                for k, v in kwargs.items():
                    if hasattr(self.quotas[user_id], k):
                        setattr(self.quotas[user_id], k, v)
            else:
                self.quotas[user_id] = QuotaConfig(user_id=user_id, **kwargs)
        self._save_quotas()

    def get_quota(self, user_id: str) -> QuotaConfig:
        """获取用户配额"""
        return self.quotas.get(user_id, QuotaConfig(user_id=user_id))

    def check_quota(self, user_id: str) -> dict:
        """
        检查用户配额状态（使用内存计数器，O(1) 性能）。
        """
        quota = self.get_quota(user_id)
        usage = self._get_today_usage(user_id)

        daily_tasks = usage["tasks"]
        daily_tokens = usage["tokens"]
        daily_cost = usage["cost"]

        if daily_tasks >= quota.daily_task_limit:
            return {
                "allowed": False,
                "reason": f"每日任务数已达上限 ({quota.daily_task_limit})",
                "daily_tasks_used": daily_tasks,
                "daily_tasks_limit": quota.daily_task_limit,
                "daily_tokens_used": daily_tokens,
                "daily_tokens_limit": quota.daily_token_limit,
            }

        if daily_tokens >= quota.daily_token_limit:
            return {
                "allowed": False,
                "reason": f"每日 token 数已达上限 ({quota.daily_token_limit})",
                "daily_tasks_used": daily_tasks,
                "daily_tasks_limit": quota.daily_task_limit,
                "daily_tokens_used": daily_tokens,
                "daily_tokens_limit": quota.daily_token_limit,
            }

        if daily_cost >= quota.daily_cost_limit:
            return {
                "allowed": False,
                "reason": f"每日成本已达上限 (${quota.daily_cost_limit})",
                "daily_tasks_used": daily_tasks,
                "daily_tasks_limit": quota.daily_task_limit,
                "daily_tokens_used": daily_tokens,
                "daily_tokens_limit": quota.daily_token_limit,
            }

        return {
            "allowed": True,
            "reason": "配额充足",
            "daily_tasks_used": daily_tasks,
            "daily_tasks_limit": quota.daily_task_limit,
            "daily_tokens_used": daily_tokens,
            "daily_tokens_limit": quota.daily_token_limit,
        }

    def record_usage(self, user_id: str, tokens: int = 0, cost: float = 0.0,
                     action: str = ""):
        """记录使用量（同时更新内存计数器和磁盘）"""
        today = time.strftime("%Y-%m-%d")
        # 更新内存计数器
        with self._lock:
            if user_id not in self._counters:
                self._counters[user_id] = {}
            if today not in self._counters[user_id]:
                self._counters[user_id][today] = {"tasks": 0, "tokens": 0, "cost": 0.0}
            self._counters[user_id][today]["tasks"] += 1
            self._counters[user_id][today]["tokens"] += tokens
            self._counters[user_id][today]["cost"] += cost

        # 写入磁盘
        entry = {
            "user_id": user_id,
            "date": today,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "tokens": tokens,
            "cost": cost,
            "action": action[:100],
        }
        with self._lock:
            with open(self.usage_file, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def get_usage_summary(self, user_id: str) -> dict:
        """获取用户使用量摘要（含百分比）"""
        quota = self.get_quota(user_id)
        today = time.strftime("%Y-%m-%d")

        daily_tasks = 0
        daily_tokens = 0
        daily_cost = 0.0

        if os.path.exists(self.usage_file):
            with open(self.usage_file) as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if entry.get("user_id") == user_id and entry.get("date") == today:
                            daily_tasks += 1
                            daily_tokens += entry.get("tokens", 0)
                            daily_cost += entry.get("cost", 0.0)
                    except (json.JSONDecodeError, KeyError):
                        continue

        task_pct = (daily_tasks / quota.daily_task_limit * 100) if quota.daily_task_limit > 0 else 0
        token_pct = (daily_tokens / quota.daily_token_limit * 100) if quota.daily_token_limit > 0 else 0
        cost_pct = (daily_cost / quota.daily_cost_limit * 100) if quota.daily_cost_limit > 0 else 0

        return {
            "user_id": user_id,
            "daily_tasks": {"used": daily_tasks, "limit": quota.daily_task_limit, "pct": round(task_pct, 1)},
            "daily_tokens": {"used": daily_tokens, "limit": quota.daily_token_limit, "pct": round(token_pct, 1)},
            "daily_cost": {"used": round(daily_cost, 4), "limit": quota.daily_cost_limit, "pct": round(cost_pct, 1)},
        }

    def check_alerts(self, user_id: str, alert_threshold: float = 80.0) -> list[str]:
        """
        检查用量告警。

        返回告警消息列表。当使用量超过阈值（默认 80%）时触发告警。
        """
        summary = self.get_usage_summary(user_id)
        alerts = []

        for metric_name, metric_label in [
            ("daily_tasks", "任务数"),
            ("daily_tokens", "Token 数"),
            ("daily_cost", "成本"),
        ]:
            metric = summary[metric_name]
            if metric["pct"] >= 100:
                alerts.append(f"⛔ {user_id} {metric_label}已达上限: {metric['used']}/{metric['limit']} ({metric['pct']}%)")
            elif metric["pct"] >= alert_threshold:
                alerts.append(f"⚠️ {user_id} {metric_label}接近上限: {metric['used']}/{metric['limit']} ({metric['pct']}%)")

        return alerts

    def check_all_alerts(self, alert_threshold: float = 80.0) -> list[str]:
        """检查所有用户的用量告警"""
        all_alerts = []
        today = time.strftime("%Y-%m-%d")
        seen_users = set()

        if os.path.exists(self.usage_file):
            with open(self.usage_file) as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if entry.get("date") == today:
                            uid = entry.get("user_id", "")
                            if uid and uid not in seen_users:
                                seen_users.add(uid)
                    except (json.JSONDecodeError, KeyError):
                        continue

        for uid in seen_users:
            all_alerts.extend(self.check_alerts(uid, alert_threshold))

        # 也检查配置了配额但今天没使用的用户
        for uid in self.quotas:
            if uid not in seen_users:
                # 没有使用记录，不需要告警
                pass

        return all_alerts


# ─── 审计日志 ──────────────────────────────────────────────────

class AuditLogger:
    """
    审计日志记录器。

    使用方法:
        audit = AuditLogger()
        audit.log(AuditEvent(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            event_type="task_execution",
            action="翻译成中文",
            details={"input": "Hello", "output": "你好", "route": "local"},
            status="success",
            duration_ms=500,
        ))
    """

    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or get_config().cache_dir
        self.audit_file = os.path.join(self.cache_dir, "audit.jsonl")
        os.makedirs(self.cache_dir, exist_ok=True)

    def log(self, event: AuditEvent):
        """记录审计事件"""
        append_jsonl(self.audit_file, asdict(event))

    def query(self, event_type: str = None, user_id: str = None,
              start_time: str = None, end_time: str = None,
              limit: int = 100) -> list[dict]:
        """
        查询审计日志。

        参数:
            event_type: 事件类型过滤
            user_id: 用户过滤
            start_time: 开始时间 (ISO 格式)
            end_time: 结束时间 (ISO 格式)
            limit: 最大返回数

        返回:
            审计事件列表
        """
        all_entries = read_jsonl(self.audit_file)

        results = []
        for entry in all_entries:
            if event_type and entry.get("event_type") != event_type:
                continue
            if user_id and entry.get("user_id") != user_id:
                continue
            if start_time and entry.get("timestamp", "") < start_time:
                continue
            if end_time and entry.get("timestamp", "") > end_time:
                continue
            results.append(entry)

        return results[-limit:]

    def get_summary(self, days: int = 7) -> dict:
        """获取审计摘要"""
        start_time = time.strftime(
            "%Y-%m-%dT%H:%M:%S",
            time.localtime(time.time() - days * 86400)
        )
        events = self.query(start_time=start_time, limit=10000)

        summary = {
            "total_events": len(events),
            "by_type": defaultdict(int),
            "by_status": defaultdict(int),
            "by_user": defaultdict(int),
            "avg_duration_ms": 0,
        }

        total_duration = 0
        for e in events:
            summary["by_type"][e.get("event_type", "unknown")] += 1
            summary["by_status"][e.get("status", "unknown")] += 1
            summary["by_user"][e.get("user_id", "unknown")] += 1
            total_duration += e.get("duration_ms", 0)

        if events:
            summary["avg_duration_ms"] = round(total_duration / len(events))

        summary["by_type"] = dict(summary["by_type"])
        summary["by_status"] = dict(summary["by_status"])
        summary["by_user"] = dict(summary["by_user"])

        return summary


# ─── 单例实例 ──────────────────────────────────────────────────

_audit_logger = None
_quota_manager = None
_api_key_manager = None

def get_audit_logger() -> AuditLogger:
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger

def get_quota_manager() -> QuotaManager:
    global _quota_manager
    if _quota_manager is None:
        _quota_manager = QuotaManager()
    return _quota_manager

def get_api_key_manager() -> ApiKeyManager:
    global _api_key_manager
    if _api_key_manager is None:
        _api_key_manager = ApiKeyManager()
    return _api_key_manager
