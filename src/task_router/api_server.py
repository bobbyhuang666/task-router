"""
TaskRouter API 服务器 (异步版本)

提供 REST API 和 Web 仪表盘，用于：
- 任务路由 API（兼容 OpenAI 格式）
- 实时监控仪表盘
- 模型管理 API
- 批量任务处理

启动方式：
    python3 api_server.py                    # 默认端口 8930
    python3 api_server.py --port 9000        # 自定义端口
    python3 api_server.py --host 0.0.0.0     # 允许外部访问
"""

import os
import sys
import json
import time
import hmac
import asyncio
import threading
from datetime import datetime

from aiohttp import web

# 添加 scripts 目录到 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from task_router.logger import setup_logging
log = setup_logging()

from task_router.task_router import (
    run_task, Task, estimate, show_usage_stats,
    decompose_complex_task, CONFIG, cache,
    get_model_registry, run_batch,
)
from task_router.audit import get_audit_logger, get_api_key_manager, get_quota_manager

# API 认证密钥（环境变量设置，为空则不启用认证）
# 支持多 Key：设置 TASKROUTER_API_KEYS 环境变量（逗号分隔）
API_KEY = os.environ.get("TASKROUTER_API_KEY", "")
API_KEYS_MULTI = [k.strip() for k in os.environ.get("TASKROUTER_API_KEYS", "").split(",") if k.strip()]
PUBLIC_PATHS = {"/", "/api/health"}

# 输入大小限制
MAX_INPUT_LENGTH = int(os.environ.get("TASKROUTER_MAX_INPUT", "100000"))  # 100K 字符
MAX_REQUEST_SIZE = int(os.environ.get("TASKROUTER_MAX_REQUEST_SIZE", str(2 * 1024 * 1024)))  # 2MB

# 速率限制（每分钟请求数，0=不限制）
RATE_LIMIT_RPM = int(os.environ.get("TASKROUTER_RATE_LIMIT_RPM", "60"))


class TokenBucketRateLimiter:
    """令牌桶速率限制器 — 按 API key / IP 限制请求频率"""

    def __init__(self, rpm: int):
        self.rpm = rpm
        self._buckets: dict[str, dict] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        """检查是否允许请求。返回 True=放行，False=限速"""
        if self.rpm <= 0:
            return True
        now = time.time()
        with self._lock:
            # 每 1000 次请求清理一次过期桶
            self._request_count = getattr(self, '_request_count', 0) + 1
            if self._request_count >= 1000:
                self._request_count = 0
                cutoff = now - 300
                stale = [k for k, v in self._buckets.items() if v["last"] < cutoff]
                for k in stale:
                    del self._buckets[k]

            bucket = self._buckets.get(key)
            if not bucket:
                self._buckets[key] = {"tokens": self.rpm - 1, "last": now}
                return True
            # 补充令牌
            elapsed = now - bucket["last"]
            bucket["tokens"] = min(self.rpm, bucket["tokens"] + elapsed * (self.rpm / 60.0))
            bucket["last"] = now
            if bucket["tokens"] >= 1:
                bucket["tokens"] -= 1
                return True
            return False


# ─── API 处理器 ──────────────────────────────────────────────────

class TaskRouterHandler:
    """异步 HTTP API 处理器"""

    def __init__(self):
        self.app = web.Application(client_max_size=MAX_REQUEST_SIZE)
        self.rate_limiter = TokenBucketRateLimiter(RATE_LIMIT_RPM)
        self._setup_routes()

    def _setup_routes(self):
        """设置路由"""
        # API 路由
        self.app.router.add_post("/api/task", self._api_task)
        self.app.router.add_post("/api/estimate", self._api_estimate)
        self.app.router.add_post("/api/decompose", self._api_decompose)
        self.app.router.add_post("/api/batch", self._api_batch)
        self.app.router.add_post("/api/benchmark", self._api_benchmark)

        # OpenAI 兼容路由
        self.app.router.add_post("/v1/chat/completions", self._api_openai_chat)

        # GET 路由
        self.app.router.add_get("/api/stats", self._api_stats)
        self.app.router.add_get("/api/models", self._api_models)
        self.app.router.add_get("/api/cache", self._api_cache_stats)
        self.app.router.add_get("/api/health", self._api_health)
        self.app.router.add_get("/api/history", self._api_history)
        self.app.router.add_get("/api/audit", self._api_audit)
        self.app.router.add_get("/api/audit/summary", self._api_audit_summary)

        # 多 Key 管理路由
        self.app.router.add_get("/api/keys", self._api_list_keys)
        self.app.router.add_post("/api/keys", self._api_add_key)
        self.app.router.add_delete("/api/keys/{key}", self._api_remove_key)

        # 用量告警路由
        self.app.router.add_get("/api/quota", self._api_quota_status)
        self.app.router.add_get("/api/quota/alerts", self._api_quota_alerts)

        # 仪表盘
        self.app.router.add_get("/", self._serve_dashboard)

        # 中间件（顺序：先认证，再速率限制，再 CORS）
        self.app.middlewares.append(self._auth_middleware)
        self.app.middlewares.append(self._rate_limit_middleware)
        self.app.middlewares.append(self._cors_middleware)

    @staticmethod
    def _safe_eq(a: str, b: str) -> bool:
        """常量时间比较，防止时序攻击"""
        return hmac.compare_digest(a.encode(), b.encode()) if a and b else False

    @staticmethod
    async def _parse_json(request: web.Request) -> dict:
        """安全解析 JSON 请求体，含大小验证"""
        try:
            data = await request.json()
        except (json.JSONDecodeError, TypeError):
            return {}
        if not isinstance(data, dict):
            return {}
        # 验证输入字段大小
        for field in ("action", "prompt", "text", "input"):
            val = data.get(field, "")
            if isinstance(val, str) and len(val) > MAX_INPUT_LENGTH:
                raise web.HTTPRequestEntityTooLarge(
                    max_size=MAX_REQUEST_SIZE,
                    actual_size=len(val),
                )
        return data

    @web.middleware
    async def _auth_middleware(self, request: web.Request, handler):
        """API 认证中间件（Bearer token 或 X-API-Key）"""
        if request.path in PUBLIC_PATHS:
            return await handler(request)

        # 提取请求中的 Key（仅支持 Header，不支持 query string 防止日志泄露）
        provided_key = ""
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            provided_key = auth_header[7:]
        elif request.headers.get("X-API-Key"):
            provided_key = request.headers.get("X-API-Key")

        if not provided_key:
            # 没有提供 Key，检查是否启用认证
            if not API_KEY and not API_KEYS_MULTI and not get_api_key_manager().keys:
                return await handler(request)
            log.warning("认证失败: %s %s (来源: %s, 未提供 Key)", request.method, request.path, request.remote)
            return web.json_response(
                {"error": "未授权：请提供有效的 API Key（Bearer token 或 X-API-Key header）"},
                status=401,
            )

        # 优先检查 ApiKeyManager（多 Key 模式）
        ak_manager = get_api_key_manager()
        key_config = ak_manager.authenticate(provided_key)
        if key_config:
            # 将 Key 信息注入请求
            request["api_key_config"] = key_config
            request["team"] = key_config.team
            return await handler(request)

        # 兼容单 Key 模式（环境变量）
        if API_KEY and self._safe_eq(provided_key, API_KEY):
            request["api_key_config"] = None
            request["team"] = "system"
            return await handler(request)

        # 兼容多 Key 环境变量
        if API_KEYS_MULTI:
            for k in API_KEYS_MULTI:
                if self._safe_eq(provided_key, k):
                    request["api_key_config"] = None
                    request["team"] = "env"
                    return await handler(request)

        log.warning("认证失败: %s %s (来源: %s, Key: %s***)", request.method, request.path, request.remote, provided_key[:4])
        return web.json_response(
            {"error": "未授权：请提供有效的 API Key（Bearer token 或 X-API-Key header）"},
            status=401,
        )

    @web.middleware
    async def _rate_limit_middleware(self, request: web.Request, handler):
        """速率限制中间件 — 按 API key 或 IP 限速"""
        if request.path in PUBLIC_PATHS:
            return await handler(request)
        # 优先用 API key，其次用 IP
        rate_key = request.get("api_key_config")
        if rate_key and hasattr(rate_key, "key"):
            rate_key = rate_key.key[:4]
        else:
            rate_key = request.remote or "unknown"
        if not self.rate_limiter.allow(str(rate_key)):
            return web.json_response(
                {"error": f"请求过于频繁，限制 {RATE_LIMIT_RPM} 次/分钟"},
                status=429,
            )
        return await handler(request)

    @web.middleware
    async def _cors_middleware(self, request: web.Request, handler):
        """CORS 中间件"""
        if request.method == "OPTIONS":
            response = web.Response()
        else:
            try:
                response = await handler(request)
            except web.HTTPException as ex:
                response = ex
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-API-Key"
        return response

    # ─── API 端点 ─────────────────────────────────────────────

    async def _api_task(self, request: web.Request) -> web.Response:
        """执行单个任务（含配额检查 + 模型访问控制）"""
        body = await self._parse_json(request)
        action = body.get("action", body.get("prompt", ""))
        text = body.get("text", body.get("input", ""))
        force = body.get("force", "")
        files = body.get("files", [])

        if not action:
            return web.json_response({"error": "缺少 action 参数"}, status=400)

        # 配额检查
        team = getattr(request, "team", "") or "default"
        qm = get_quota_manager()
        quota = qm.check_quota(team)
        if not quota["allowed"]:
            return web.json_response({"error": quota["reason"], "quota": quota}, status=429)

        task = Task(action=action, text=text, files=files)
        start = time.time()
        result = await asyncio.to_thread(run_task, task, force_route=force)
        elapsed = int((time.time() - start) * 1000)

        # 模型访问控制
        key_config = request.get("api_key_config")
        if key_config:
            akm = get_api_key_manager()
            if not akm.check_model_access(key_config.key, result.model_used):
                return web.json_response({"error": f"无权使用模型: {result.model_used}"}, status=403)

        # 记录用量
        total_tokens = result.tokens_input + result.tokens_output
        qm.record_usage(team, tokens=total_tokens, cost=result.cost_saved, action=action[:50])

        log.info("任务完成: route=%s model=%s time=%dms action=%s",
                 result.route, result.model_used, elapsed, action[:50])

        return web.json_response({
            "output": result.output,
            "route": result.route,
            "model": result.model_used,
            "tokens_input": result.tokens_input,
            "tokens_output": result.tokens_output,
            "time_ms": elapsed,
            "cost_saved": result.cost_saved,
        })

    async def _api_estimate(self, request: web.Request) -> web.Response:
        """估算任务路由"""
        body = await self._parse_json(request)
        action = body.get("action", body.get("prompt", ""))
        if not action:
            return web.json_response({"error": "缺少 action 参数"}, status=400)
        result = estimate(action)
        return web.json_response(result)

    async def _api_decompose(self, request: web.Request) -> web.Response:
        """拆解复杂任务"""
        body = await self._parse_json(request)
        action = body.get("action", body.get("prompt", ""))
        text = body.get("text", "")
        if not action:
            return web.json_response({"error": "缺少 action 参数"}, status=400)
        subtasks = decompose_complex_task(action, text)
        if not subtasks:
            # 尝试递归拆解
            from task_router import _recursive_decompose
            subtasks = _recursive_decompose(action, text) or []
        return web.json_response({"task": action, "subtasks": subtasks})

    MAX_BATCH_SIZE = 100

    async def _api_batch(self, request: web.Request) -> web.Response:
        """批量任务处理（含批量限制 + 配额检查）"""
        body = await self._parse_json(request)
        tasks = body.get("tasks", [])
        concurrency = body.get("concurrency", 1)
        if not tasks:
            return web.json_response({"error": "缺少 tasks 数组"}, status=400)

        # 批量大小限制
        if len(tasks) > self.MAX_BATCH_SIZE:
            return web.json_response(
                {"error": f"批量上限 {self.MAX_BATCH_SIZE} 个任务，当前 {len(tasks)} 个"},
                status=400,
            )

        # 配额检查
        team = getattr(request, "team", "") or "default"
        qm = get_quota_manager()
        quota = qm.check_quota(team)
        if not quota["allowed"]:
            return web.json_response({"error": quota["reason"], "quota": quota}, status=429)

        start = time.time()
        results = await asyncio.to_thread(run_batch, tasks, concurrency=concurrency)
        elapsed = int((time.time() - start) * 1000)

        output = []
        total_tokens = 0
        for r in results:
            output.append({
                "action": r.action[:100],
                "output": r.output,
                "route": r.route,
                "model": r.model_used,
                "time_ms": r.time_ms,
                "cost_saved": r.cost_saved,
            })
            total_tokens += r.tokens_input + r.tokens_output

        # 记录用量
        qm.record_usage(team, tokens=total_tokens, action=f"batch({len(tasks)})")

        return web.json_response({
            "results": output,
            "count": len(output),
            "total_time_ms": elapsed,
            "concurrency": concurrency,
        })

    async def _api_stats(self, request: web.Request) -> web.Response:
        """使用统计"""
        stats_text = show_usage_stats()
        return web.json_response({"stats": stats_text})

    async def _api_models(self, request: web.Request) -> web.Response:
        """模型列表"""
        registry = get_model_registry()
        registry.discover()
        return web.json_response(registry.get_status())

    async def _api_cache_stats(self, request: web.Request) -> web.Response:
        """缓存统计"""
        stats = cache.stats()
        return web.json_response(stats)

    async def _api_health(self, request: web.Request) -> web.Response:
        """健康检查"""
        return web.json_response({
            "status": "healthy",
            "version": "6.0.0",
            "timestamp": datetime.now().isoformat(),
        })

    async def _api_history(self, request: web.Request) -> web.Response:
        """最近任务历史"""
        log_file = os.path.join(CONFIG.cache_dir, "usage.jsonl")
        limit = int(request.query.get("limit", "20"))
        entries = []
        if os.path.exists(log_file):
            with open(log_file) as f:
                lines = f.readlines()
                for line in lines[-limit:]:
                    try:
                        entries.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        pass
        return web.json_response({"history": entries, "count": len(entries)})

    async def _api_benchmark(self, request: web.Request) -> web.Response:
        """运行基准测试"""
        body = await self._parse_json(request)
        model = body.get("model")
        registry = get_model_registry()
        registry.discover()
        results = await asyncio.to_thread(registry.run_benchmark, model)
        return web.json_response({"results": results})

    async def _api_audit(self, request: web.Request) -> web.Response:
        """查询审计日志"""
        audit = get_audit_logger()
        limit = int(request.query.get("limit", "50"))
        event_type = request.query.get("type")
        events = audit.query(event_type=event_type, limit=limit)
        return web.json_response({"events": events, "count": len(events)})

    async def _api_audit_summary(self, request: web.Request) -> web.Response:
        """审计摘要"""
        audit = get_audit_logger()
        days = int(request.query.get("days", "7"))
        summary = audit.get_summary(days=days)
        return web.json_response(summary)

    # ── 多 Key 管理 ──

    async def _api_list_keys(self, request: web.Request) -> web.Response:
        """列出所有 API Key（脱敏）"""
        ak_manager = get_api_key_manager()
        return web.json_response({"keys": ak_manager.list_keys()})

    @staticmethod
    def _require_admin(request: web.Request) -> web.Response | None:
        """检查管理员权限，无权限返回 403 响应，有权限返回 None"""
        key_config = request.get("api_key_config")
        if key_config and key_config.role != "admin":
            return web.json_response({"error": "需要管理员权限"}, status=403)
        # 兼容单 Key 模式：环境变量 Key 拥有管理员权限
        return None

    async def _api_add_key(self, request: web.Request) -> web.Response:
        """添加新的 API Key（需要管理员权限）"""
        deny = self._require_admin(request)
        if deny:
            return deny

        body = await self._parse_json(request)
        key = body.get("key", "")
        team = body.get("team", "default")
        if not key:
            return web.json_response({"error": "key is required"}, status=400)

        ak_manager = get_api_key_manager()
        ak_manager.add_key(
            key=key,
            team=team,
            description=body.get("description", ""),
            role=body.get("role", "user"),
            monthly_task_limit=body.get("monthly_task_limit", 10000),
            monthly_token_limit=body.get("monthly_token_limit", 1000000),
            monthly_cost_limit=body.get("monthly_cost_limit", 100.0),
            allowed_models=body.get("allowed_models", []),
        )
        return web.json_response({"status": "created", "key_prefix": key[:4] + "****", "team": team})

    async def _api_remove_key(self, request: web.Request) -> web.Response:
        """删除 API Key（需要管理员权限）"""
        deny = self._require_admin(request)
        if deny:
            return deny

        key = request.match_info["key"]
        ak_manager = get_api_key_manager()
        if ak_manager.remove_key(key):
            return web.json_response({"status": "deleted"})
        return web.json_response({"error": "key not found"}, status=404)

    # ── 用量告警 ──

    async def _api_quota_status(self, request: web.Request) -> web.Response:
        """查看配额状态"""
        user_id = request.query.get("user_id", "")
        team = getattr(request, "team", "") or ""
        if not user_id:
            user_id = team or "default"

        qm = get_quota_manager()
        summary = qm.get_usage_summary(user_id)
        return web.json_response(summary)

    async def _api_quota_alerts(self, request: web.Request) -> web.Response:
        """查看用量告警"""
        qm = get_quota_manager()
        threshold = float(request.query.get("threshold", "80"))
        alerts = qm.check_all_alerts(alert_threshold=threshold)
        return web.json_response({"alerts": alerts, "count": len(alerts)})

    async def _api_openai_chat(self, request: web.Request) -> web.Response:
        """OpenAI 兼容的 Chat Completions API"""
        body = await self._parse_json(request)
        messages = body.get("messages", [])
        if not messages:
            return web.json_response({
                "error": {
                    "message": "messages is required",
                    "type": "invalid_request_error",
                    "code": "missing_parameter",
                }
            }, status=400)

        # 验证 messages 数组大小和总内容长度
        if len(messages) > 500:
            return web.json_response({
                "error": {"message": "messages 数组超过最大长度 (500)", "type": "invalid_request_error"}
            }, status=400)
        total_content_len = sum(len(str(m.get("content", ""))) for m in messages)
        if total_content_len > MAX_INPUT_LENGTH:
            raise web.HTTPRequestEntityTooLarge(max_size=MAX_REQUEST_SIZE, actual_size=total_content_len)

        # 从 messages 中提取 action 和 text
        # 支持多种格式：
        # 1. 单条 user 消息: "翻译成中文：Hello" 或 "帮我翻译 Hello"
        # 2. system + user: system 作为任务指令，user 作为内容
        # 3. 多轮对话: 取 system + 最后一条 user 消息
        action = ""
        text = ""
        system_msg = ""
        user_messages = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                system_msg = content
            elif role == "user":
                user_messages.append(content)

        if system_msg:
            # 有 system 消息时，system 作为 action，user 作为 text
            action = system_msg
            text = "\n".join(user_messages) if user_messages else ""
        elif len(user_messages) == 1:
            # 单条 user 消息，尝试分离 action 和 text
            content = user_messages[0]
            # 尝试用冒号分隔
            for sep in ["：", ":"]:
                if sep in content:
                    parts = content.split(sep, 1)
                    # 检查第一部分是否像任务指令（短且包含动词）
                    potential_action = parts[0].strip()
                    if len(potential_action) < 30:
                        action = potential_action
                        text = parts[1].strip()
                        break
            if not action:
                # 无法分隔，整个内容作为 action
                action = content
        else:
            # 多条 user 消息，取最后一条作为 text
            action = user_messages[0] if user_messages else ""
            text = "\n".join(user_messages[1:]) if len(user_messages) > 1 else ""

        if not action:
            action = text
            text = ""

        # 配额检查
        team = getattr(request, "team", "") or "default"
        qm = get_quota_manager()
        quota = qm.check_quota(team)
        if not quota["allowed"]:
            return web.json_response({
                "error": {"message": quota["reason"], "type": "quota_exceeded", "code": "quota_exceeded"}
            }, status=429)

        task = Task(action=action, text=text)
        start = time.time()
        result = await asyncio.to_thread(run_task, task)
        elapsed = time.time() - start

        # 模型访问控制
        key_config = request.get("api_key_config")
        if key_config:
            akm = get_api_key_manager()
            if not akm.check_model_access(key_config.key, result.model_used):
                return web.json_response({
                    "error": {"message": f"无权使用模型: {result.model_used}", "type": "permission_denied"}
                }, status=403)

        # 记录用量
        total_tokens = result.tokens_input + result.tokens_output
        qm.record_usage(team, tokens=total_tokens, action=f"openai:{action[:50]}")

        log.info("OpenAI API: route=%s model=%s time=%.0fms", result.route, result.model_used, elapsed * 1000)

        response = {
            "id": f"chatcmpl-{int(time.time()*1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", result.model_used or "task-router"),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result.output,
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": result.tokens_input,
                "completion_tokens": result.tokens_output,
                "total_tokens": result.tokens_input + result.tokens_output,
            },
            "system_fingerprint": f"fp_{result.route}",
        }

        return web.json_response(response)

    # ─── 仪表盘 ──────────────────────────────────────────────

    async def _serve_dashboard(self, request: web.Request) -> web.Response:
        """主仪表盘页面"""
        return web.Response(text=DASHBOARD_HTML, content_type="text/html", charset="utf-8")


# ─── 仪表盘 HTML ────────────────────────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TaskRouter Dashboard</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; }
.header { background: linear-gradient(135deg, #1e293b 0%, #334155 100%); padding: 20px 30px; border-bottom: 1px solid #475569; }
.header h1 { font-size: 24px; color: #f8fafc; }
.header p { color: #94a3b8; margin-top: 4px; }
.container { max-width: 1200px; margin: 0 auto; padding: 20px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 16px; margin-bottom: 20px; }
.card { background: #1e293b; border-radius: 12px; padding: 20px; border: 1px solid #334155; }
.card h3 { color: #94a3b8; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; }
.card .value { font-size: 32px; font-weight: bold; color: #f8fafc; margin-top: 8px; }
.card .sub { color: #64748b; font-size: 13px; margin-top: 4px; }
.green { color: #4ade80; }
.blue { color: #60a5fa; }
.yellow { color: #facc15; }
.red { color: #f87171; }
.test-section { background: #1e293b; border-radius: 12px; padding: 24px; border: 1px solid #334155; margin-bottom: 20px; }
.test-section h2 { margin-bottom: 16px; font-size: 18px; }
.input-group { display: flex; gap: 10px; margin-bottom: 12px; }
input, textarea, select { background: #0f172a; border: 1px solid #475569; color: #e2e8f0; padding: 10px 14px; border-radius: 8px; font-size: 14px; flex: 1; }
textarea { min-height: 80px; resize: vertical; }
button { background: #3b82f6; color: white; border: none; padding: 10px 20px; border-radius: 8px; cursor: pointer; font-size: 14px; font-weight: 500; }
button:hover { background: #2563eb; }
button.secondary { background: #475569; }
button.secondary:hover { background: #64748b; }
.result { background: #0f172a; border-radius: 8px; padding: 16px; margin-top: 12px; white-space: pre-wrap; font-family: monospace; font-size: 13px; display: none; }
.route-badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; }
.route-local { background: #065f46; color: #6ee7b7; }
.route-cloud { background: #7c2d12; color: #fdba74; }
.route-cache { background: #1e3a5f; color: #93c5fd; }
.route-hybrid { background: #4c1d95; color: #c4b5fd; }
.model-table { width: 100%; border-collapse: collapse; margin-top: 12px; }
.model-table th, .model-table td { padding: 10px; text-align: left; border-bottom: 1px solid #334155; }
.model-table th { color: #94a3b8; font-size: 12px; text-transform: uppercase; }
.bar { display: inline-block; height: 16px; background: #3b82f6; border-radius: 3px; vertical-align: middle; }
</style>
</head>
<body>
<div class="header">
  <h1>TaskRouter Dashboard</h1>
  <p>自进化 AI 成本优化引擎 — A3M 多信号路由 + 蒸馏学习闭环</p>
</div>
<div class="container">
  <div class="grid" id="stats-grid">
    <div class="card"><h3>本地调用</h3><div class="value green" id="stat-local">-</div><div class="sub">免费执行</div></div>
    <div class="card"><h3>云端调用</h3><div class="value blue" id="stat-cloud">-</div><div class="sub">付费执行</div></div>
    <div class="card"><h3>缓存命中</h3><div class="value yellow" id="stat-cache">-</div><div class="sub">0 token 消耗</div></div>
    <div class="card"><h3>累计节约</h3><div class="value green" id="stat-saved">-</div><div class="sub">相比全云端</div></div>
  </div>

  <div class="test-section">
    <h2>测试任务路由</h2>
    <div class="input-group">
      <input type="text" id="task-action" placeholder="任务描述，如：翻译成中文、分类、提取关键词">
    </div>
    <div class="input-group">
      <textarea id="task-text" placeholder="待处理文本（可选）"></textarea>
    </div>
    <div class="input-group">
      <select id="task-force">
        <option value="">自动路由</option>
        <option value="local">强制本地</option>
        <option value="cloud">强制云端</option>
      </select>
      <button onclick="runTask()">执行任务</button>
      <button class="secondary" onclick="estimateTask()">预估路由</button>
    </div>
    <div class="result" id="task-result"></div>
  </div>

  <div class="test-section">
    <h2>模型管理</h2>
    <button onclick="loadModels()" style="margin-bottom: 12px">刷新模型列表</button>
    <table class="model-table" id="models-table">
      <thead><tr><th>模型</th><th>参数</th><th>延迟</th><th>速度</th><th>调用</th><th>成功率</th></tr></thead>
      <tbody id="models-body"></tbody>
    </table>
  </div>
</div>

<script>
const API = '';

async function api(method, path, body) {
  const opts = { method, headers: {'Content-Type': 'application/json'} };
  if (body) opts.body = JSON.stringify(body);
  const resp = await fetch(API + path, opts);
  return resp.json();
}

async function loadStats() {
  const data = await api('GET', '/api/stats');
  const stats = data.stats || '';
  const local = stats.match(/本地调用: (\\d+)/);
  const cloud = stats.match(/云端调用: (\\d+)/);
  const cache = stats.match(/缓存命中: (\\d+)/);
  const saved = stats.match(/累计节约成本: \\$([\\d.]+)/);
  if (local) document.getElementById('stat-local').textContent = local[1] + ' 次';
  if (cloud) document.getElementById('stat-cloud').textContent = cloud[1] + ' 次';
  if (cache) document.getElementById('stat-cache').textContent = cache[1] + ' 次';
  if (saved) document.getElementById('stat-saved').textContent = '$' + saved[1];
}

async function runTask() {
  const action = document.getElementById('task-action').value;
  const text = document.getElementById('task-text').value;
  const force = document.getElementById('task-force').value;
  if (!action) return alert('请输入任务描述');

  const result = document.getElementById('task-result');
  result.style.display = 'block';
  result.textContent = '执行中...';

  const data = await api('POST', '/api/task', { action, text, force });
  const routeClass = data.route?.includes('cache') ? 'route-cache' :
                     data.route?.includes('local') ? 'route-local' :
                     data.route?.includes('hybrid') ? 'route-hybrid' : 'route-cloud';

  result.innerHTML =
    `<span class="route-badge ${routeClass}">${data.route || 'unknown'}</span> ` +
    `模型: ${data.model || '-'} | 耗时: ${data.time_ms || 0}ms | ` +
    `输入: ${data.tokens_input || 0} tokens | 输出: ${data.tokens_output || 0} tokens` +
    (data.cost_saved > 0 ? ` | <span class="green">节约: $${data.cost_saved}</span>` : '') +
    `\\n\\n${data.output || data.error || '无输出'}`;

  loadStats();
}

async function estimateTask() {
  const action = document.getElementById('task-action').value;
  if (!action) return alert('请输入任务描述');

  const result = document.getElementById('task-result');
  result.style.display = 'block';
  result.textContent = '估算中...';

  const data = await api('POST', '/api/estimate', { action });
  const tag = data.will_save ? '本地 (免费)' : '云端 (付费)';
  const cls = data.will_save ? 'route-local' : 'route-cloud';
  result.innerHTML =
    `<span class="route-badge ${cls}">${tag}</span> 评分: ${data.score || '-'}\\n` +
    `原因: ${data.reason || '-'}\\n预估云端成本: ${data.estimated_cloud_cost || '-'}`;
}

async function loadModels() {
  const data = await api('GET', '/api/models');
  const tbody = document.getElementById('models-body');
  tbody.innerHTML = '';
  for (const [name, info] of Object.entries(data)) {
    const caps = Object.entries(info.capabilities || {})
      .filter(([k,v]) => v > 0)
      .map(([k,v]) => `${k}:${(v*100).toFixed(0)}%`)
      .join(', ');
    const latency = info.avg_latency_ms > 0 ? info.avg_latency_ms.toFixed(0) + 'ms' : '-';
    const speed = info.tokens_per_second > 0 ? info.tokens_per_second.toFixed(0) + 't/s' : '-';
    const rate = (info.success_rate * 100).toFixed(0) + '%';
    tbody.innerHTML += `<tr>
      <td>${name}</td>
      <td>${info.parameter_size || '-'}</td>
      <td>${latency}</td>
      <td>${speed}</td>
      <td>${info.total_calls}</td>
      <td>${rate}</td>
    </tr>`;
  }
}

// 初始加载
loadStats();
loadModels();
setInterval(loadStats, 10000);
</script>
</body>
</html>"""


# ─── 服务器启动 ──────────────────────────────────────────────────

def start_server(host="127.0.0.1", port=8930):
    """启动 API 服务器"""
    handler = TaskRouterHandler()
    app = handler.app

    has_key = bool(API_KEY or API_KEYS_MULTI or get_api_key_manager().keys)
    auth_status = "已启用" if has_key else "未设置（开放访问）"
    log.info("TaskRouter API v6.0.0 启动: %s:%d, 认证: %s", host, port, auth_status)

    if not has_key:
        log.warning("⚠️ 未配置 API Key，所有接口无需认证即可访问。生产环境请设置 TASKROUTER_API_KEY 环境变量。")

    print("TaskRouter API 服务器启动 (异步)")
    print(f"  地址: http://{host}:{port}")
    print(f"  仪表盘: http://{host}:{port}/")
    print(f"  认证: {auth_status}")
    if not has_key:
        print("  ⚠️ 警告: 未配置 API Key，所有接口无需认证即可访问！")
    print("  API 文档:")
    print("    POST /api/task              - 执行任务")
    print("    POST /api/estimate          - 预估路由")
    print("    POST /api/decompose         - 拆解任务")
    print("    POST /api/batch             - 批量处理")
    print("    POST /v1/chat/completions   - OpenAI 兼容 API")
    print("    GET  /api/stats             - 使用统计")
    print("    GET  /api/models            - 模型列表")
    print("    GET  /api/health            - 健康检查")
    print("    GET  /api/history           - 任务历史")
    print()

    web.run_app(app, host=host, port=port, print=None)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TaskRouter API 服务器")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8930, help="监听端口")
    args = parser.parse_args()
    start_server(args.host, args.port)
