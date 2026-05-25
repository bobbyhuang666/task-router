"""
模型调用 — Ollama 本地 + 云端 API，含熔断器和重试
"""

import time
import threading
from typing import Any, Optional

from task_router.config import get_config

# 延迟加载的单例（线程安全）
_model_registry_instance = None
_model_registry_lock = threading.Lock()


# ─── 熔断器（线程安全）─────────────────────────────────────────

class CircuitBreaker:
    """云端 API 熔断器（支持 CLOSED → OPEN → HALF_OPEN → CLOSED）"""

    STATE_CLOSED = "closed"
    STATE_OPEN = "open"
    STATE_HALF_OPEN = "half_open"

    def __init__(self, max_failures: int = 3, cooldown_seconds: int = 120, half_open_max: int = 1):
        self._lock = threading.Lock()
        self.failures: int = 0
        self.last_failure: float = 0.0
        self.open_until: float = 0.0
        self.state: str = self.STATE_CLOSED
        self.max_failures = max_failures
        self.cooldown_seconds = cooldown_seconds
        self.half_open_max = half_open_max
        self.half_open_attempts: int = 0

    def is_open(self) -> bool:
        with self._lock:
            if self.state == self.STATE_OPEN:
                if time.time() >= self.open_until:
                    self.state = self.STATE_HALF_OPEN
                    self.half_open_attempts = 0
                    return False
                return True
            return False

    def allow_request(self) -> bool:
        """判断是否允许请求通过（OPEN 状态冷却后进入 HALF_OPEN 放行一次探测）"""
        with self._lock:
            if self.state == self.STATE_CLOSED:
                return True
            if self.state == self.STATE_OPEN:
                if time.time() >= self.open_until:
                    self.state = self.STATE_HALF_OPEN
                    self.half_open_attempts = 0
                    return True
                return False
            if self.state == self.STATE_HALF_OPEN:
                return self.half_open_attempts < self.half_open_max
            return True

    def record_success(self) -> None:
        with self._lock:
            self.state = self.STATE_CLOSED
            self.failures = 0
            self.open_until = 0.0
            self.half_open_attempts = 0

    def record_failure(self) -> None:
        with self._lock:
            if self.state == self.STATE_HALF_OPEN:
                self.state = self.STATE_OPEN
                self.open_until = time.time() + self.cooldown_seconds
                self.half_open_attempts = 0
                return
            self.failures += 1
            self.last_failure = time.time()
            if self.failures >= self.max_failures:
                self.state = self.STATE_OPEN
                self.open_until = time.time() + self.cooldown_seconds

    def remaining_seconds(self) -> int:
        with self._lock:
            return max(0, int(self.open_until - time.time()))

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self.state,
                "failures": self.failures,
                "last_failure": self.last_failure,
                "open_until": self.open_until,
                "max_failures": self.max_failures,
                "cooldown_seconds": self.cooldown_seconds,
                "half_open_attempts": self.half_open_attempts,
            }


# 全局熔断器实例
circuit_breaker = CircuitBreaker()


def _get_model_registry() -> Any:
    """延迟加载模型注册表（线程安全单例）"""
    global _model_registry_instance
    if _model_registry_instance is not None:
        return _model_registry_instance
    with _model_registry_lock:
        if _model_registry_instance is not None:
            return _model_registry_instance
        try:
            from task_router.model_registry import ModelRegistry
            config = get_config()
            _model_registry_instance = ModelRegistry(cache_dir=config.cache_dir)
        except ImportError:
            pass
    return _model_registry_instance


# ─── Ollama 调用 ──────────────────────────────────────────────

def call_ollama(prompt: str, model: Optional[str] = None, max_tokens: Optional[int] = None,
                with_logprobs: bool = False) -> dict[str, Any]:
    """调用 Ollama 本地模型（可选 logprobs 用于置信度提取）"""
    import requests
    config = get_config()
    model = model or config.local_model
    max_tokens = max_tokens or config.local_max_tokens
    start = time.time()

    try:
        body = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        if with_logprobs:
            body["logprobs"] = True

        resp = requests.post(
            f"{config.ollama_base}/api/generate",
            json=body,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        elapsed = int((time.time() - start) * 1000)

        result = {
            "text": data["response"].strip(),
            "tokens_input": data.get("prompt_eval_count", 0),
            "tokens_output": data.get("eval_count", 0),
            "time_ms": elapsed,
        }

        # 提取 logprobs（如果请求了）
        if with_logprobs:
            result["logprobs"] = data.get("logprobs", [])

        # 更新模型统计
        registry = _get_model_registry()
        if registry:
            registry.update_after_call(
                model, success=True, latency_ms=elapsed,
                tokens_in=result["tokens_input"], tokens_out=result["tokens_output"]
            )

        return result
    except Exception:
        registry = _get_model_registry()
        if registry:
            registry.update_after_call(model, success=False, latency_ms=0)
        raise


# ─── 云端 API 调用 ──────────────────────────────────────────────

def call_cloud_api(prompt: str, text: str = "") -> dict[str, Any]:
    """调用云端 API（含 PII 脱敏、重试、熔断）"""
    config = get_config()

    if not config.cloud_api_key:
        return {
            "text": "[云端未配置] 设置 CLOUD_API_KEY 环境变量启用云端路由",
            "tokens_input": 0, "tokens_output": 0, "time_ms": 0,
        }

    # 熔断检查（支持半开状态探测）
    if not circuit_breaker.allow_request():
        remaining = circuit_breaker.remaining_seconds()
        return {
            "text": f"[云端熔断中] 连续失败{circuit_breaker.failures}次，{remaining}秒后重试",
            "tokens_input": 0, "tokens_output": 0, "time_ms": 0,
            "circuit_open": True,
        }
    if circuit_breaker.state == CircuitBreaker.STATE_HALF_OPEN:
        with circuit_breaker._lock:
            circuit_breaker.half_open_attempts += 1

    import requests

    # PII 脱敏
    pf = _get_privacy_filter()
    anon_result = None
    if pf:
        anon_result = pf.anonymize(prompt)
        prompt = anon_result.text
        if text:
            text_anon = pf.anonymize(text)
            text = text_anon.text
            anon_result.anonymized_count += text_anon.anonymized_count

    if text:
        full_content = f"{prompt}\n\n内容：\n{text}"
    else:
        full_content = prompt
    messages = [{"role": "user", "content": full_content}]

    max_retries = 2
    last_error: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            start = time.time()
            resp = requests.post(
                f"{config.cloud_api_url}/v1/chat/completions",
                headers={"Authorization": f"Bearer {config.cloud_api_key}"},
                json={
                    "model": config.cloud_model,
                    "messages": messages,
                    "max_tokens": 4096,
                },
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            elapsed = int((time.time() - start) * 1000)
            usage = data.get("usage", {})

            circuit_breaker.record_success()

            result_text = data["choices"][0]["message"]["content"].strip()
            if pf and anon_result and anon_result.anonymized_count > 0:
                result_text = pf.deanonymize(result_text)

            return {
                "text": result_text,
                "tokens_input": usage.get("prompt_tokens", 0),
                "tokens_output": usage.get("completion_tokens", 0),
                "time_ms": elapsed,
            }
        except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(1 * (attempt + 1))
                continue
            break

    circuit_breaker.record_failure()

    return {
        "text": f"[云端调用失败] {last_error}",
        "tokens_input": 0, "tokens_output": 0, "time_ms": 0,
        "error": str(last_error),
    }


def _get_privacy_filter() -> Any:
    """获取全局 PrivacyFilter 单例"""
    from task_router.privacy import get_privacy_filter
    return get_privacy_filter()
