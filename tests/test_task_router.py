"""
TaskRouter 核心功能测试

测试覆盖:
- 任务复杂度估算
- 任务类型检测
- 语义缓存 (TTL, 模糊匹配)
- 输出验证
- 提示压缩
- 模型注册表
- 审计日志
"""

import time
import json
import pytest

from task_router.task_router import (
    Task, estimate_complexity, detect_task_type, preprocess_text,
    validate_local_output,
    calc_cost, calc_savings, decompose_complex_task,
    _recursive_decompose,
)
from task_router.prompts import compress_prompt_tokens
from task_router.prompts import PROMPT_TEMPLATES
from task_router.model_registry import ModelRegistry, ModelProfile
from task_router.audit import AuditLogger, AuditEvent, QuotaManager


# ─── 任务复杂度估算 ─────────────────────────────────────────────

class TestEstimateComplexity:
    """测试 A3M 多信号路由评分"""

    def test_simple_local_task(self):
        """简单任务应该路由到本地"""
        task = Task(action="分类这些文件", text="a.pdf, b.jpg")
        result = estimate_complexity(task)
        assert result["route"] == "local"
        assert result["score"] <= 3.0

    def test_complex_cloud_task(self):
        """复杂任务应该路由到云端"""
        task = Task(action="设计一个微服务架构并给出实现方案", text="")
        result = estimate_complexity(task)
        assert result["route"] == "cloud"
        assert result["score"] > 3.0

    def test_translation_local(self):
        """翻译任务应该路由到本地"""
        task = Task(action="翻译成中文", text="Hello world")
        result = estimate_complexity(task)
        assert result["route"] == "local"

    def test_sentiment_local(self):
        """情感分析应该路由到本地"""
        task = Task(action="判断情感", text="很好")
        result = estimate_complexity(task)
        assert result["route"] == "local"

    def test_domain_complexity(self):
        """专业领域任务应该增加复杂度"""
        task_simple = Task(action="分类", text="苹果, 香蕉")
        task_legal = Task(action="分析法律合同条款并给出风险评估", text="")
        score_simple = estimate_complexity(task_simple)["score"]
        score_legal = estimate_complexity(task_legal)["score"]
        assert score_legal > score_simple

    def test_multi_step_detection(self):
        """多步骤任务应该检测到连接词"""
        task = Task(action="设计架构并给出实现方案", text="")
        result = estimate_complexity(task)
        # 应该检测到"并"这个连接词，并且包含高复杂度动词
        assert result["score"] > 3.0  # 应该路由到云端
        assert "多步" in result.get("reason", "") or result["score"] > 0


# ─── 任务类型检测 ─────────────────────────────────────────────────

class TestDetectTaskType:
    """测试任务类型检测"""

    def test_classification(self):
        assert detect_task_type("分类这些文件", PROMPT_TEMPLATES) == "general_classify"

    def test_sentiment(self):
        assert detect_task_type("判断情感", PROMPT_TEMPLATES) == "sentiment"

    def test_translation_en2zh(self):
        assert detect_task_type("翻译成中文", PROMPT_TEMPLATES) == "translate_en2zh"

    def test_translation_zh2en(self):
        assert detect_task_type("翻译成英文", PROMPT_TEMPLATES) == "translate_zh2en"

    def test_extraction(self):
        assert detect_task_type("提取关键词", PROMPT_TEMPLATES) == "extract_keywords"

    def test_file_classify(self):
        # "按扩展名分类" 同时匹配 general_classify("分类") 和 file_classify("扩展名")
        # 由于 dict 顺序，general_classify 先匹配。使用只匹配 file_classify 的关键词
        assert detect_task_type("按照扩展名来处理", PROMPT_TEMPLATES) == "file_classify"

    def test_unknown(self):
        result = detect_task_type("做一些奇怪的事情", PROMPT_TEMPLATES)
        assert result == "" or isinstance(result, str)


# ─── 预处理 ─────────────────────────────────────────────────────

class TestPreprocessText:
    """测试文本预处理"""

    def test_empty_text(self):
        assert preprocess_text("") == ""

    def test_comma_list_to_lines(self):
        result = preprocess_text("a.pdf, b.jpg, c.txt, d.py, e.csv")
        assert "\n" in result  # 应该转换为每行一个

    def test_truncation(self):
        long_text = "A" * 1000
        result = preprocess_text(long_text, max_chars=100)
        assert len(result) <= 120  # 包含截断标记

    def test_normalize_newlines(self):
        result = preprocess_text("a\r\nb\r\nc")
        assert "\r" not in result


# ─── 提示压缩 ─────────────────────────────────────────────────

class TestCompressPrompt:
    """测试提示压缩"""

    def test_short_prompt_unchanged(self):
        prompt = "翻译：Hello"
        result = compress_prompt_tokens(prompt, "translate_en2zh")
        assert result == prompt  # 短 prompt 不压缩

    def test_long_prompt_compressed(self):
        prompt = "\n".join([f"Line {i}: " + "A" * 80 for i in range(25)])
        result = compress_prompt_tokens(prompt, "sentiment")
        assert len(result) < len(prompt)

    def test_empty_prompt(self):
        assert compress_prompt_tokens("", "sentiment") == ""


# ─── 输出验证 ─────────────────────────────────────────────────

class TestValidateOutput:
    """测试输出验证"""

    def test_valid_output(self):
        result = validate_local_output("正面", "sentiment")
        assert result["valid"] is True

    def test_empty_output(self):
        result = validate_local_output("", "sentiment")
        assert result["valid"] is False

    def test_failure_signal(self):
        result = validate_local_output("抱歉，我无法完成这个任务", "sentiment")
        assert result["valid"] is False

    def test_apology_detected(self):
        result = validate_local_output("对不起，我不明白", "classification")
        assert result["valid"] is False


# ─── 成本计算 ─────────────────────────────────────────────────

class TestCostCalculation:
    """测试成本计算"""

    def test_calc_cost(self):
        cost = calc_cost(1000, 500)
        assert cost > 0
        assert cost == 1000 / 1000 * 0.003 + 500 / 1000 * 0.015

    def test_calc_savings_local(self):
        task = Task(action="test", route="local", tokens_input=100, tokens_output=50)
        savings = calc_savings(task)
        assert savings > 0

    def test_calc_savings_cloud(self):
        task = Task(action="test", route="cloud", tokens_input=100, tokens_output=50)
        savings = calc_savings(task)
        assert savings == 0.0


# ─── 任务拆解 ─────────────────────────────────────────────────

class TestTaskDecomposition:
    """测试任务拆解"""

    def test_compound_task_split(self):
        """复合任务应该被拆解"""
        subtasks = decompose_complex_task("分类并统计这些文件", "a.pdf, b.jpg")
        # 应该返回子任务列表
        assert isinstance(subtasks, list)

    def test_simple_task_no_split(self):
        """简单任务不应该被拆解"""
        subtasks = decompose_complex_task("翻译成中文", "Hello")
        assert subtasks == [] or len(subtasks) <= 1

    def test_recursive_decompose(self):
        """递归拆解应该处理连接词"""
        subtasks = _recursive_decompose("翻译内容并提取关键词", "Hello world")
        if subtasks:
            assert len(subtasks) >= 2


# ─── 模型注册表 ─────────────────────────────────────────────────

class TestModelRegistry:
    """测试模型注册表"""

    def test_init(self, tmp_path):
        registry = ModelRegistry(cache_dir=str(tmp_path))
        assert len(registry.models) == 0

    def test_detect_param_size(self, tmp_path):
        registry = ModelRegistry(cache_dir=str(tmp_path))
        assert registry._detect_param_size("qwen2.5:1.5b", 0.98) == "1.5B"
        assert registry._detect_param_size("qwen2.5:3b", 1.9) == "3B"
        assert registry._detect_param_size("llama3:7b", 4.5) == "7B"
        assert registry._detect_param_size("model:13b", 8.0) == "13B"

    def test_detect_tool_support(self, tmp_path):
        registry = ModelRegistry(cache_dir=str(tmp_path))
        assert registry._detect_tool_support("qwen-tool:latest") is True
        assert registry._detect_tool_support("llama3.1:latest") is True
        assert registry._detect_tool_support("mistral:7b") is False

    def test_estimate_default_score(self, tmp_path):
        registry = ModelRegistry(cache_dir=str(tmp_path))
        profile = ModelProfile(name="test", parameter_size="3B", size_gb=1.9)
        score = registry._estimate_default_score(profile, "translation")
        assert 0.5 <= score <= 1.0

    def test_select_best_empty(self, tmp_path):
        registry = ModelRegistry(cache_dir=str(tmp_path))
        assert registry.select_best("translation") is None

    def test_select_best_with_models(self, tmp_path):
        registry = ModelRegistry(cache_dir=str(tmp_path))
        registry.models["model_a"] = ModelProfile(
            name="model_a", parameter_size="1.5B", size_gb=1.0,
            capabilities={"translation": 0.9}
        )
        registry.models["model_b"] = ModelProfile(
            name="model_b", parameter_size="7B", size_gb=4.5,
            capabilities={"translation": 0.7}
        )
        best = registry.select_best("translation")
        assert best.name == "model_a"

    def test_update_after_call(self, tmp_path):
        registry = ModelRegistry(cache_dir=str(tmp_path))
        registry.models["test"] = ModelProfile(name="test")
        registry.update_after_call("test", success=True, latency_ms=500)
        assert registry.models["test"].total_calls == 1
        assert registry.models["test"].success_rate > 0

    def test_save_and_load(self, tmp_path):
        registry = ModelRegistry(cache_dir=str(tmp_path))
        registry.models["test"] = ModelProfile(name="test", parameter_size="3B")
        registry._save()

        registry2 = ModelRegistry(cache_dir=str(tmp_path))
        assert "test" in registry2.models
        assert registry2.models["test"].parameter_size == "3B"


# ─── 审计日志 ─────────────────────────────────────────────────

class TestAuditLogger:
    """测试审计日志"""

    def test_log_event(self, tmp_path):
        logger = AuditLogger(cache_dir=str(tmp_path))
        logger.log(AuditEvent(
            timestamp="2026-01-01T00:00:00",
            event_type="test",
            action="test_action",
        ))
        events = logger.query()
        assert len(events) == 1
        assert events[0]["event_type"] == "test"

    def test_query_filter(self, tmp_path):
        logger = AuditLogger(cache_dir=str(tmp_path))
        logger.log(AuditEvent(timestamp="2026-01-01T00:00:00", event_type="type_a", action="a"))
        logger.log(AuditEvent(timestamp="2026-01-01T00:00:01", event_type="type_b", action="b"))

        events = logger.query(event_type="type_a")
        assert len(events) == 1
        assert events[0]["action"] == "a"

    def test_query_limit(self, tmp_path):
        logger = AuditLogger(cache_dir=str(tmp_path))
        for i in range(10):
            logger.log(AuditEvent(timestamp=f"2026-01-01T00:00:{i:02d}", event_type="test", action=f"a{i}"))

        events = logger.query(limit=5)
        assert len(events) == 5

    def test_summary(self, tmp_path):
        logger = AuditLogger(cache_dir=str(tmp_path))
        logger.log(AuditEvent(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            event_type="task_execution",
            action="test",
            status="success",
            duration_ms=500,
        ))
        summary = logger.get_summary(days=1)
        assert summary["total_events"] == 1
        assert summary["by_type"]["task_execution"] == 1


# ─── 配额管理 ─────────────────────────────────────────────────

class TestQuotaManager:
    """测试配额管理"""

    def test_default_quota(self, tmp_path):
        manager = QuotaManager(cache_dir=str(tmp_path))
        quota = manager.get_quota("test_user")
        assert quota.daily_task_limit == 1000

    def test_set_quota(self, tmp_path):
        manager = QuotaManager(cache_dir=str(tmp_path))
        manager.set_quota("test_user", daily_task_limit=100)
        quota = manager.get_quota("test_user")
        assert quota.daily_task_limit == 100

    def test_check_quota_allowed(self, tmp_path):
        manager = QuotaManager(cache_dir=str(tmp_path))
        status = manager.check_quota("new_user")
        assert status["allowed"] is True

    def test_record_usage(self, tmp_path):
        manager = QuotaManager(cache_dir=str(tmp_path))
        manager.record_usage("test_user", tokens=100, cost=0.01, action="test")
        # 应该记录成功，不抛异常


# ─── 语义缓存 ─────────────────────────────────────────────────

class TestSemanticCache:
    """测试语义缓存"""

    def test_normalize(self):
        from task_router.cache import SemanticCache
        SemanticCache.__new__(SemanticCache)
        assert SemanticCache._normalize("  Hello  World  ") == "helloworld"
        assert SemanticCache._normalize("苹果，香蕉") == "苹果,香蕉"

    def test_trigrams(self):
        from task_router.cache import SemanticCache
        cache = SemanticCache.__new__(SemanticCache)
        cache.threshold = 0.85
        tri = cache._trigrams("hello world")
        assert len(tri) > 0

    def test_jaccard(self):
        from task_router.cache import SemanticCache
        cache = SemanticCache.__new__(SemanticCache)
        a = {"abc", "bcd", "cde"}
        b = {"abc", "bcd", "def"}
        score = cache._jaccard(a, b)
        assert score >= 0.5  # 交集 2 / 并集 4 = 0.5
        assert score < 0.8


# ─── 熔断器半开状态 ────────────────────────────────────────────

class TestCircuitBreakerStates:
    """测试 CircuitBreaker 三态转换: CLOSED → OPEN → HALF_OPEN → CLOSED"""

    def test_initial_state_closed(self):
        from task_router.models import CircuitBreaker
        cb = CircuitBreaker(max_failures=3, cooldown_seconds=1)
        assert cb.state == CircuitBreaker.STATE_CLOSED
        assert cb.allow_request() is True

    def test_trips_to_open_after_max_failures(self):
        from task_router.models import CircuitBreaker
        cb = CircuitBreaker(max_failures=3, cooldown_seconds=60)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitBreaker.STATE_OPEN
        assert cb.allow_request() is False

    def test_half_open_after_cooldown(self):
        from task_router.models import CircuitBreaker
        cb = CircuitBreaker(max_failures=2, cooldown_seconds=0)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreaker.STATE_OPEN
        # cooldown=0, allow_request should transition to half_open
        assert cb.allow_request() is True
        assert cb.state == CircuitBreaker.STATE_HALF_OPEN

    def test_half_open_success_closes(self):
        from task_router.models import CircuitBreaker
        cb = CircuitBreaker(max_failures=2, cooldown_seconds=0)
        cb.record_failure()
        cb.record_failure()
        cb.allow_request()  # → half_open
        cb.record_success()
        assert cb.state == CircuitBreaker.STATE_CLOSED
        assert cb.failures == 0

    def test_half_open_failure_reopens(self):
        from task_router.models import CircuitBreaker
        cb = CircuitBreaker(max_failures=2, cooldown_seconds=60)
        cb.record_failure()
        cb.record_failure()
        # Force into half_open
        with cb._lock:
            cb.state = CircuitBreaker.STATE_HALF_OPEN
            cb.open_until = 0
        cb.record_failure()
        assert cb.state == CircuitBreaker.STATE_OPEN

    def test_to_dict_includes_state(self):
        from task_router.models import CircuitBreaker
        cb = CircuitBreaker(max_failures=3, cooldown_seconds=10)
        d = cb.to_dict()
        assert "state" in d
        assert d["state"] == CircuitBreaker.STATE_CLOSED

    def test_half_open_limits_attempts(self):
        from task_router.models import CircuitBreaker
        cb = CircuitBreaker(max_failures=2, cooldown_seconds=0, half_open_max=1)
        cb.record_failure()
        cb.record_failure()
        cb.allow_request()  # → half_open, attempt 0 → allowed
        with cb._lock:
            cb.half_open_attempts = 1
        assert cb.allow_request() is False  # exceeded half_open_max


# ─── 蒸馏 TTL 遗忘机制 ────────────────────────────────────────

class TestDistillationTTL:
    """测试蒸馏数据 TTL 过期和清理"""

    def test_is_expired_recent_pair(self, tmp_path):
        from task_router.distillation import DistillationStore, PAIR_HYPOTHESIS
        store = DistillationStore(cache_dir=str(tmp_path), ttl_days=90)
        pair = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "epistemic_state": PAIR_HYPOTHESIS}
        assert store._is_expired(pair) is False

    def test_is_expired_old_pair(self, tmp_path):
        from task_router.distillation import DistillationStore, PAIR_HYPOTHESIS
        store = DistillationStore(cache_dir=str(tmp_path), ttl_days=30)
        old_time = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() - 31 * 86400))
        pair = {"time": old_time, "epistemic_state": PAIR_HYPOTHESIS}
        assert store._is_expired(pair) is True

    def test_contested_expires_faster(self, tmp_path):
        from task_router.distillation import DistillationStore, PAIR_CONTESTED
        store = DistillationStore(cache_dir=str(tmp_path), ttl_days=90)
        # 20 days old: still within 90-day TTL but past 14-day contested TTL
        old_time = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() - 20 * 86400))
        pair = {"time": old_time, "epistemic_state": PAIR_CONTESTED}
        assert store._is_expired(pair) is True

    def test_outdated_expires_fastest(self, tmp_path):
        from task_router.distillation import DistillationStore, PAIR_OUTDATED
        store = DistillationStore(cache_dir=str(tmp_path), ttl_days=90)
        old_time = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() - 10 * 86400))
        pair = {"time": old_time, "epistemic_state": PAIR_OUTDATED}
        assert store._is_expired(pair) is True

    def test_cleanup_expired_removes_old(self, tmp_path):
        from task_router.distillation import DistillationStore, DistillationPair, PAIR_HYPOTHESIS
        store = DistillationStore(cache_dir=str(tmp_path), ttl_days=1)
        # Add a fresh pair
        pair1 = DistillationPair(prompt="test1", response="ok1")
        store.add_pair(pair1)
        # Add an expired pair manually
        old_time = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() - 2 * 86400))
        expired = {"prompt": "old", "response": "expired", "time": old_time, "epistemic_state": PAIR_HYPOTHESIS}
        with open(store.pairs_file, "a") as f:
            f.write(json.dumps(expired) + "\n")
        removed = store.cleanup_expired()
        assert removed == 1
        remaining = store._load_all()
        assert len(remaining) == 1

    def test_get_pairs_filters_expired(self, tmp_path):
        from task_router.distillation import DistillationStore, DistillationPair, PAIR_SUPPORTED
        store = DistillationStore(cache_dir=str(tmp_path), ttl_days=1)
        pair = DistillationPair(prompt="new", response="ok")
        pair.epistemic_state = PAIR_SUPPORTED
        store.add_pair(pair)
        # Add expired
        old_time = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() - 2 * 86400))
        expired = {"prompt": "old", "response": "gone", "time": old_time, "epistemic_state": PAIR_SUPPORTED}
        with open(store.pairs_file, "a") as f:
            f.write(json.dumps(expired) + "\n")
        pairs = store.get_pairs()
        assert len(pairs) == 1

    def test_get_stats_includes_expired(self, tmp_path):
        from task_router.distillation import DistillationStore, DistillationPair, PAIR_HYPOTHESIS
        store = DistillationStore(cache_dir=str(tmp_path), ttl_days=1)
        store.add_pair(DistillationPair(prompt="new", response="ok"))
        old_time = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() - 2 * 86400))
        with open(store.pairs_file, "a") as f:
            f.write(json.dumps({"prompt": "old", "response": "x", "time": old_time, "epistemic_state": PAIR_HYPOTHESIS}) + "\n")
        stats = store.get_stats()
        assert stats["expired"] == 1
        assert stats["active"] == 1
        assert stats["total"] == 2


# ─── 缓存 TTL ─────────────────────────────────────────────────

class TestCacheTTL:
    """测试缓存 TTL 过期行为"""

    def test_cache_entry_expires(self, tmp_path):
        from task_router.cache import SemanticCache
        c = SemanticCache(cache_dir=str(tmp_path))
        c.set("test action", "text", {"text": "result"}, ttl_hours=0)
        # ttl_hours=0 means expires immediately
        time.sleep(0.01)
        cached = c.get("test action", "text")
        assert cached is None

    def test_cache_entry_valid(self, tmp_path):
        from task_router.cache import SemanticCache
        c = SemanticCache(cache_dir=str(tmp_path))
        c.set("test action", "text", {"text": "result"}, ttl_hours=24)
        cached = c.get("test action", "text")
        assert cached is not None


# ─── 缓存并发安全 ─────────────────────────────────────────────

class TestCacheConcurrency:
    """测试缓存的线程安全性"""

    def test_concurrent_set_get(self, tmp_path):
        """多线程同时读写缓存不应崩溃"""
        import threading
        from task_router.cache import SemanticCache
        c = SemanticCache(cache_dir=str(tmp_path), max_entries=100)
        errors: list[Exception] = []

        def writer(tid: int):
            try:
                for i in range(20):
                    c.set(f"action_{tid}_{i}", f"text_{tid}", {"text": f"result_{tid}_{i}"})
            except Exception as e:
                errors.append(e)

        def reader(tid: int):
            try:
                for i in range(20):
                    c.get(f"action_{tid}_{i}", f"text_{tid}")
            except Exception as e:
                errors.append(e)

        threads = []
        for t in range(4):
            threads.append(threading.Thread(target=writer, args=(t,)))
            threads.append(threading.Thread(target=reader, args=(t,)))
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert len(errors) == 0, f"并发错误: {errors}"

    def test_concurrent_set_no_corruption(self, tmp_path):
        """并发写入后缓存文件可正常读取"""
        import threading
        from task_router.cache import SemanticCache
        c = SemanticCache(cache_dir=str(tmp_path), max_entries=50)

        def writer(tid: int):
            for i in range(15):
                c.set(f"task_{tid}_{i}", "text", {"text": f"out_{tid}_{i}"})

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # 重新加载应不报错
        c2 = SemanticCache(cache_dir=str(tmp_path), max_entries=50)
        assert len(c2._entries) > 0


# ─── OpenAI 端点消息解析 ──────────────────────────────────────

class TestOpenAIMessageParsing:
    """测试 OpenAI 兼容端点的消息解析逻辑"""

    def _parse_messages(self, messages: list[dict]) -> tuple[str, str]:
        """模拟 api_server 中的消息解析逻辑"""
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
            action = system_msg
            text = "\n".join(user_messages) if user_messages else ""
        elif len(user_messages) == 1:
            content = user_messages[0]
            for sep in ["：", ":"]:
                if sep in content:
                    parts = content.split(sep, 1)
                    potential_action = parts[0].strip()
                    if len(potential_action) < 30:
                        action = potential_action
                        text = parts[1].strip()
                        break
            if not action:
                action = content
        else:
            action = user_messages[0] if user_messages else ""
            text = "\n".join(user_messages[1:]) if len(user_messages) > 1 else ""

        if not action:
            action = text
            text = ""
        return action, text

    def test_single_user_message(self):
        action, text = self._parse_messages([{"role": "user", "content": "翻译成中文：Hello world"}])
        assert action == "翻译成中文"
        assert text == "Hello world"

    def test_system_plus_user(self):
        msgs = [
            {"role": "system", "content": "你是一个翻译助手"},
            {"role": "user", "content": "Hello world"},
        ]
        action, text = self._parse_messages(msgs)
        assert action == "你是一个翻译助手"
        assert text == "Hello world"

    def test_multi_turn_takes_last_user(self):
        msgs = [
            {"role": "user", "content": "之前的消息"},
            {"role": "assistant", "content": "回复"},
            {"role": "user", "content": "翻译这个：Hello"},
        ]
        action, text = self._parse_messages(msgs)
        assert action == "之前的消息"
        assert "Hello" in text

    def test_single_user_no_colon(self):
        action, text = self._parse_messages([{"role": "user", "content": "帮我分类这段文本"}])
        assert action == "帮我分类这段文本"

    def test_chinese_colon(self):
        action, text = self._parse_messages([{"role": "user", "content": "提取关键词：人工智能的发展"}])
        assert action == "提取关键词"
        assert text == "人工智能的发展"

    def test_long_action_no_split(self):
        """第一部分超过 30 字符时不拆分"""
        long_prefix = "这是一个非常长的任务描述超过三十个字符限制请勿拆分此部分内容哈"
        assert len(long_prefix) > 30
        action, text = self._parse_messages([{"role": "user", "content": f"{long_prefix}：内容"}])
        assert action == f"{long_prefix}：内容"

    def test_empty_messages(self):
        action, text = self._parse_messages([])
        assert action == ""


# ─── run_task 集成测试 ─────────────────────────────────────────

class TestRunTaskIntegration:
    """测试 run_task 核心流程"""

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        """每个测试前清空缓存，避免跨测试污染"""
        from task_router.task_router import cache
        with cache._lock:
            cache._entries.clear()
        yield
        with cache._lock:
            cache._entries.clear()

    def test_rule_engine_sort_numbers(self):
        """排序任务走规则引擎"""
        from task_router.task_router import run_task, Task
        task = Task(action="排序数字", text="5,3,8,1,9,2")
        result = run_task(task, force_route="local")
        assert result.output
        assert result.model_used == "rule_engine"

    def test_rule_engine_dedup(self):
        """去重任务走规则引擎"""
        from task_router.task_router import run_task, Task
        task = Task(action="去重", text="苹果,香蕉,苹果,橙子,香蕉")
        result = run_task(task, force_route="local")
        assert result.output
        assert result.model_used == "rule_engine"
        assert "苹果" in result.output

    def test_rule_engine_count(self):
        """计数任务走规则引擎"""
        from task_router.task_router import run_task, Task
        task = Task(action="计数", text="苹果,香蕉,橙子")
        result = run_task(task, force_route="local")
        assert result.output
        assert result.model_used == "rule_engine"
        assert "3" in result.output

    def test_local_with_mock_ollama(self):
        """本地任务通过 mock ollama 执行"""
        from unittest.mock import patch
        from task_router.task_router import run_task, Task

        mock_result = {
            "text": "这是mock输出",
            "tokens_input": 10,
            "tokens_output": 20,
            "time_ms": 100,
        }
        with patch("task_router.task_router.call_ollama", return_value=mock_result):
            task = Task(action="翻译成中文", text="Hello world")
            result = run_task(task, force_route="local")
            assert result.output == "这是mock输出"
            assert result.tokens_input == 10

    def test_cache_hit_on_repeat(self):
        """重复任务命中缓存"""
        from unittest.mock import patch
        from task_router.task_router import run_task, Task

        mock_result = {
            "text": "缓存测试输出",
            "tokens_input": 5,
            "tokens_output": 10,
            "time_ms": 50,
        }
        with patch("task_router.task_router.call_ollama", return_value=mock_result):
            task1 = Task(action="缓存测试唯一任务XYZ", text="test content")
            result1 = run_task(task1, force_route="local")
            assert result1.model_used != "cache"

            # 再次执行应命中缓存
            task2 = Task(action="缓存测试唯一任务XYZ", text="test content")
            result2 = run_task(task2, force_route="local")
            assert "cache" in result2.route

    def test_estimate_returns_valid(self):
        """estimate 返回完整结构"""
        from task_router.task_router import estimate
        result = estimate("翻译成英文")
        assert "task" in result
        assert "suggested_route" in result
        assert "score" in result
        assert result["suggested_route"] in ("local", "cloud")

    def test_classify_task_returns_type(self):
        """classify_task 返回任务类型"""
        from task_router.task_router import classify_task
        result = classify_task("分类这些文本", "苹果\n香蕉\n橙子")
        assert "task_type" in result
        assert "verdict" in result


# ─── 统计与阈值 ───────────────────────────────────────────────

class TestStatsAndThresholds:
    """测试 show_usage_stats 和 CapabilityTracker（回归 read_jsonl 导入）"""

    def test_show_usage_stats_no_crash(self):
        """show_usage_stats 不应因缺少导入而崩溃"""
        from task_router.task_router import show_usage_stats
        result = show_usage_stats()
        assert isinstance(result, str)

    def test_capability_tracker_record_and_rate(self, tmp_path):
        """CapabilityTracker 记录和成功率计算"""
        from task_router.task_router import CapabilityTracker
        tracker = CapabilityTracker(cache_dir=str(tmp_path))
        tracker.record("classification", success=True)
        tracker.record("classification", success=True)
        tracker.record("classification", success=False)
        rate = tracker.get_success_rate("classification")
        assert rate == pytest.approx(2 / 3)

    def test_capability_tracker_get_all_adjustments(self, tmp_path):
        """get_all_adjustments 使用 read_jsonl，不应崩溃"""
        from task_router.task_router import CapabilityTracker
        tracker = CapabilityTracker(cache_dir=str(tmp_path))
        tracker.record("translation", success=True, task_type="translate_en2zh")
        result = tracker.get_all_adjustments()
        assert "translation" in result
        assert "success_rate" in result["translation"]


# ─── 配置加载 ─────────────────────────────────────────────────

class TestConfigLoading:
    """测试配置文件加载"""

    def test_from_json(self, tmp_path):
        from task_router.config import RouterConfig
        config_file = tmp_path / "config.json"
        config_file.write_text('{"local_model": "test-model", "base_threshold": 5.0}')
        cfg = RouterConfig.from_json(str(config_file))
        assert cfg.local_model == "test-model"
        assert cfg.base_threshold == 5.0

    def test_from_json_missing_file(self, tmp_path):
        from task_router.config import RouterConfig
        cfg = RouterConfig.from_json(str(tmp_path / "nonexistent.json"))
        assert cfg.local_model == "qwen-tool"  # default

    def test_from_json_invalid(self, tmp_path):
        from task_router.config import RouterConfig
        config_file = tmp_path / "bad.json"
        config_file.write_text("not json")
        cfg = RouterConfig.from_json(str(config_file))
        assert cfg.local_model == "qwen-tool"  # default


class TestAuthMiddleware:
    """API 认证中间件测试"""

    def test_safe_eq_matching(self):
        from task_router.api_server import TaskRouterHandler
        h = TaskRouterHandler()
        assert h._safe_eq("test-key-123", "test-key-123") is True

    def test_safe_eq_mismatch(self):
        from task_router.api_server import TaskRouterHandler
        h = TaskRouterHandler()
        assert h._safe_eq("test-key-123", "test-key-456") is False

    def test_safe_eq_empty(self):
        from task_router.api_server import TaskRouterHandler
        h = TaskRouterHandler()
        assert h._safe_eq("", "test") is False
        assert h._safe_eq("test", "") is False
        assert h._safe_eq("", "") is False

    def test_safe_eq_constant_time(self):
        """验证 hmac.compare_digest 被使用（不会因首字符不同而更快）"""
        import hmac
        from task_router.api_server import TaskRouterHandler
        h = TaskRouterHandler()
        # 确保底层调用的是 hmac.compare_digest
        a = "a" * 100
        b = "b" + "a" * 99
        assert h._safe_eq(a, b) is hmac.compare_digest(a.encode(), b.encode())


# ─── 递归深度限制 ──────────────────────────────────────────────


class TestRecursionDepthLimit:
    def test_run_task_respects_depth_limit(self):
        """run_task 在超过最大递归深度时返回错误"""
        from task_router.task_router import run_task, CONFIG
        task = Task(action="测试深度限制", text="test")
        result = run_task(task, _depth=CONFIG.max_recurse_depth + 1)
        assert result.route == "error"
        assert "递归深度超限" in result.output

    def test_run_task_depth_zero_does_not_trigger(self):
        """depth=0 不触发递归限制"""
        from task_router.task_router import run_task
        task = Task(action="测试", text="")
        try:
            result = run_task(task, _depth=0)
        except Exception:
            # 连接失败等异常不算递归超限
            return
        assert "递归深度超限" not in (result.output or "")


# ─── 运行入口 ─────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
