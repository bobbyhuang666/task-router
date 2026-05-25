#!/usr/bin/env python3
"""
TaskRouter — 企业 LLM 网关

核心编排器：路由决策 → 缓存 → 执行 → 学习 → 审计

模块拆分：
- preprocessing.py: 文本预处理/后处理
- cost.py: 成本计算、使用统计、CapabilityTracker
- task_planner.py: 任务分解、计划执行
- 本文件: 核心编排逻辑
"""

import os  # noqa: F401 — re-exported for downstream modules
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

log = logging.getLogger("task_router")

# ─── 模块导入 ──────────────────────────────────────────────────

from task_router.config import get_config, TASK_TO_CAPABILITY
from task_router.routing import (
    Task, estimate_complexity, detect_task_type,
    decompose_complex_task, _recursive_decompose,
)
from task_router.cache import SemanticCache
from task_router.models import call_ollama, call_cloud_api, circuit_breaker  # noqa: F401
from task_router.prompts import (
    PROMPT_TEMPLATES, build_optimized_prompt, get_max_tokens,
)
from task_router.rules import rule_execute
from task_router.validation import validate_local_output
from task_router.distillation import (
    DistillationStore, collect_distillation_pair,
    get_dynamic_examples,
)
from task_router.reasoning import (
    select_strategy, enhance_prompt_with_strategy,
    STRATEGY_TOKEN_MULTIPLIER, TOKEN_BUDGET, get_strategy_tracker,
)
from task_router.adaptive_compression import compress_adaptive
from task_router.confidence import extract_confidence, extract_confidence_from_text, CascadeDecision
from task_router.meta_learner import extract_routing_features, get_meta_learner, get_active_learner
from task_router.tqbc import TQBCRouter, extract_quantile_features, TokenQuantileFeatures
from task_router.outcome_cache import OutcomeAwareCache
from task_router.conformal_routing import ConformalizedRouter

# 从拆分模块导入（保持向后兼容的导入路径）
from task_router.io_utils import read_jsonl, append_jsonl  # noqa: F401
from task_router.preprocessing import preprocess_text, postprocess_output
from task_router.cost import (  # noqa: F401
    calc_cost, calc_savings, log_usage, _scrub_pii,
    _classify_route, show_usage_stats, CapabilityTracker,
)

# Re-export task_planner API (used by tests and CLI)
from task_router.task_planner import (  # noqa: F401
    estimate, classify_task, decompose_task, execute_plan,
    DECOMPOSE_TEMPLATES,
)


# ─── 全局实例（延迟初始化，集中管理）──────────────────────────

CONFIG = get_config()
cache = SemanticCache(
    cache_dir=CONFIG.cache_dir,
    max_entries=CONFIG.cache_max_entries,
    fuzzy_threshold=CONFIG.cache_fuzzy_threshold,
)
store = DistillationStore(cache_dir=CONFIG.cache_dir)
cap_tracker = CapabilityTracker(CONFIG.cache_dir)

RECURSE_SCORE_MIN = CONFIG.recurse_score_min
RECURSE_SCORE_MAX = CONFIG.recurse_score_max
DEFAULT_CAPABILITY_THRESHOLD = CONFIG.base_threshold

# ─── 延迟初始化单例（统一管理）──────────────────────────────────

_registry: dict[str, Any] = {
    "model_registry": None,
    "weight_tracker": None,
    "cascade": None,
    "tqbc": None,
    "outcome_cache": None,
    "conformal": None,
}
_registry_lock = threading.Lock()


def _get_instance(name: str, factory) -> Any:
    """统一的延迟初始化单例获取"""
    if _registry[name] is not None:
        return _registry[name]
    with _registry_lock:
        if _registry[name] is not None:
            return _registry[name]
        _registry[name] = factory()
        return _registry[name]


def get_model_registry() -> Any:
    def _create():
        from task_router.model_registry import ModelRegistry
        r = ModelRegistry(cache_dir=CONFIG.cache_dir)
        if not r.models:
            r.discover()
        return r
    return _get_instance("model_registry", _create)


def _get_weight_tracker():
    def _create():
        from task_router.weights import get_weight_tracker
        return get_weight_tracker(CONFIG.cache_dir)
    return _get_instance("weight_tracker", _create)


def get_cascade():
    return _get_instance("cascade", lambda: CascadeDecision(CONFIG.cache_dir))


def get_tqbc():
    return _get_instance("tqbc", lambda: TQBCRouter(CONFIG.cache_dir))


def get_outcome_cache():
    return _get_instance("outcome_cache", lambda: OutcomeAwareCache(CONFIG.cache_dir))


def get_conformal_router():
    return _get_instance("conformal", lambda: ConformalizedRouter(CONFIG.cache_dir))


def reset_singletons() -> None:
    """重置所有单例（测试用）"""
    with _registry_lock:
        for key in _registry:
            _registry[key] = None


# ─── 蒸馏辅助 ──────────────────────────────────────────────────


def enrich_prompt_with_examples(prompt: str, task_type: str, text: str) -> str:
    """从蒸馏池注入动态 few-shot 示例"""
    capability = TASK_TO_CAPABILITY.get(task_type, "")
    if not capability:
        return prompt
    examples = get_dynamic_examples(capability, store, limit=2)
    if not examples:
        return prompt
    example_block = "\n".join(f"示例: {e.get('prompt', '')[:50]} → {e.get('response', '')[:50]}" for e in examples)
    return f"{example_block}\n\n{prompt}"


def auto_collect_on_cloud(task: Task, result: dict) -> None:
    if not result.get("text"):
        return
    pair = collect_distillation_pair(
        prompt=_scrub_pii(task.action), response=_scrub_pii(result["text"]),
        task_type=detect_task_type(task.action, PROMPT_TEMPLATES),
        route="cloud", action=_scrub_pii(task.action), model_used=task.model_used,
    )
    store.add_pair(pair)


def auto_collect_on_local_failure(task: Task, local_output: str, cloud_output: str, failure_type: str = "") -> None:
    pair = collect_distillation_pair(
        prompt=_scrub_pii(task.action), response=_scrub_pii(cloud_output),
        task_type=detect_task_type(task.action, PROMPT_TEMPLATES),
        route="local_fallback", action=_scrub_pii(task.action), model_used=task.model_used,
    )
    pair.local_response = _scrub_pii(local_output)
    pair.failure_type = failure_type
    store.add_pair(pair)


# ─── 核心执行 ──────────────────────────────────────────────────


def _check_cache(task: Task) -> Optional[Task]:
    """检查语义缓存（含质量感知），命中则返回填充好的 Task，否则返回 None"""
    cached = cache.get(task.action, task.text)
    if not cached:
        return None

    # OATS-Inspired 质量感知：检查缓存条目历史质量
    cache_key = cache._make_key(task.action, task.text)
    outcome_cache = get_outcome_cache()
    quality = outcome_cache.get_quality(cache_key)
    if quality < 0.2:
        log.debug("缓存质量过低 (%.2f)，跳过: %s", quality, cache_key[:30])
        return None

    task.output = cached.get("result", {}).get("text", cached.get("output", ""))
    task.route = f"cache({cached.get('match_type', 'hit')})"
    task.model_used = "cache"
    task.tokens_input = 0
    task.tokens_output = 0
    task.time_ms = 0
    task.cost_saved = round(calc_cost(
        cached.get("result", {}).get("tokens_input", 0),
        cached.get("result", {}).get("tokens_output", 0)), 6)
    return task


def _run_local_subtasks(task: Task, subtasks: list[dict], clean_text: str) -> bool:
    """执行本地子任务列表，成功则填充 task 并返回 True"""
    if not subtasks:
        return False
    outputs: list[str] = []
    all_rules = True
    for st in subtasks:
        st_type = st.get("type", "")
        st_action = st.get("action", "")
        st_text = st.get("text", clean_text)
        rule_result = rule_execute(st_type, st_text)
        if rule_result:
            outputs.append(rule_result)
            continue
        all_rules = False
        st_prompt = build_optimized_prompt(st_type, st_action, st_text, task.files)
        st_prompt = enrich_prompt_with_examples(st_prompt, st_type, st_text)
        result = call_ollama(st_prompt, max_tokens=get_max_tokens(st_type))
        out = postprocess_output(result["text"], st_type)
        outputs.append(f"[{st_type}]\n{out}")
        task.tokens_input += result["tokens_input"]
        task.tokens_output += result["tokens_output"]
        task.time_ms += result["time_ms"]
    task.output = "\n\n".join(outputs)
    task.model_used = "rule_engine" if all_rules else CONFIG.local_model
    task.cost_saved = calc_savings(task)
    return True


def _run_local_single(task: Task, clean_action: str, clean_text: str) -> dict:
    """执行单个本地任务，返回 result 字典（含置信度数据）"""
    task_type = detect_task_type(clean_action, PROMPT_TEMPLATES)

    # 规则执行
    rule_result = rule_execute(task_type, clean_text or clean_action)
    if rule_result:
        task.output = rule_result
        task.model_used = "rule_engine"
        task.route = "local(rule)"
        return {"text": rule_result, "tokens_input": 0, "tokens_output": 0, "time_ms": 0,
                "confidence": {"confidence": 1.0, "entropy": 0.0, "margin": 1.0}}

    # 智能模型选择
    selected_model: Optional[str] = None
    try:
        registry = get_model_registry()
        if registry.models:
            capability = TASK_TO_CAPABILITY.get(task_type, "")
            if capability:
                best = registry.select_best(capability, prefer_speed=True)
                if best and best.name != CONFIG.local_model:
                    selected_model = best.name
    except Exception as e:
        log.debug("模型注册表查询失败: %s", e)

    prompt = build_optimized_prompt(task_type, clean_action, clean_text, task.files)

    # 获取动态示例
    examples_str = ""
    has_examples = False
    examples = []
    capability = TASK_TO_CAPABILITY.get(task_type, "")
    if capability:
        examples = get_dynamic_examples(capability, store, limit=2)
    if examples:
        has_examples = True
        example_block = "\n".join(
            f"示例: {e.get('prompt', '')[:50]} → {e.get('response', '')[:50]}" for e in examples)
        prompt = f"{example_block}\n\n{prompt}"
        examples_str = "\n".join(
            f"{e.get('prompt', '')[:50]} → {e.get('response', '')[:50]}" for e in examples)

    # 推理策略选择
    routing_result = estimate_complexity(task)
    strategy_decision = select_strategy(
        action=clean_action, text=clean_text,
        complexity_score=routing_result["score"], task_type=task_type,
        logprobs=None, has_examples=has_examples,
    )
    strategy = strategy_decision.strategy

    # 策略反馈优化
    tracker = get_strategy_tracker()
    historical_best = tracker.get_best_strategy(task_type)
    if historical_best and historical_best != strategy:
        log.debug("策略反馈优化: %s → %s (历史数据)", strategy, historical_best)
        strategy = historical_best

    if has_examples and strategy != "direct":
        strategy = "few_shot"

    prompt = enhance_prompt_with_strategy(prompt, strategy, examples_str)

    # 自适应 Prompt 压缩
    strategy_confidence_map = {
        "direct": 0.85, "cod": 0.65, "cot": 0.5, "few_shot": 0.6, "structured": 0.45,
    }
    estimated_conf = strategy_confidence_map.get(strategy, 0.5)
    compression_result = compress_adaptive(prompt, confidence=estimated_conf, task_type=task_type)
    prompt = compression_result.compressed_prompt
    if compression_result.level != "none":
        log.debug("Prompt 压缩: %s 级别, %.0f%% 保留", compression_result.level,
                  compression_result.compression_ratio * 100)

    max_tokens = int(get_max_tokens(task_type) * STRATEGY_TOKEN_MULTIPLIER.get(strategy, 1.0))

    # 模型调用（请求 logprobs）
    result = call_ollama(prompt, model=selected_model, max_tokens=max_tokens, with_logprobs=True)

    # 提取置信度信号
    result["strategy"] = strategy
    logprobs = result.get("logprobs", [])
    if logprobs:
        result["confidence"] = extract_confidence(logprobs)
    else:
        result["confidence"] = extract_confidence_from_text(result["text"])

    # 模型调用后重新评估
    adaptive_decision = select_strategy(
        action=clean_action, text=clean_text,
        complexity_score=routing_result["score"], task_type=task_type,
        logprobs=logprobs, has_examples=has_examples,
    )
    result["adaptive_strategy"] = adaptive_decision.strategy
    result["adaptive_confidence"] = adaptive_decision.confidence_signal
    result["token_budget_factor"] = adaptive_decision.token_budget_factor

    # Token 预算优化评估
    optimal_budget = TOKEN_BUDGET.get(adaptive_decision.strategy, 1.0)
    current_budget = STRATEGY_TOKEN_MULTIPLIER.get(strategy, 1.0)
    if optimal_budget < current_budget * 0.5:
        log.info("Token 预算优化机会: 策略 %s (预算 %.1fx) → %s (预算 %.1fx), 节省 %.0f%%",
                 strategy, current_budget, adaptive_decision.strategy, optimal_budget,
                 (1 - optimal_budget / current_budget) * 100)

    task.output = postprocess_output(result["text"], task_type)
    task.model_used = selected_model or CONFIG.local_model
    return result


def _validate_and_fallback(task: Task, result: dict, task_type: str) -> dict:
    """验证本地输出质量，必要时降级到云端"""
    validation = validate_local_output(task.output, task_type)
    cap = TASK_TO_CAPABILITY.get(task_type, "")
    if cap:
        cap_tracker.record(cap, success=validation["valid"],
                           score=1.0 if validation["valid"] else 0.3, task_type=task_type)

    if validation["valid"]:
        return result

    if CONFIG.cloud_api_key:
        cloud_result = call_cloud_api(task.action, task.text)
        if cloud_result.get("circuit_open"):
            task.route = "local(degraded)"
        else:
            auto_collect_on_local_failure(task, task.output, cloud_result["text"], "quality_fallback")
            task.output = cloud_result["text"]
            task.route = "cloud_fallback"
            task.model_used = CONFIG.cloud_model
            result = cloud_result
            task.cost_saved = 0

    return result


def _run_local(task: Task) -> Task:
    """执行本地任务（含置信度门控级联）"""
    clean_text = preprocess_text(task.text or "")
    clean_action = preprocess_text(task.action, max_chars=200)

    # 尝试拆解复合任务
    subtasks = decompose_complex_task(clean_action, clean_text)
    if not subtasks:
        subtasks = _recursive_decompose(clean_action, clean_text)

    if _run_local_subtasks(task, subtasks, clean_text):
        return task

    # 单任务
    task_type = detect_task_type(clean_action, PROMPT_TEMPLATES)
    result = _run_local_single(task, clean_action, clean_text)

    if task.model_used == "rule_engine":
        return task

    # ── 五层决策融合 ──
    cascade = get_cascade()
    conf_data = result.get("confidence", {})
    cascade_decision = cascade.should_escalate(conf_data)

    ml = get_meta_learner()
    al = get_active_learner()
    routing_decision = estimate_complexity(task)
    cap = TASK_TO_CAPABILITY.get(task_type, "")
    cap_success = cap_tracker.get_success_rate(cap) if cap else 0.5
    features = extract_routing_features(
        complexity_score=routing_decision["score"],
        confidence_data=conf_data, text_length=len(clean_text),
        file_count=len(task.files), capability_success_rate=cap_success,
        strategy=result.get("strategy", "direct"),
    )
    ml_prediction = ml.predict(features)

    active_verify = al.should_request_verification(task_type, min_samples=5) and CONFIG.cloud_api_key

    tqbc = get_tqbc()
    logprobs = result.get("logprobs", [])
    tqbc_decision = tqbc.decide(
        logprobs=logprobs, complexity_score=routing_decision["score"],
        task_type=task_type, text_length=len(clean_text),
        capability_success_rate=cap_success,
    )

    raw_should_escalate = (
        cascade_decision["escalate"]
        or (not ml_prediction["should_use_local"] and ml_prediction["confidence"] > 0.5 and ml.get_stats()["total"] >= 50)
        or active_verify
        or tqbc_decision.should_escalate
    )

    # Conformalized Router
    quantile_features = extract_quantile_features(logprobs) if logprobs else TokenQuantileFeatures()
    conformal = get_conformal_router()
    conformal_decision = conformal.decide(
        cascade_decision=cascade_decision, tqbc_decision=tqbc_decision,
        ml_prediction=ml_prediction, active_verify=active_verify,
        features=features, task_type=task_type,
        raw_should_escalate=raw_should_escalate,
        quantile_features=quantile_features,
    )
    should_escalate = conformal_decision.should_escalate

    def _record_and_return(escalated: bool, cloud_used: bool = False) -> None:
        local_success = not cloud_used and task.route != "cloud_fallback" and task.route != "error"
        cascade.record_outcome(conf_data, was_correct=local_success or cloud_used, escalated=escalated)
        ml.record_and_learn(features, success=local_success, route=task.route, task_type=task_type)
        al.record(task_type, ml_prediction["local_success_prob"], local_success)
        tqbc.record_outcome(decision=tqbc_decision, success=local_success or cloud_used,
                            escalated=escalated, task_type=task_type)
        conformal.record_outcome(decision=conformal_decision, success=local_success or cloud_used,
                                 escalated=escalated, task_type=task_type, features=features)
        used_strategy = result.get("strategy", "direct")
        get_strategy_tracker().record(used_strategy, task_type, local_success or cloud_used)
        if task.model_used == "cache":
            cache_key = cache._make_key(task.action, task.text)
            get_outcome_cache().record_outcome(cache_key, success=local_success or cloud_used)

    if should_escalate and CONFIG.cloud_api_key:
        cloud_result = call_cloud_api(task.action, task.text)
        if not cloud_result.get("circuit_open"):
            task.output = cloud_result["text"]
            task.route = "cascade_escalated"
            task.model_used = CONFIG.cloud_model
            task.tokens_input = cloud_result.get("tokens_input", 0)
            task.tokens_output = cloud_result.get("tokens_output", 0)
            task.time_ms = cloud_result.get("time_ms", 0)
            task.cost_saved = 0
            _record_and_return(escalated=True, cloud_used=True)
            return task
        cascade_escalated = False
    else:
        cascade_escalated = False

    result = _validate_and_fallback(task, result, task_type)
    task.tokens_input = result.get("tokens_input", 0)
    task.tokens_output = result.get("tokens_output", 0)
    task.time_ms = result.get("time_ms", 0)
    task.cost_saved = calc_savings(task)

    _record_and_return(escalated=cascade_escalated or task.route == "cloud_fallback")
    return task


def _run_cloud(task: Task, force_route: str, _depth: int = 0) -> Task:
    """执行云端任务（含递归拆解）"""
    if not force_route:
        decision = estimate_complexity(task, base_threshold=CONFIG.base_threshold)
    else:
        decision = {"score": 9, "route": force_route}

    if RECURSE_SCORE_MIN <= decision["score"] <= RECURSE_SCORE_MAX:
        subtasks = _recursive_decompose(task.action, task.text)
        if subtasks:
            outputs = []
            for st in subtasks:
                st_action = st.get("action", st.get("type", ""))
                st_text = st.get("text", task.text)
                st_task = Task(action=st_action, text=st_text)
                st_result = run_task(st_task, _depth=_depth + 1)
                outputs.append(f"[{st_action[:30]}]({st_result.route})\n{st_result.output}")
                task.tokens_input += st_result.tokens_input
                task.tokens_output += st_result.tokens_output
                task.time_ms += st_result.time_ms
                task.cost_saved += st_result.cost_saved
            task.output = "\n\n---\n\n".join(outputs)
            task.route = "hybrid(recurse)"
            task.model_used = "mixed"
            return task

    result = call_cloud_api(task.action, task.text)
    task.output = result["text"]
    task.tokens_input = result.get("tokens_input", 0)
    task.tokens_output = result.get("tokens_output", 0)
    task.time_ms = result.get("time_ms", 0)
    task.cost_saved = 0
    auto_collect_on_cloud(task, result)
    return task


def _finalize_task(task: Task) -> None:
    """写入缓存、日志、审计"""
    if task.output and task.output.strip():
        cache.set(task.action, task.text,
                  {"text": task.output, "tokens_input": task.tokens_input, "tokens_output": task.tokens_output},
                  ttl_hours=CONFIG.get_cache_ttl(detect_task_type(task.action, PROMPT_TEMPLATES)))
    log_usage(task)

    try:
        from task_router.audit import get_audit_logger, AuditEvent
        audit = get_audit_logger()
        audit.log(AuditEvent(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            event_type="task_execution",
            action=_scrub_pii(task.action[:100]),
            details={"route": task.route, "model": task.model_used,
                     "tokens_input": task.tokens_input, "tokens_output": task.tokens_output,
                     "cost_saved": task.cost_saved, "output_length": len(task.output or "")},
            status="success" if task.route != "error" else "failure",
            duration_ms=task.time_ms,
        ))
    except Exception as e:
        log.warning("审计日志写入失败: %s", e)


# ─── 核心入口 ──────────────────────────────────────────────────


def run_task(task: Task, force_route: str = "", _depth: int = 0) -> Task:
    """执行单个任务（核心入口）"""
    if _depth > CONFIG.max_recurse_depth:
        task.output = f"[递归深度超限] 最大深度 {CONFIG.max_recurse_depth}"
        task.route = "error"
        return task

    # 1. 路由决策
    wt = _get_weight_tracker()
    routing_score = 0.0
    if force_route:
        task.route = force_route
        task.model_used = CONFIG.local_model if force_route == "local" else CONFIG.cloud_model
    else:
        decision = estimate_complexity(
            task, base_threshold=wt.get_weights().base_threshold,
            weights=wt.get_weights(),
        )
        task.route = decision["route"]
        task.model_used = CONFIG.local_model if decision["route"] == "local" else CONFIG.cloud_model
        routing_score = decision.get("score", 0.0)

    # 2. 语义缓存查找
    cached = _check_cache(task)
    if cached:
        log_usage(cached)
        return cached

    # 3. 执行
    if task.route == "local":
        task = _run_local(task)
    else:
        task = _run_cloud(task, force_route, _depth=_depth)

    # 4. 收尾
    _finalize_task(task)

    # 5. 学习反馈
    task_type = detect_task_type(task.action, PROMPT_TEMPLATES)
    success = task.route != "error" and bool(task.output and task.output.strip())
    wt.record_outcome(
        task_type=task_type, route=task.route,
        score=routing_score, success=success, local_model=task.model_used,
    )

    return task


# ─── 批量执行 ──────────────────────────────────────────────────


def run_batch(tasks_data: list[dict], concurrency: int = 1) -> list[Task]:
    """批量执行任务"""
    tasks = [Task(action=t.get("action", t.get("prompt", "")),
                  text=t.get("text", t.get("input", "")),
                  files=t.get("files", []))
             for t in tasks_data]

    if concurrency <= 1:
        return [run_task(t) for t in tasks]

    results: list[Optional[Task]] = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_idx = {executor.submit(run_task, t): i for i, t in enumerate(tasks)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = Task(action=tasks[idx].action, output=f"[错误] {e}", route="error")
    return [r for r in results if r is not None]
