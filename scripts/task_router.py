#!/usr/bin/env python3
"""
TaskRouter — 企业 LLM 网关

安全审计 · 智能路由 · 自动优化
统一管理所有 LLM 调用，三层决策融合确保路由准确，蒸馏闭环让系统越用越聪明。
模块化架构：config / routing / cache / models / prompts / validation / distillation / rules
"""

import os
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

log = logging.getLogger("task_router")

# ─── 模块导入 ──────────────────────────────────────────────────

from config import get_config, TASK_TO_CAPABILITY
from routing import (
    Task, estimate_complexity, detect_task_type,
    decompose_complex_task, _recursive_decompose,
)
from cache import SemanticCache
from models import call_ollama, call_cloud_api, circuit_breaker
from prompts import (
    PROMPT_TEMPLATES, build_optimized_prompt, get_max_tokens,
)
from rules import rule_execute
from validation import validate_local_output
from distillation import (
    DistillationStore, collect_distillation_pair,
    get_dynamic_examples,
)
from io_utils import read_jsonl, append_jsonl
from reasoning import (
    select_strategy, enhance_prompt_with_strategy,
    STRATEGY_TOKEN_MULTIPLIER, TOKEN_BUDGET, get_strategy_tracker,
)
from adaptive_compression import compress_adaptive
from confidence import extract_confidence, extract_confidence_from_text, CascadeDecision
from meta_learner import extract_routing_features, get_meta_learner, get_active_learner
from tqbc import TQBCRouter, extract_quantile_features, TokenQuantileFeatures
from outcome_cache import OutcomeAwareCache
from conformal_routing import ConformalizedRouter

# ─── 全局实例（延迟初始化）──────────────────────────────────────

CONFIG = get_config()
cache = SemanticCache(
    cache_dir=CONFIG.cache_dir,
    max_entries=CONFIG.cache_max_entries,
    fuzzy_threshold=CONFIG.cache_fuzzy_threshold,
)
store = DistillationStore(cache_dir=CONFIG.cache_dir)

# 模型注册表（延迟加载）
_model_registry = None
_log_lock = threading.Lock()

RECURSE_SCORE_MIN = CONFIG.recurse_score_min
RECURSE_SCORE_MAX = CONFIG.recurse_score_max
DEFAULT_CAPABILITY_THRESHOLD = CONFIG.base_threshold


def get_model_registry() -> Any:
    global _model_registry
    if _model_registry is None:
        from model_registry import ModelRegistry
        _model_registry = ModelRegistry(cache_dir=CONFIG.cache_dir)
        if not _model_registry.models:
            _model_registry.discover()
    return _model_registry


class CapabilityTracker:
    """自适应阈值追踪器"""

    LOOKBACK = 20

    def __init__(self, cache_dir: str):
        self.data_file = os.path.join(cache_dir, "capability_tracker.jsonl")
        os.makedirs(cache_dir, exist_ok=True)

    def record(self, capability: str, success: bool, score: float = 0.0, task_type: str = "") -> None:
        entry = {
            "capability": capability, "success": success, "score": score,
            "task_type": task_type, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        append_jsonl(self.data_file, entry)

    def get_success_rate(self, capability: str) -> float:
        records = self._load_recent(capability)
        if not records:
            return 0.8  # 默认
        return sum(1 for r in records if r.get("success")) / len(records)

    def get_adjusted_threshold(self, capability: str) -> float:
        rate = self.get_success_rate(capability)
        if rate >= 0.9:
            return DEFAULT_CAPABILITY_THRESHOLD - 0.5
        elif rate >= 0.7:
            return DEFAULT_CAPABILITY_THRESHOLD
        elif rate >= 0.5:
            return DEFAULT_CAPABILITY_THRESHOLD + 0.5
        else:
            return DEFAULT_CAPABILITY_THRESHOLD + 1.0

    def get_all_adjustments(self) -> dict:
        all_entries = read_jsonl(self.data_file)
        all_caps = {e.get("capability", "") for e in all_entries if e.get("capability")}
        result = {}
        for cap in sorted(all_caps):
            rate = self.get_success_rate(cap)
            threshold = self.get_adjusted_threshold(cap)
            diff = round(threshold - DEFAULT_CAPABILITY_THRESHOLD, 1)
            result[cap] = {
                "success_rate": round(rate, 2), "threshold": threshold,
                "adjustment": f"{'+' if diff > 0 else ''}{diff}",
                "direction": "上调" if diff > 0 else "下调" if diff < 0 else "不变",
            }
        return result

    def _load_recent(self, capability: str, n: int = LOOKBACK) -> list[dict]:
        all_entries = read_jsonl(self.data_file)
        return [e for e in all_entries if e.get("capability") == capability][-n:]


cap_tracker = CapabilityTracker(CONFIG.cache_dir)


# 可学习 A3M 权重
_weight_tracker = None


def _get_weight_tracker():
    global _weight_tracker
    if _weight_tracker is None:
        from weights import get_weight_tracker
        _weight_tracker = get_weight_tracker(CONFIG.cache_dir)
    return _weight_tracker


# 置信度门控级联
_cascade = None


def get_cascade():
    global _cascade
    if _cascade is None:
        _cascade = CascadeDecision(CONFIG.cache_dir)
    return _cascade


# TQBC（Token-Quantile Bayesian Cascade）路由
_tqbc = None


def get_tqbc():
    global _tqbc
    if _tqbc is None:
        _tqbc = TQBCRouter(CONFIG.cache_dir)
    return _tqbc


# 结果感知缓存（OATS-Inspired Quality Cache）
_outcome_cache = None


def get_outcome_cache():
    global _outcome_cache
    if _outcome_cache is None:
        _outcome_cache = OutcomeAwareCache(CONFIG.cache_dir)
    return _outcome_cache


# Conformalized Router（不确定性量化路由）
_conformal = None


def get_conformal_router():
    global _conformal
    if _conformal is None:
        _conformal = ConformalizedRouter(CONFIG.cache_dir)
    return _conformal


# ─── 预处理/后处理 ──────────────────────────────────────────────

def preprocess_text(text: str, max_chars: int = 800) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if "," in text and "\n" not in text:
        items = [x.strip() for x in text.split(",") if x.strip()]
        if len(items) > 3:
            text = "\n".join(items)
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n... (截断，共 {len(text)} 字符)"
    return text


def postprocess_output(output: str, task_type: str = "") -> str:
    if not output:
        return ""
    output = output.strip()
    # 移除模型可能添加的前缀
    for prefix in ["输出：", "结果：", "答案：", "Output:", "Result:", "Answer:"]:
        if output.startswith(prefix):
            output = output[len(prefix):].strip()
    # 情感分析特殊处理
    if task_type == "sentiment":
        if "正面" in output and "负面" not in output:
            return "正面"
        elif "负面" in output and "正面" not in output:
            return "负面"
    return output


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


# ─── 使用日志 ──────────────────────────────────────────────────

def _scrub_pii(text: str) -> str:
    """对日志中的文本进行 PII 脱敏（延迟导入避免循环依赖）"""
    try:
        from privacy import get_privacy_filter
        result = get_privacy_filter().anonymize(text)
        return result.text
    except Exception:
        return text


def log_usage(task: Task) -> None:
    log_file = os.path.join(CONFIG.cache_dir, "usage.jsonl")
    entry = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "date": time.strftime("%Y-%m-%d"),
        "action": _scrub_pii(task.action[:100]),
        "route": task.route,
        "model": task.model_used,
        "tokens_input": task.tokens_input,
        "tokens_output": task.tokens_output,
        "cost_saved": task.cost_saved,
        "time_ms": task.time_ms,
    }
    with _log_lock:
        append_jsonl(log_file, entry, max_lines=10000)


def calc_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1000 * CONFIG.cost_per_1k_input
            + output_tokens / 1000 * CONFIG.cost_per_1k_output)


def calc_savings(task: Task) -> float:
    if task.route != "local":
        return 0.0
    return round(calc_cost(task.tokens_input, task.tokens_output), 6)


# ─── 蒸馏辅助 ──────────────────────────────────────────────────

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


# ─── 核心执行（拆分为子函数）────────────────────────────────────

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
        # 极低质量的缓存条目，跳过（触发正常管道）
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
        try:
            result = call_ollama(st_prompt, max_tokens=get_max_tokens(st_type))
        except Exception as e:
            log.error("子任务 %s 模型调用失败: %s", st_type, e)
            outputs.append(f"[{st_type}]\n[错误: {e}]")
            continue
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

    # 规则执行（规则引擎 100% 置信度，无需 logprobs）
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

    # 获取动态示例（只查询一次，同时用于 prompt 注入和策略选择）
    examples_str = ""
    has_examples = False
    examples = []
    capability = TASK_TO_CAPABILITY.get(task_type, "")
    if capability:
        examples = get_dynamic_examples(capability, store, limit=2)
    if examples:
        has_examples = True
        example_block = "\n".join(
            f"示例: {e.get('prompt', '')[:50]} → {e.get('response', '')[:50]}"
            for e in examples
        )
        prompt = f"{example_block}\n\n{prompt}"
        examples_str = "\n".join(
            f"{e.get('prompt', '')[:50]} → {e.get('response', '')[:50]}"
            for e in examples
        )

    # 推理策略选择：统一入口（关键词 + Token 分位数 + 历史反馈）
    routing_result = estimate_complexity(task)

    # 第一次选择：用关键词 + 复杂度（模型调用前没有 logprobs）
    strategy_decision = select_strategy(
        action=clean_action,
        text=clean_text,
        complexity_score=routing_result["score"],
        task_type=task_type,
        logprobs=None,  # 调用前没有 logprobs
        has_examples=has_examples,
    )
    strategy = strategy_decision.strategy

    # 策略反馈优化：利用历史数据选择最优策略
    tracker = get_strategy_tracker()
    historical_best = tracker.get_best_strategy(task_type)
    if historical_best and historical_best != strategy:
        log.debug("策略反馈优化: %s → %s (历史数据)", strategy, historical_best)
        strategy = historical_best

    # 蒸馏示例 → few_shot
    if has_examples and strategy != "direct":
        strategy = "few_shot"

    prompt = enhance_prompt_with_strategy(prompt, strategy, examples_str)

    # 自适应 Prompt 压缩：根据策略预估置信度选择压缩级别
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

    # 多模型级联：简单任务用快速模型，复杂任务用主力模型
    if not selected_model:
        if strategy == "direct" and CONFIG.speed_model:
            selected_model = CONFIG.speed_model
            log.debug("速度优化: 简单任务用快速模型 %s", CONFIG.speed_model)
        else:
            selected_model = None  # 使用默认 local_model

    # 请求 logprobs 用于置信度提取
    try:
        result = call_ollama(prompt, model=selected_model, max_tokens=max_tokens, with_logprobs=True)
    except Exception as e:
        log.error("模型调用失败: %s", e)
        return {
            "text": "", "tokens_input": 0, "tokens_output": 0, "time_ms": 0,
            "confidence": {"confidence": 0.0, "entropy": 1.0, "margin": 0.0},
            "strategy": strategy, "error": str(e),
        }

    # 提取置信度信号
    result["strategy"] = strategy
    logprobs = result.get("logprobs", [])
    if logprobs:
        result["confidence"] = extract_confidence(logprobs)
    else:
        # Ollama 不支持 logprobs 时降级到启发式
        result["confidence"] = extract_confidence_from_text(result["text"])

    # 模型调用后重新评估：用 Token 分位数特征选择最优策略
    adaptive_decision = select_strategy(
        action=clean_action,
        text=clean_text,
        complexity_score=routing_result["score"],
        task_type=task_type,
        logprobs=logprobs,  # 现在有 logprobs 了
        has_examples=has_examples,
    )
    result["adaptive_strategy"] = adaptive_decision.strategy
    result["adaptive_confidence"] = adaptive_decision.confidence_signal
    result["token_budget_factor"] = adaptive_decision.token_budget_factor

    # Token 预算优化评估：记录实际 vs 最优 token 使用
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
            # 熔断中：保留原始本地输出，不追加警告到 output（避免污染缓存）
            task.route = "local(degraded)"
        else:
            auto_collect_on_local_failure(task, task.output, cloud_result["text"], "quality_fallback")
            task.output = cloud_result["text"]
            task.route = "cloud_fallback"
            task.model_used = CONFIG.cloud_model
            result = cloud_result
            task.cost_saved = 0
    else:
        # 无云端 API：保留原始输出，不追加警告（避免污染缓存）
        pass

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

    # 规则引擎命中时直接返回
    if task.model_used == "rule_engine":
        return task

    # ── 四层决策：Cascade + Meta-Learner + Active Learner + TQBC ──
    cascade = get_cascade()
    conf_data = result.get("confidence", {})
    cascade_decision = cascade.should_escalate(conf_data)

    # Meta-Learner：统一所有信号做全局决策
    ml = get_meta_learner()
    al = get_active_learner()
    routing_decision = estimate_complexity(task)
    cap = TASK_TO_CAPABILITY.get(task_type, "")
    cap_success = cap_tracker.get_success_rate(cap) if cap else 0.5
    features = extract_routing_features(
        complexity_score=routing_decision["score"],
        confidence_data=conf_data,
        text_length=len(clean_text),
        file_count=len(task.files),
        capability_success_rate=cap_success,
        strategy=result.get("strategy", "direct"),
    )
    ml_prediction = ml.predict(features)

    # 主动学习：不确定的任务类型请求云端验证（带冷启动保护）
    active_verify = al.should_request_verification(task_type, min_samples=5) and CONFIG.cloud_api_key

    # TQBC：Token-Quantile Bayesian Cascade（第四层创新信号）
    tqbc = get_tqbc()
    logprobs = result.get("logprobs", [])
    tqbc_decision = tqbc.decide(
        logprobs=logprobs,
        complexity_score=routing_decision["score"],
        task_type=task_type,
        text_length=len(clean_text),
        capability_success_rate=cap_success,
    )

    # 决策融合：四个信号投票
    # - cascade: 基于置信度阈值（原始信号）
    # - ml_prediction: 基于全局特征的 P(本地成功)（原始信号）
    # - active_verify: 基于不确定性（原始信号）
    # - tqbc_decision: 基于 Token 分位数 + Thompson Sampling + 贝叶斯校准（创新信号）
    raw_should_escalate = (
        cascade_decision["escalate"]
        or (not ml_prediction["should_use_local"] and ml_prediction["confidence"] > 0.5 and ml.get_stats()["total"] >= 50)
        or active_verify
        or tqbc_decision.should_escalate
    )

    # 第五层：Conformalized Router（不确定性量化）
    # 将点估计转换为带统计覆盖保证的预测集合
    # 使用原始 logprob 特征（不依赖在线学习），满足 exchangeability
    quantile_features = extract_quantile_features(logprobs) if logprobs else TokenQuantileFeatures()
    conformal = get_conformal_router()
    conformal_decision = conformal.decide(
        cascade_decision=cascade_decision,
        tqbc_decision=tqbc_decision,
        ml_prediction=ml_prediction,
        active_verify=active_verify,
        features=features,
        task_type=task_type,
        raw_should_escalate=raw_should_escalate,
        quantile_features=quantile_features,
    )
    should_escalate = conformal_decision.should_escalate

    def _record_and_return(escalated: bool, cloud_used: bool = False) -> None:
        """统一记录所有学习信号（避免提前 return 跳过学习）"""
        local_success = not cloud_used and task.route != "cloud_fallback" and task.route != "error"
        cascade.record_outcome(conf_data, was_correct=local_success or cloud_used, escalated=escalated)
        ml.record_and_learn(features, success=local_success, route=task.route, task_type=task_type)
        al.record(task_type, ml_prediction["local_success_prob"], local_success)
        tqbc.record_outcome(
            decision=tqbc_decision,
            success=local_success or cloud_used,
            escalated=escalated,
            task_type=task_type,
        )
        conformal.record_outcome(
            decision=conformal_decision,
            success=local_success or cloud_used,
            escalated=escalated,
            task_type=task_type,
            features=features,
        )

        # 策略反馈记录：追踪推理策略效果
        used_strategy = result.get("strategy", "direct")
        get_strategy_tracker().record(used_strategy, task_type, local_success or cloud_used)

        # 结果感知缓存反馈：更新缓存条目质量分
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
        # 云端熔断中，降级使用本地结果
        cascade_escalated = False
    else:
        cascade_escalated = False

    # 输出验证 + 云端降级
    result = _validate_and_fallback(task, result, task_type)
    task.tokens_input = result.get("tokens_input", 0)
    task.tokens_output = result.get("tokens_output", 0)
    task.time_ms = result.get("time_ms", 0)
    task.cost_saved = calc_savings(task)

    # 统一记录学习信号
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

    # 审计日志（PII 脱敏）
    try:
        from audit import get_audit_logger, AuditEvent
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


def run_task(task: Task, force_route: str = "", _depth: int = 0) -> Task:
    """执行单个任务（核心入口）"""
    if _depth > CONFIG.max_recurse_depth:
        task.output = f"[递归深度超限] 最大深度 {CONFIG.max_recurse_depth}"
        task.route = "error"
        return task
    # 1. 路由决策（使用可学习权重）
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

    # 4. 收尾（缓存、日志、审计）
    _finalize_task(task)

    # 5. 学习反馈
    task_type = detect_task_type(task.action, PROMPT_TEMPLATES)
    success = task.route != "error" and bool(task.output and task.output.strip())
    wt.record_outcome(
        task_type=task_type,
        route=task.route,
        score=routing_score,
        success=success,
        local_model=task.model_used,
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


# ─── 预估和分类 ──────────────────────────────────────────────────

def estimate(task_description: str) -> dict:
    dummy = Task(action=task_description)
    decision = estimate_complexity(dummy, base_threshold=CONFIG.base_threshold)
    est_input = max(50, len(task_description) // 2)
    est_output = max(50, len(task_description))
    return {
        "task": task_description[:100],
        "suggested_route": decision["route"],
        "reason": decision["reason"],
        "score": decision["score"],
        "estimated_cloud_cost": f"${calc_cost(est_input, est_output):.6f}",
        "will_save": decision["route"] == "local",
    }


def classify_task(task_desc: str, text: str = "") -> dict:
    """细粒度任务分类"""
    task = Task(action=task_desc, text=text)
    decision = estimate_complexity(task, base_threshold=CONFIG.base_threshold)
    task_type = detect_task_type(task_desc, PROMPT_TEMPLATES)

    return {
        "task": task_desc[:100],
        "task_type": task_type,
        "verdict": decision["route"],
        "score": decision["score"],
        "reason": decision["reason"],
        "confidence": "high" if abs(decision["score"] - CONFIG.base_threshold) > 2 else "medium",
        "text_length": len(text),
    }


# ─── 统计 ──────────────────────────────────────────────────────

def _classify_route(route: str) -> str:
    """路由分类: local / cache / cloud"""
    if route == "local":
        return "local"
    if "cache" in route:
        return "cache"
    return "cloud"


def show_usage_stats() -> str:
    entries = read_jsonl(os.path.join(CONFIG.cache_dir, "usage.jsonl"))
    if not entries:
        return "暂无使用记录"

    totals = {"local": 0, "cloud": 0, "cache": 0}
    total_input = total_output = 0
    total_saved = 0.0
    daily: dict[str, dict] = {}

    for e in entries:
        cat = _classify_route(e.get("route", ""))
        totals[cat] += 1
        total_input += e.get("tokens_input", 0)
        total_output += e.get("tokens_output", 0)
        total_saved += e.get("cost_saved", 0)

        day = e.get("date", e.get("time", "")[:10])
        if day:
            if day not in daily:
                daily[day] = {"local": 0, "cloud": 0, "cache": 0, "saved": 0.0}
            daily[day][cat] += 1
            daily[day]["saved"] += e.get("cost_saved", 0)

    cs = cache.stats()
    result = (
        f"TaskRouter 使用统计\n{'='*40}\n"
        f"本地调用: {totals['local']} 次\n云端调用: {totals['cloud']} 次\n缓存命中: {totals['cache']} 次\n"
        f"总输入 tokens: {total_input:,}\n总输出 tokens: {total_output:,}\n"
        f"理论最大节约: ${total_saved:.4f} (若全部走云端需支付的费用)\n"
    )
    if cs["total"] > 0:
        result += f"\n语义缓存:\n  缓存条目: {cs['total']}\n  缓存节约: ${cs['estimated_cost_saved']:.4f}\n"

    for day in sorted(daily, reverse=True)[:7]:
        d = daily[day]
        result += f"  {day}: {d['local']+d['cloud']+d['cache']}次 (本地{d['local']}/云端{d['cloud']}/缓存{d['cache']}) 省${d['saved']:.4f}\n"

    if circuit_breaker.state == "open":
        result += f"\n云端熔断器: 🔴 熔断中 ({circuit_breaker.remaining_seconds()}秒后恢复)\n"
    elif circuit_breaker.state == "half_open":
        result += "\n云端熔断器: 🟡 半开状态 (探测中)\n"
    elif circuit_breaker.failures > 0:
        result += "\n云端熔断器: 🟢 已恢复\n"

    return result


# ─── 任务计划执行 ──────────────────────────────────────────────

DECOMPOSE_TEMPLATES: dict[str, dict] = {
    "电商数据分析": {
        "match": ["电商", "销售", "商品", "订单"],
        "subtasks": ["清洗数据（去空值、统一格式）", "按类别分类商品", "统计各品类销售额", "分析销售趋势和异常", "给出优化建议并生成报告"],
        "routes": ["local", "local", "local", "cloud", "cloud"],
    },
    "文件批量整理": {
        "match": ["文件", "桌面", "整理", "归类"],
        "subtasks": ["扫描并列出所有文件", "按扩展名分类", "建议目录结构", "生成整理脚本"],
        "routes": ["local", "local", "local", "cloud"],
    },
}


def decompose_task(task_description: str, text_content: str = "") -> dict:
    """将大任务拆解为子任务列表"""
    desc = task_description.lower()
    for template_name, template in DECOMPOSE_TEMPLATES.items():
        if any(kw in desc for kw in template["match"]):
            subtasks = [{"id": i + 1, "action": sa, "text": text_content, "route": r}
                        for i, (sa, r) in enumerate(zip(template["subtasks"], template["routes"]))]
            return {"task": task_description[:200], "template": template_name,
                    "total_subtasks": len(subtasks),
                    "local_count": sum(1 for s in subtasks if s["route"] == "local"),
                    "cloud_count": sum(1 for s in subtasks if s["route"] == "cloud"),
                    "subtasks": subtasks}

    # 自动检测
    cl = classify_task(task_description, text_content)
    return {"task": task_description[:200], "template": "auto", "total_subtasks": 1,
            "local_count": 1 if cl["verdict"] == "local" else 0,
            "cloud_count": 1 if cl["verdict"] == "cloud" else 0,
            "subtasks": [{"id": 1, "action": task_description[:200], "text": text_content,
                          "route": cl["verdict"], "reason": cl["reason"]}]}


def execute_plan(plan: dict) -> dict:
    """执行任务计划"""
    results: list[dict] = []
    total_input = total_output = total_time = 0
    total_saved = 0.0

    for i, step in enumerate(plan.get("subtasks", [])):
        task = Task(action=step["action"], text=step.get("text", ""))
        task = run_task(task, force_route=step.get("route", ""))
        total_input += task.tokens_input
        total_output += task.tokens_output
        total_time += task.time_ms
        total_saved += task.cost_saved
        results.append({"id": step.get("id", i + 1), "action": step["action"],
                        "route": task.route, "output": task.output,
                        "tokens_input": task.tokens_input, "tokens_output": task.tokens_output,
                        "time_ms": task.time_ms, "cost_saved": task.cost_saved})

    return {"task": plan.get("task", ""), "total_steps": len(results),
            "local_steps": sum(1 for r in results if r["route"] == "local"),
            "cloud_steps": sum(1 for r in results if "cloud" in r["route"]),
            "total_tokens_input": total_input, "total_tokens_output": total_output,
            "total_time_ms": total_time, "total_cost_saved": round(total_saved, 6), "steps": results}


# ─── CLI 入口已拆分到 cli.py ─────────────────────────────────────
# 运行 CLI: python cli.py --task "..." 或 python -m cli
