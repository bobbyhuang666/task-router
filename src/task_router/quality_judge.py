"""
LLM-as-Judge 多维度质量评估器

用 LLM 评估路由决策和输出质量，替代简单的规则检查。
为 Self-Reflective Routing (SRR) 的反思引擎提供评估数据。

与 distillation.py 的 QualityEvaluator 的区别：
- QualityEvaluator: 规则-based，评估蒸馏对质量（内部模块）
- QualityJudge: LLM-based，评估路由决策 + 输出质量（语义级别）

评估维度：
1. relevance:     输出与任务的相关性 [0-10]
2. completeness:  输出的完整性 [0-10]
3. accuracy:      输出的准确性 [0-10]
4. efficiency:    路由决策的效率 [0-10]
5. correctness:   路由选择的正确性 [0-10]
"""

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from task_router.io_utils import read_jsonl, append_jsonl

log = logging.getLogger(__name__)


# ─── 评分维度定义 ──────────────────────────────────────────────


QUALITY_DIMENSIONS = [
    "relevance",
    "completeness",
    "accuracy",
    "efficiency",
    "correctness",
]

ROUTING_ERROR_TYPES = [
    "none",               # 路由正确
    "over_escalated",     # 该走本地但走了云端（浪费钱）
    "under_escalated",    # 该走云端但走了本地（质量差）
    "wrong_strategy",     # 策略选择不当
    "token_waste",        # token 消耗过多
]

STRATEGY_OPTIONS = ["direct", "cod", "cot", "few_shot", "structured"]
ROUTE_OPTIONS = ["local", "cloud"]


# ─── Judge Prompt 模板 ──────────────────────────────────────────


SINGLE_EPISODE_PROMPT = """你是一个 LLM 路由系统的质量评估专家。
请评估以下路由决策和输出质量。

## 任务信息
- 任务描述: {action}
- 任务类型: {task_type}
- 路由决策: {route}
- 使用模型: {model_used}
- 推理策略: {strategy}
- 复杂度评分: {complexity_score}
- Token 消耗: 输入 {tokens_input}, 输出 {tokens_output}

## 输入文本
{text}

## 系统输出
{output}

## 要求

请用 JSON 格式输出评分（每项 0-10 分）：
{{
    "relevance": <0-10>,       // 输出是否正确回应了任务需求
    "completeness": <0-10>,    // 输出是否完整（不是半成品）
    "accuracy": <0-10>,        // 输出内容是否准确
    "efficiency": <0-10>,      // 路由和策略选择是否经济高效
    "correctness": <0-10>,     // 路由选择是否正确
    "optimal_route": "<local|cloud>",
    "optimal_strategy": "<direct|cod|cot|few_shot|structured>",
    "routing_error": "<none|over_escalated|under_escalated|wrong_strategy|token_waste>",
    "notes": "<一句话说明>"
}}

只输出 JSON，不要其他文字。"""


BATCH_EPISODE_PROMPT = """你是一个 LLM 路由系统的质量评估专家。
请逐一评估以下 {count} 个路由决策和输出质量。

{episodes_text}

## 要求

对每个 episode 用 JSON 格式输出评分（每项 0-10 分），返回一个 JSON 数组：
[
    {{
        "episode_id": "<episode_id>",
        "relevance": <0-10>,
        "completeness": <0-10>,
        "accuracy": <0-10>,
        "efficiency": <0-10>,
        "correctness": <0-10>,
        "optimal_route": "<local|cloud>",
        "optimal_strategy": "<direct|cod|cot|few_shot|structured>",
        "routing_error": "<none|over_escalated|under_escalated|wrong_strategy|token_waste>",
        "notes": "<一句话说明>"
    }},
    ...
]

只输出 JSON 数组，不要其他文字。"""


SINGLE_EPISODE_TEMPLATE = """### Episode {idx}: {episode_id}
- 任务: {action}
- 类型: {task_type} | 路由: {route} | 模型: {model_used} | 策略: {strategy}
- 复杂度: {complexity_score} | Token: {tokens_input}+{tokens_output}
- 输入: {text}
- 输出: {output}
"""


# ─── 评分数据结构 ──────────────────────────────────────────────


@dataclass
class QualityScores:
    """QualityJudge 的评估结果"""

    episode_id: str = ""
    timestamp: str = ""

    # 五维度评分 (0-10)
    relevance: float = 0.0
    completeness: float = 0.0
    accuracy: float = 0.0
    efficiency: float = 0.0
    correctness: float = 0.0

    # 最优决策建议
    optimal_route: str = ""
    optimal_strategy: str = ""
    routing_error: str = "none"
    notes: str = ""

    # 元数据
    judge_method: str = ""  # "llm" / "fallback"
    raw_response: str = ""

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "timestamp": self.timestamp,
            "relevance": self.relevance,
            "completeness": self.completeness,
            "accuracy": self.accuracy,
            "efficiency": self.efficiency,
            "correctness": self.correctness,
            "optimal_route": self.optimal_route,
            "optimal_strategy": self.optimal_strategy,
            "routing_error": self.routing_error,
            "notes": self.notes,
            "judge_method": self.judge_method,
        }

    @property
    def overall(self) -> float:
        """加权综合分 (0-1)"""
        return (
            0.20 * self.relevance
            + 0.20 * self.completeness
            + 0.25 * self.accuracy
            + 0.15 * self.efficiency
            + 0.20 * self.correctness
        ) / 10.0

    @property
    def route_correct(self) -> bool:
        """路由是否正确"""
        return self.routing_error in ("none", "") and self.correctness >= 6.0


# ─── LLM 调用接口 ──────────────────────────────────────────────


def _default_llm_caller(prompt: str) -> str:
    """默认 LLM 调用者 — 通过 TaskRouter 的云端 API 调用。

    可通过 QualityJudge.set_llm_caller() 替换为其他实现。
    """
    from task_router.models import call_cloud_api
    result = call_cloud_api(prompt)
    return result.get("text", "")


def _parse_single_judgment(response: str, episode_id: str = "") -> dict:
    """解析 LLM 返回的单条评分 JSON。

    容错处理：尝试提取 JSON 子串、处理 markdown 代码块。
    """
    text = response.strip()

    # 处理 markdown 代码块
    if text.startswith("```"):
        lines = text.split("\n")
        json_lines = []
        in_block = False
        for line in lines:
            if line.strip().startswith("```") and not in_block:
                in_block = True
                continue
            elif line.strip() == "```" and in_block:
                break
            elif in_block:
                json_lines.append(line)
        text = "\n".join(json_lines)

    # 尝试直接解析
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            if episode_id:
                data["episode_id"] = episode_id
            return data
    except json.JSONDecodeError:
        pass

    # 尝试提取 JSON 子串
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start:end])
            if isinstance(data, dict):
                if episode_id:
                    data["episode_id"] = episode_id
                return data
        except json.JSONDecodeError:
            pass

    return {}


def _parse_batch_judgment(response: str) -> list[dict]:
    """解析 LLM 返回的批量评分 JSON 数组。"""
    text = response.strip()

    # 处理 markdown 代码块
    if text.startswith("```"):
        lines = text.split("\n")
        json_lines = []
        in_block = False
        for line in lines:
            if line.strip().startswith("```") and not in_block:
                in_block = True
                continue
            elif line.strip() == "```" and in_block:
                break
            elif in_block:
                json_lines.append(line)
        text = "\n".join(json_lines)

    # 尝试直接解析
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # 尝试提取 JSON 数组
    start = text.find("[")
    end = text.rfind("]") + 1
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start:end])
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    return []


def _clamp_score(value: Any, min_val: float = 0.0, max_val: float = 10.0) -> float:
    """将评分限制在 [min_val, max_val] 范围内"""
    try:
        v = float(value)
        return max(min_val, min(max_val, v))
    except (TypeError, ValueError):
        return 0.0


def _scores_from_judgment(data: dict, episode_id: str = "") -> QualityScores:
    """将解析后的 JSON dict 转为 QualityScores 对象"""
    return QualityScores(
        episode_id=data.get("episode_id", episode_id),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        relevance=_clamp_score(data.get("relevance")),
        completeness=_clamp_score(data.get("completeness")),
        accuracy=_clamp_score(data.get("accuracy")),
        efficiency=_clamp_score(data.get("efficiency")),
        correctness=_clamp_score(data.get("correctness")),
        optimal_route=data.get("optimal_route", ""),
        optimal_strategy=data.get("optimal_strategy", ""),
        routing_error=data.get("routing_error", "none"),
        notes=data.get("notes", ""),
    )


# ─── Fallback 评估（不需要 LLM）──────────────────────────────────


def _fallback_judge(episode: dict) -> QualityScores:
    """当 LLM 不可用时的降级评估。

    使用规则-based 方法，结合 episode 中已有信号做近似评估。
    """
    from task_router.validation import validate_local_output

    output = episode.get("output", "")
    task_type = episode.get("task_type", "")
    route = episode.get("route", "")
    confidence_data = episode.get("confidence_data", {})

    # 结构验证
    validation = validate_local_output(output, task_type)
    completeness = 8.0 if validation["valid"] else 3.0
    if validation.get("signals"):
        completeness -= min(5.0, len(validation["signals"]) * 1.5)

    # 相关性：基于输出长度和失败信号
    relevance = 7.0
    if any(s in output for s in ["抱歉", "无法", "不能", "Error"]):
        relevance = 2.0

    # 准确性：基于置信度信号
    confidence = confidence_data.get("confidence", 0.5)
    accuracy = 3.0 + confidence * 7.0  # 映射到 [3, 10]

    # 效率：基于 token 消耗
    tokens = episode.get("tokens_input", 0) + episode.get("tokens_output", 0)
    if route.startswith("cache"):
        efficiency = 10.0
    elif route.startswith("local"):
        efficiency = 8.0 if tokens < 500 else 6.0
    else:
        efficiency = 5.0  # 云端消耗更高

    # 正确性：基于路由和置信度
    if route.startswith("cache"):
        correctness = 9.0
    elif route.startswith("local") and confidence >= 0.5:
        correctness = 7.0
    elif route.startswith("cloud") and confidence < 0.3:
        correctness = 7.0  # 正确地升级到云端
    elif route.startswith("local") and confidence < 0.3:
        correctness = 3.0  # 不应该走本地
    else:
        correctness = 5.0

    # 路由错误判断
    routing_error = "none"
    if route.startswith("local") and confidence < 0.3 and relevance < 4.0:
        routing_error = "under_escalated"
    elif route.startswith("cloud") and confidence >= 0.7 and completeness >= 7.0:
        routing_error = "over_escalated"

    return QualityScores(
        episode_id=episode.get("episode_id", ""),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        relevance=round(relevance, 1),
        completeness=round(completeness, 1),
        accuracy=round(accuracy, 1),
        efficiency=round(efficiency, 1),
        correctness=round(correctness, 1),
        optimal_route="local" if confidence >= 0.5 else "cloud",
        optimal_strategy="",
        routing_error=routing_error,
        notes="fallback 规则评估",
        judge_method="fallback",
    )


# ─── QualityJudge 主类 ──────────────────────────────────────────


class QualityJudge:
    """LLM-as-Judge 多维度质量评估器。

    评估维度：
    1. relevance:    输出与任务的相关性 [0-10]
    2. completeness: 输出的完整性 [0-10]
    3. accuracy:     输出的准确性 [0-10]
    4. efficiency:   路由决策的效率 [0-10]
    5. correctness:  路由选择的正确性 [0-10]

    使用方式：
        judge = QualityJudge(cache_dir="/path/to/cache")
        scores = judge.judge_episode(episode_dict)

        # 或注入自定义 LLM（测试时用 mock）
        judge.set_llm_caller(mock_llm_func)
    """

    def __init__(self, cache_dir: str, use_llm: bool = True):
        self.judgments_file = os.path.join(cache_dir, "quality_judgments.jsonl")
        self._llm_caller: Optional[Callable[[str], str]] = None
        self._use_llm = use_llm
        self._lock = threading.Lock()

    def set_llm_caller(self, caller: Callable[[str], str]) -> None:
        """注入自定义 LLM 调用者（用于测试或自定义后端）"""
        self._llm_caller = caller

    def _call_llm(self, prompt: str) -> str:
        """调用 LLM（优先自定义调用者，其次默认云端 API）"""
        if self._llm_caller:
            return self._llm_caller(prompt)
        return _default_llm_caller(prompt)

    # ── 单条评估 ──

    def judge_episode(self, episode: dict) -> QualityScores:
        """评估单个 episode。

        参数:
            episode: Episode.to_dict() 的输出

        返回:
            QualityScores 对象
        """
        episode_id = episode.get("episode_id", "")

        # 检查是否已评估
        existing = self.get_judgment(episode_id)
        if existing:
            return existing

        if not self._use_llm:
            scores = _fallback_judge(episode)
            self._save_judgment(scores)
            return scores

        # 构造 prompt
        prompt = SINGLE_EPISODE_PROMPT.format(
            action=episode.get("action", ""),
            task_type=episode.get("task_type", "未知"),
            route=episode.get("route", ""),
            model_used=episode.get("model_used", ""),
            strategy=episode.get("strategy", ""),
            complexity_score=episode.get("complexity_score", 0),
            tokens_input=episode.get("tokens_input", 0),
            tokens_output=episode.get("tokens_output", 0),
            text=episode.get("text", "(无)"),
            output=episode.get("output", "(无输出)"),
        )

        try:
            response = self._call_llm(prompt)
            data = _parse_single_judgment(response, episode_id)
            if data:
                scores = _scores_from_judgment(data, episode_id)
                scores.judge_method = "llm"
                scores.raw_response = response
                self._save_judgment(scores)
                return scores
        except Exception as e:
            log.warning("LLM 评估失败，降级到 fallback: %s", e)

        # 降级
        scores = _fallback_judge(episode)
        self._save_judgment(scores)
        return scores

    # ── 批量评估 ──

    def judge_batch(self, episodes: list[dict], batch_size: int = 5) -> list[QualityScores]:
        """批量评估 episodes。

        将 episodes 分组，每组 batch_size 个合并到一个 prompt 中评估。
        已评估的 episode 自动跳过。

        参数:
            episodes: Episode dict 列表
            batch_size: 每批评估的 episode 数量

        返回:
            QualityScores 列表
        """
        # 过滤已评估
        unjudged = self.get_unjudged(episodes)
        if not unjudged:
            return [self.get_judgment(e["episode_id"]) for e in episodes
                    if self.get_judgment(e.get("episode_id", ""))]

        all_scores: list[QualityScores] = []

        # 已评估的直接加入
        for e in episodes:
            existing = self.get_judgment(e.get("episode_id", ""))
            if existing:
                all_scores.append(existing)

        # 分批评估未评估的
        for i in range(0, len(unjudged), batch_size):
            batch = unjudged[i:i + batch_size]

            if not self._use_llm or len(batch) == 1:
                # 单条或无 LLM：逐条评估
                for ep in batch:
                    scores = self.judge_episode(ep)
                    all_scores.append(scores)
                continue

            # 批量 LLM 评估
            episodes_text = ""
            for idx, ep in enumerate(batch, 1):
                episodes_text += SINGLE_EPISODE_TEMPLATE.format(
                    idx=idx,
                    episode_id=ep.get("episode_id", ""),
                    action=ep.get("action", ""),
                    task_type=ep.get("task_type", ""),
                    route=ep.get("route", ""),
                    model_used=ep.get("model_used", ""),
                    strategy=ep.get("strategy", ""),
                    complexity_score=ep.get("complexity_score", 0),
                    tokens_input=ep.get("tokens_input", 0),
                    tokens_output=ep.get("tokens_output", 0),
                    text=ep.get("text", "(无)"),
                    output=ep.get("output", "(无输出)"),
                )

            prompt = BATCH_EPISODE_PROMPT.format(
                count=len(batch),
                episodes_text=episodes_text,
            )

            try:
                response = self._call_llm(prompt)
                judgments = _parse_batch_judgment(response)
                for jdata in judgments:
                    eid = jdata.get("episode_id", "")
                    scores = _scores_from_judgment(jdata, eid)
                    scores.judge_method = "llm_batch"
                    scores.raw_response = response
                    self._save_judgment(scores)
                    all_scores.append(scores)
            except Exception as e:
                log.warning("批量 LLM 评估失败，逐条降级: %s", e)
                for ep in batch:
                    scores = self.judge_episode(ep)
                    all_scores.append(scores)

        return all_scores

    # ── 存储 ──

    def _save_judgment(self, scores: QualityScores) -> None:
        """保存评分到磁盘"""
        with self._lock:
            append_jsonl(self.judgments_file, scores.to_dict())

    def get_judgment(self, episode_id: str) -> Optional[QualityScores]:
        """获取指定 episode 的已有评分"""
        if not episode_id:
            return None
        for entry in read_jsonl(self.judgments_file):
            if entry.get("episode_id") == episode_id:
                return QualityScores(
                    episode_id=entry.get("episode_id", ""),
                    timestamp=entry.get("timestamp", ""),
                    relevance=_clamp_score(entry.get("relevance")),
                    completeness=_clamp_score(entry.get("completeness")),
                    accuracy=_clamp_score(entry.get("accuracy")),
                    efficiency=_clamp_score(entry.get("efficiency")),
                    correctness=_clamp_score(entry.get("correctness")),
                    optimal_route=entry.get("optimal_route", ""),
                    optimal_strategy=entry.get("optimal_strategy", ""),
                    routing_error=entry.get("routing_error", "none"),
                    notes=entry.get("notes", ""),
                    judge_method=entry.get("judge_method", ""),
                )
        return None

    def get_unjudged(self, episodes: list[dict]) -> list[dict]:
        """筛选尚未评估的 episode"""
        judged_ids = {e.get("episode_id") for e in read_jsonl(self.judgments_file)}
        return [e for e in episodes if e.get("episode_id") not in judged_ids]

    def get_all_judgments(self) -> list[dict]:
        """获取所有评分记录"""
        return read_jsonl(self.judgments_file)

    def count(self) -> int:
        """获取评分总数"""
        return len(read_jsonl(self.judgments_file))

    def clear(self) -> None:
        """清空所有评分（测试用）"""
        from task_router.io_utils import write_jsonl as _write
        with self._lock:
            _write(self.judgments_file, [])


# ─── 全局实例 ──────────────────────────────────────────────────


_judge: Optional[QualityJudge] = None
_judge_lock = threading.Lock()


def get_quality_judge(cache_dir: Optional[str] = None, use_llm: bool = True) -> QualityJudge:
    """获取全局 QualityJudge 实例"""
    global _judge
    if _judge is None:
        with _judge_lock:
            if _judge is None:
                if cache_dir is None:
                    from task_router.config import get_config
                    cache_dir = get_config().cache_dir
                _judge = QualityJudge(cache_dir=cache_dir, use_llm=use_llm)
    return _judge


def reset_quality_judge() -> None:
    """重置全局实例（测试用）"""
    global _judge
    with _judge_lock:
        _judge = None
