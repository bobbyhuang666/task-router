"""
推理策略选择器 — 统一入口

整合了关键词匹配和 Token 分位数两种策略选择方法：
- 有 logprobs → Token 分位数驱动（更精确，来自 adaptive_reasoning）
- 无 logprobs → 关键词 + 复杂度启发式（降级方案）

推理策略：
- direct: 模型高度自信 → 直接回答（最少 token）
- cot: 中等任务 → Chain-of-Thought（逐步推理）
- cod: 新增！Chain-of-Draft — 极简推理（CoT 的 7.6% token）
- few_shot: 模式匹配 → 动态示例
- structured: 复杂输出 → 结构化格式
"""

import os
import time
import threading
from dataclasses import dataclass
from typing import Optional

from task_router.io_utils import read_jsonl, append_jsonl


# ─── 推理策略定义 ──────────────────────────────────────────────

STRATEGY_DIRECT = "direct"
STRATEGY_COT = "cot"
STRATEGY_COD = "cod"
STRATEGY_FEW_SHOT = "few_shot"
STRATEGY_STRUCTURED = "structured"

# 策略对 token 消耗的影响（相对于 direct 的倍数）
# Chain-of-Draft: 仅 7.6% 推理 token（来自论文 "Chain of Draft"）
STRATEGY_TOKEN_MULTIPLIER = {
    STRATEGY_DIRECT: 1.0,
    STRATEGY_COT: 1.8,
    STRATEGY_COD: 0.3,
    STRATEGY_FEW_SHOT: 1.5,
    STRATEGY_STRUCTURED: 1.3,
}

# Token 预算别名（兼容 adaptive_reasoning.py）
TOKEN_BUDGET = STRATEGY_TOKEN_MULTIPLIER

# 策略适用的任务复杂度范围
STRATEGY_COMPLEXITY_RANGE = {
    STRATEGY_DIRECT: (0.0, 2.0),
    STRATEGY_COT: (2.0, 5.0),
    STRATEGY_COD: (1.5, 4.0),
    STRATEGY_FEW_SHOT: (1.5, 4.0),
    STRATEGY_STRUCTURED: (3.0, 8.0),
}

# CoT 触发关键词
COT_TRIGGERS = [
    "分析", "推理", "评价", "评估", "解释", "为什么",
    "原因", "影响", "优缺点", "利弊", "权衡", "选择", "决策",
    "推导", "论证", "证明", "假设", "如果",
]

# 结构化输出触发关键词
STRUCTURED_TRIGGERS = [
    "列出", "列举", "大纲", "结构", "框架", "步骤", "流程",
    "方案", "计划", "报告", "总结", "汇总", "清单",
    "表格", "对比", "比较",
]


# ─── 策略增强 Prompt ──────────────────────────────────────────

STRATEGY_ENHANCEMENTS = {
    STRATEGY_DIRECT: "",  # 不增强，使用原始 prompt

    STRATEGY_COT: "\n\n请一步一步思考：\n1. 先理解问题的关键点\n2. 逐步分析\n3. 给出最终答案\n",

    # Chain-of-Draft：极简推理提示（来自论文 "Chain of Draft"）
    STRATEGY_COD: "\n\n用最少的词写出关键推理步骤，然后给出答案。\n",

    STRATEGY_FEW_SHOT: "",  # 由 distillation 系统动态注入

    STRATEGY_STRUCTURED: "\n\n请用以下格式输出：\n- 要点1\n- 要点2\n- ...\n- 结论\n",
}


# ─── 统一策略选择器 ──────────────────────────────────────────────

@dataclass
class StrategyDecision:
    """策略选择结果"""
    strategy: str
    reason: str
    complexity_score: float
    confidence_signal: float = 0.0      # 置信度信号（来自 token 分位数）
    token_budget_factor: float = 1.0    # token 预算倍数


def select_strategy(
    action: str,
    text: str,
    complexity_score: float,
    task_type: str = "",
    logprobs: Optional[list[dict]] = None,
    has_examples: bool = False,
) -> StrategyDecision:
    """
    统一策略选择器 — 有 logprobs 时用 Token 分位数，否则用关键词匹配。

    参数:
        action: 任务描述
        text: 输入文本
        complexity_score: 复杂度评分
        task_type: 任务类型
        logprobs: token 级 logprobs（可选，来自模型输出）
        has_examples: 是否有蒸馏示例

    返回:
        StrategyDecision
    """
    if logprobs:
        return _select_by_quantile(
            logprobs=logprobs,
            task_type=task_type,
            has_examples=has_examples,
            complexity_score=complexity_score,
            action=action,
        )
    else:
        return _select_by_keywords(action, text, complexity_score, task_type)


def _select_by_keywords(
    action: str,
    text: str,
    complexity_score: float,
    task_type: str = "",
) -> StrategyDecision:
    """关键词 + 复杂度启发式策略选择（无 logprobs 时的降级方案）"""
    action_lower = action.lower()

    # 1. 简单任务 → 直接回答
    if complexity_score < 2.0:
        return StrategyDecision(
            strategy=STRATEGY_DIRECT,
            reason=f"简单任务 (score={complexity_score:.1f})",
            complexity_score=complexity_score,
        )

    # 2. 检测是否需要 CoT
    cot_matches = [kw for kw in COT_TRIGGERS if kw in action_lower]
    if cot_matches and complexity_score >= 2.0:
        return StrategyDecision(
            strategy=STRATEGY_COT,
            reason=f"需要推理 (触发词: {','.join(cot_matches[:3])})",
            complexity_score=complexity_score,
        )

    # 3. 检测是否需要结构化输出
    struct_matches = [kw for kw in STRUCTURED_TRIGGERS if kw in action_lower]
    if struct_matches and complexity_score >= 3.0:
        return StrategyDecision(
            strategy=STRATEGY_STRUCTURED,
            reason=f"需要结构化输出 (触发词: {','.join(struct_matches[:3])})",
            complexity_score=complexity_score,
        )

    # 4. 中等复杂度 → CoT
    if 2.0 <= complexity_score < 5.0:
        return StrategyDecision(
            strategy=STRATEGY_COT,
            reason=f"中等复杂度 (score={complexity_score:.1f})",
            complexity_score=complexity_score,
        )

    # 5. 高复杂度 → 结构化
    return StrategyDecision(
        strategy=STRATEGY_STRUCTURED,
        reason=f"高复杂度 (score={complexity_score:.1f})",
        complexity_score=complexity_score,
    )


def _select_by_quantile(
    logprobs: list[dict],
    task_type: str = "",
    has_examples: bool = False,
    complexity_score: float = 0.0,
    action: str = "",
) -> StrategyDecision:
    """
    Token 分位数驱动的策略选择（核心创新）。

    用模型自身的不确定性信号来选择推理策略，
    而不是依赖外部关键词匹配。优势：
    1. 适应性：直接反映模型对当前任务的理解程度
    2. 精确性：比关键词匹配更细粒度
    3. 语言无关：不依赖特定语言的关键词
    """
    # 延迟导入避免循环依赖
    from task_router.tqbc import TokenQuantileFeatures, extract_quantile_features

    # 提取 token 分位数特征
    qf = extract_quantile_features(logprobs) if logprobs else TokenQuantileFeatures()

    # 计算综合置信度信号
    # 使用 q50_margin（长度无关）和 first_token_margin（关键信号）
    if qf.token_count > 0:
        # 加权组合：首 token 40% + 中位数 40% + 最低值 20%
        confidence = (
            0.4 * qf.first_token_margin +
            0.4 * qf.q50_margin +
            0.2 * qf.min_margin
        )
        # 惩罚高方差（不一致的输出更不可靠）
        variance_penalty = min(0.2, qf.entropy_variance * 0.1)
        confidence = max(0, confidence - variance_penalty)
    else:
        # 无 logprobs，使用复杂度作为代理
        confidence = max(0, 1.0 - complexity_score / 10.0)

    # 结构化输出检测
    action_lower = (action or "").lower()
    needs_structured = any(t in action_lower for t in STRUCTURED_TRIGGERS)

    # ── 策略选择决策树 ──

    # 1. 需要结构化输出 → structured
    if needs_structured and complexity_score >= 2.5:
        trigger = next(t for t in STRUCTURED_TRIGGERS if t in action_lower)
        return StrategyDecision(
            strategy=STRATEGY_STRUCTURED,
            reason=f"结构化输出需求 (触发词: {trigger})",
            complexity_score=complexity_score,
            confidence_signal=confidence,
            token_budget_factor=TOKEN_BUDGET[STRATEGY_STRUCTURED],
        )

    # 2. 高置信度 → direct（模型很确定）
    if confidence >= 0.7 and qf.token_count > 0:
        return StrategyDecision(
            strategy=STRATEGY_DIRECT,
            reason=f"高置信度 (q50_margin={qf.q50_margin:.2f}, 首token={qf.first_token_margin:.2f})",
            complexity_score=complexity_score,
            confidence_signal=confidence,
            token_budget_factor=TOKEN_BUDGET[STRATEGY_DIRECT],
        )

    # 3. 有蒸馏示例 → few_shot（优先利用历史知识）
    if has_examples and 0.3 <= confidence < 0.7:
        return StrategyDecision(
            strategy=STRATEGY_FEW_SHOT,
            reason=f"有蒸馏示例且中等置信度 (conf={confidence:.2f})",
            complexity_score=complexity_score,
            confidence_signal=confidence,
            token_budget_factor=TOKEN_BUDGET[STRATEGY_FEW_SHOT],
        )

    # 4. 中等置信度 → Chain-of-Draft（极简推理，节省 token）
    if 0.4 <= confidence < 0.7 and complexity_score < 4.0:
        return StrategyDecision(
            strategy=STRATEGY_COD,
            reason=f"中等置信度，用 Chain-of-Draft 节省 token (conf={confidence:.2f})",
            complexity_score=complexity_score,
            confidence_signal=confidence,
            token_budget_factor=TOKEN_BUDGET[STRATEGY_COD],
        )

    # 5. 中等偏低置信度 → CoT（逐步推理提升准确率）
    if 0.2 <= confidence < 0.5:
        return StrategyDecision(
            strategy=STRATEGY_COT,
            reason=f"中低置信度，需要逐步推理 (conf={confidence:.2f})",
            complexity_score=complexity_score,
            confidence_signal=confidence,
            token_budget_factor=TOKEN_BUDGET[STRATEGY_COT],
        )

    # 6. 低置信度且无 logprobs → 基于复杂度的启发式
    if qf.token_count == 0:
        if complexity_score < 2.0:
            return StrategyDecision(
                strategy=STRATEGY_DIRECT,
                reason=f"无 logprobs，低复杂度 (score={complexity_score:.1f})",
                complexity_score=complexity_score,
                confidence_signal=confidence,
                token_budget_factor=TOKEN_BUDGET[STRATEGY_DIRECT],
            )
        elif complexity_score < 5.0:
            return StrategyDecision(
                strategy=STRATEGY_COT,
                reason=f"无 logprobs，中等复杂度 (score={complexity_score:.1f})",
                complexity_score=complexity_score,
                confidence_signal=confidence,
                token_budget_factor=TOKEN_BUDGET[STRATEGY_COT],
            )
        else:
            return StrategyDecision(
                strategy=STRATEGY_STRUCTURED,
                reason=f"无 logprobs，高复杂度 (score={complexity_score:.1f})",
                complexity_score=complexity_score,
                confidence_signal=confidence,
                token_budget_factor=TOKEN_BUDGET[STRATEGY_STRUCTURED],
            )

    # 7. 默认：CoT（安全选择）
    return StrategyDecision(
        strategy=STRATEGY_COT,
        reason=f"默认 CoT (conf={confidence:.2f}, tokens={qf.token_count})",
        complexity_score=complexity_score,
        confidence_signal=confidence,
        token_budget_factor=TOKEN_BUDGET[STRATEGY_COT],
    )


# ─── 兼容旧接口 ──────────────────────────────────────────────

def select_reasoning_strategy(
    action: str,
    text: str,
    complexity_score: float,
    task_type: str = "",
) -> StrategyDecision:
    """兼容旧接口 — 仅关键词匹配，不使用 logprobs"""
    return _select_by_keywords(action, text, complexity_score, task_type)


def select_adaptive_strategy(
    logprobs: list[dict],
    task_type: str = "",
    has_examples: bool = False,
    complexity_score: float = 0.0,
    action: str = "",
) -> StrategyDecision:
    """兼容旧接口 — Token 分位数驱动"""
    return _select_by_quantile(
        logprobs=logprobs,
        task_type=task_type,
        has_examples=has_examples,
        complexity_score=complexity_score,
        action=action,
    )


# ─── 策略增强 Prompt ──────────────────────────────────────────

def enhance_prompt_with_strategy(prompt: str, strategy: str, examples: str = "") -> str:
    """
    用选定的策略增强 prompt。

    参数:
        prompt: 原始 prompt
        strategy: 推理策略
        examples: 动态示例（用于 few_shot）

    返回:
        增强后的 prompt
    """
    if strategy == STRATEGY_DIRECT:
        return prompt

    if strategy == STRATEGY_FEW_SHOT and examples:
        return prompt + "\n\n参考示例：\n" + examples

    enhancement = STRATEGY_ENHANCEMENTS.get(strategy, "")
    if enhancement:
        return prompt + enhancement

    return prompt


# ─── 策略反馈追踪（合并自 StrategyFeedbackTracker） ──────────────────

class StrategyTracker:
    """
    追踪不同策略的性能表现。

    职责：
    - 记录每次策略执行的成功/失败
    - 按 task_type × strategy 维度统计成功率
    - 为 select_strategy 提供历史最优策略建议

    设计模式：观察者 — 由 task_router.py 在每次任务完成后调用 record()，
    而非主动拉取结果。这种解耦确保策略选择器不依赖任务执行管道。
    """

    LOOKBACK = 30  # 滑动窗口大小

    def __init__(self, cache_dir: str):
        self.data_file = os.path.join(cache_dir, "strategy_history.jsonl")
        self._history: list[dict] = []
        self._success_stats: dict[str, list[bool]] = {}  # "task_type:strategy" -> [success, ...]
        self._load()

    def _load(self) -> None:
        self._history = read_jsonl(self.data_file)
        # 重建 success_stats
        for e in self._history:
            key = f"{e.get('task_type', 'unknown')}:{e.get('strategy', 'direct')}"
            if key not in self._success_stats:
                self._success_stats[key] = []
            self._success_stats[key].append(e.get("success", False))
            # 保持滑动窗口
            if len(self._success_stats[key]) > self.LOOKBACK:
                self._success_stats[key] = self._success_stats[key][-self.LOOKBACK:]

    def record(
        self,
        strategy: str,
        task_type: str,
        success: bool,
        tokens_used: int = 0,
        latency_ms: int = 0,
    ) -> None:
        """记录一次策略执行结果"""
        entry = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "strategy": strategy,
            "task_type": task_type,
            "success": success,
            "tokens": tokens_used,
            "latency_ms": latency_ms,
        }
        append_jsonl(self.data_file, entry)
        self._history.append(entry)

        # 更新滑动窗口统计
        key = f"{task_type}:{strategy}"
        if key not in self._success_stats:
            self._success_stats[key] = []
        self._success_stats[key].append(success)
        if len(self._success_stats[key]) > self.LOOKBACK:
            self._success_stats[key] = self._success_stats[key][-self.LOOKBACK:]

    def get_success_rate(self, task_type: str, strategy: str) -> float:
        """获取指定 task_type × strategy 的成功率，-1 表示数据不足"""
        key = f"{task_type}:{strategy}"
        records = self._success_stats.get(key, [])
        if len(records) < 3:
            return -1.0
        return sum(records) / len(records)

    def get_best_strategy(self, task_type: str, candidates: list[str] | None = None) -> Optional[str]:
        """
        获取某任务类型的最优策略（成功率 > 0.6 时才返回）。

        参数:
            task_type: 任务类型
            candidates: 候选策略列表（默认所有已知策略）
        """
        if candidates is None:
            candidates = list(STRATEGY_TOKEN_MULTIPLIER.keys())

        best_rate = -1.0
        best_strategy = None
        for s in candidates:
            rate = self.get_success_rate(task_type, s)
            if rate > best_rate:
                best_rate = rate
                best_strategy = s
        return best_strategy if best_rate > 0.6 else None

    def get_stats(self) -> dict:
        """获取策略统计"""
        if not self._history:
            return {"total": 0, "by_strategy": {}, "by_task_type": {}}

        by_strategy: dict[str, dict] = {}
        by_task_type: dict[str, dict] = {}

        for e in self._history:
            s = e.get("strategy", "unknown")
            t = e.get("task_type", "unknown")

            if s not in by_strategy:
                by_strategy[s] = {"total": 0, "success": 0, "tokens": 0}
            by_strategy[s]["total"] += 1
            if e.get("success"):
                by_strategy[s]["success"] += 1
            by_strategy[s]["tokens"] += e.get("tokens", 0)

            if t not in by_task_type:
                by_task_type[t] = {"total": 0, "success": 0}
            by_task_type[t]["total"] += 1
            if e.get("success"):
                by_task_type[t]["success"] += 1

        return {
            "total": len(self._history),
            "by_strategy": by_strategy,
            "by_task_type": by_task_type,
        }


# 全局实例（延迟初始化，线程安全）
_tracker: Optional[StrategyTracker] = None
_tracker_lock = threading.Lock()


def get_strategy_tracker(cache_dir: Optional[str] = None) -> StrategyTracker:
    global _tracker
    if _tracker is None:
        with _tracker_lock:
            if _tracker is None:
                if cache_dir is None:
                    from task_router.config import get_config
                    cache_dir = get_config().cache_dir
                _tracker = StrategyTracker(cache_dir)
    return _tracker
