"""
自适应推理策略 — 向后兼容包装

本模块的功能已合并到 reasoning.py。
保留此文件以兼容现有测试和导入。

迁移指南：
- select_adaptive_strategy() → reasoning.select_adaptive_strategy()
- select_strategy(logprobs=...)  ← 推荐的新入口
"""

from task_router.reasoning import (
    select_adaptive_strategy,
    select_strategy,
    enhance_prompt_with_strategy as enhance_prompt_adaptive,
    TOKEN_BUDGET,
    StrategyDecision as AdaptiveStrategyDecision,
)

__all__ = [
    "select_adaptive_strategy",
    "select_strategy",
    "enhance_prompt_adaptive",
    "TOKEN_BUDGET",
    "AdaptiveStrategyDecision",
]
