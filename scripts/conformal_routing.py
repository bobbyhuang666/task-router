"""
Conformalized Routing — 基于 Conformal Prediction 的不确定性量化路由

核心思想：将点估计的置信度转换为带统计覆盖保证的预测集合，
解决路由系统"自信地做出错误决策"的问题。

关键组件：
- AdaptiveConformalInference (ACI): 在线调整 miscoverage level α
- SlidingWindowCalibrator: 滑动窗口计算 conformal threshold
- UncertaintyDecomposer: 不确定性来源分解
- ConformalizedRouter: 编排器，输出 ConformalDecision

参考文献：
- Gibbs & Candes, "Adaptive Conformal Inference Under Distribution Shift", NeurIPS 2021
- RouteNLP (arXiv 2604.23577): CP 级联在生产中的应用
- CP-Router (AAAI 2026): CP 驱动的 LLM/LRM 路由
"""

import bisect
import json
import logging
import math
import os
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


# ─── 数据结构 ──────────────────────────────────────────────────

@dataclass
class ConformalDecision:
    """Conformal Prediction 路由决策输出"""
    # 最终路由建议
    route: str                               # "local" 或 "cloud"
    should_escalate: bool

    # Conformal Prediction 输出
    prediction_set: list[str]                # {"local"}, {"cloud"}, 或 {"local", "cloud"}
    confidence_interval: tuple[float, float]  # (lower, upper) for P(local_success)
    interval_width: float                    # upper - lower（不确定性度量）

    # ACI 状态
    alpha: float                             # 当前 miscoverage level
    target_coverage: float                   # 目标覆盖率

    # Nonconformity 分数
    nonconformity_score: float               # s(x) 当前样本
    threshold: float                         # 当前 conformal threshold q_hat

    # 不确定性来源分解
    uncertainty_sources: dict = field(default_factory=dict)
    # Keys: "epistemic", "aleatoric", "distribution_shift", "cold_start"

    # 集成元数据
    layer_signals: dict = field(default_factory=dict)
    reason: str = ""


# ─── ACI 状态机 ──────────────────────────────────────────────────

class AdaptiveConformalInference:
    """
    Adaptive Conformal Inference (ACI) — 在线调整 miscoverage level。

    alpha_{t+1} = alpha_t + gamma * (err_t - alpha_target)

    当覆盖率低于目标时，alpha 减小（扩大预测集）；
    当覆盖率高于目标时，alpha 增大（缩小预测集）。

    参考: Gibbs & Candes, NeurIPS 2021
    """

    def __init__(self, target_coverage: float = 0.9, gamma: float = 0.005,
                 alpha_min: float = 0.01, alpha_max: float = 0.5):
        self.alpha_target = 1.0 - target_coverage
        self.alpha = self.alpha_target
        self.gamma = gamma
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.n_steps = 0
        self.cumulative_err = 0.0

    def update(self, err: float) -> float:
        """
        更新 alpha。

        参数:
            err: 1.0 如果真实值不在预测集中（miscoverage），0.0 如果在（coverage）

        返回:
            更新后的 alpha
        """
        self.alpha = self.alpha + self.gamma * (err - self.alpha_target)
        self.alpha = max(self.alpha_min, min(self.alpha_max, self.alpha))
        self.n_steps += 1
        self.cumulative_err += err
        return self.alpha

    def get_empirical_coverage(self) -> float:
        """获取经验覆盖率"""
        if self.n_steps == 0:
            return 1.0
        return 1.0 - self.cumulative_err / self.n_steps

    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "alpha_target": self.alpha_target,
            "gamma": self.gamma,
            "alpha_min": self.alpha_min,
            "alpha_max": self.alpha_max,
            "n_steps": self.n_steps,
            "cumulative_err": self.cumulative_err,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AdaptiveConformalInference":
        aci = cls(
            target_coverage=1.0 - data["alpha_target"],
            gamma=data["gamma"],
            alpha_min=data.get("alpha_min", 0.01),
            alpha_max=data.get("alpha_max", 0.5),
        )
        aci.alpha = data["alpha"]
        aci.n_steps = data["n_steps"]
        aci.cumulative_err = data["cumulative_err"]
        return aci


# ─── 滑动窗口校准器 ──────────────────────────────────────────────

class SlidingWindowCalibrator:
    """
    滑动窗口校准器 — 维护最近 W 个 nonconformity scores。

    q_hat = quantile(scores, ceil((n+1)(1-alpha)) / n)

    窗口大小 W 控制自适应速度 tradeoff:
    - 小 W (如 100): 快速适应，高方差
    - 大 W (如 500): 稳定，慢适应
    """

    def __init__(self, window_size: int = 200):
        self.window_size = window_size
        self.scores: deque[tuple[float, bool]] = deque(maxlen=window_size)
        self._sorted_cache: list[float] | None = None

    def add_score(self, score: float, was_correct: bool) -> None:
        """添加新的 nonconformity score"""
        self.scores.append((score, was_correct))
        self._sorted_cache = None  # 失效缓存

    def get_sorted_scores(self) -> list[float]:
        """获取排序后的所有 scores（缓存）"""
        if self._sorted_cache is None:
            self._sorted_cache = sorted(s[0] for s in self.scores)
        return self._sorted_cache

    def get_threshold(self, alpha: float) -> float:
        """
        计算给定 alpha 水平下的 conformal threshold。

        q_hat = sorted_scores[ceil((n+1)(1-alpha)) - 1]
        """
        if not self.scores:
            return 1.0  # 无数据 → 总是包含两个类别

        sorted_scores = self.get_sorted_scores()
        n = len(sorted_scores)
        idx = min(n - 1, math.ceil((n + 1) * (1 - alpha)) - 1)
        return sorted_scores[max(0, idx)]

    def get_correct_scores(self) -> list[float]:
        """获取正确预测的 scores"""
        return [s for s, correct in self.scores if correct]

    def get_incorrect_scores(self) -> list[float]:
        """获取错误预测的 scores"""
        return [s for s, correct in self.scores if not correct]

    def to_list(self) -> list:
        return list(self.scores)

    @classmethod
    def from_list(cls, data: list, window_size: int = 200) -> "SlidingWindowCalibrator":
        cal = cls(window_size=window_size)
        cal.scores = deque([(item[0], item[1]) for item in data[-window_size:]],
                           maxlen=window_size)
        return cal


# ─── 不确定性分解器 ──────────────────────────────────────────────

class UncertaintyDecomposer:
    """
    将总不确定性分解为可解释的来源：
    - epistemic: 模型不确定性（可通过更多数据减少）
    - aleatoric: 任务固有模糊性（不可减少）
    - distribution_shift: 输入与训练数据的分布差异
    - cold_start: 该任务类型的样本不足
    """

    def __init__(self):
        self._task_type_counts: dict[str, int] = {}
        self._feature_means: list[float] = []
        self._feature_vars: list[float] = []
        self._n_samples = 0

    def update(self, task_type: str, features: list[float]) -> None:
        """更新统计信息"""
        self._task_type_counts[task_type] = self._task_type_counts.get(task_type, 0) + 1
        self._n_samples += 1

        # 在线更新特征均值和方差（Welford 算法）
        # _feature_vars 存储 M2（偏差平方和），get_variance() 返回方差
        if not self._feature_means:
            self._feature_means = list(features)
            self._feature_vars = [0.0] * len(features)
        else:
            n = self._n_samples
            for i in range(min(len(features), len(self._feature_means))):
                old_mean = self._feature_means[i]
                self._feature_means[i] += (features[i] - old_mean) / n
                self._feature_vars[i] += (features[i] - old_mean) * (features[i] - self._feature_means[i])

    def decompose(
        self,
        tqbc_uncertainty: float,
        vote_entropy: float,
        task_type: str,
        features: list[float],
    ) -> dict:
        """
        分解不确定性来源。

        返回:
            dict with keys: epistemic, aleatoric, distribution_shift, cold_start
            每个值在 [0, 1] 范围内
        """
        # epistemic: 来自 TQBC 贝叶斯后验方差
        epistemic = min(1.0, max(0.0, tqbc_uncertainty))

        # aleatoric: 四层投票熵（归一化）
        max_entropy = math.log(4)  # 4 层投票
        aleatoric = min(1.0, vote_entropy / max_entropy) if max_entropy > 0 else 0.0

        # distribution_shift: 特征向量与历史均值的距离
        distribution_shift = self._compute_distribution_shift(features)

        # cold_start: 该任务类型的样本数
        n_type = self._task_type_counts.get(task_type, 0)
        cold_start = max(0.0, 1.0 - n_type / 50.0)

        return {
            "epistemic": round(epistemic, 4),
            "aleatoric": round(aleatoric, 4),
            "distribution_shift": round(distribution_shift, 4),
            "cold_start": round(cold_start, 4),
        }

    def _compute_distribution_shift(self, features: list[float]) -> float:
        """计算特征向量与历史均值的归一化距离"""
        if not self._feature_means or self._n_samples < 5:
            return 0.5  # 数据不足时返回中等值

        total_dist = 0.0
        n_dims = min(len(features), len(self._feature_means))
        if n_dims == 0:
            return 0.5
        for i in range(n_dims):
            # _feature_vars 存储 M2（偏差平方和），方差 = M2 / n
            var = max(self._feature_vars[i] / self._n_samples, 1e-6)
            z = (features[i] - self._feature_means[i]) / math.sqrt(var)
            total_dist += z * z

        # 归一化到 [0, 1]（使用 sigmoid 映射）
        avg_dist = total_dist / n_dims
        return 1.0 / (1.0 + math.exp(-avg_dist + 2))  # 偏移使 0 附近映射到 ~0.1

    def to_dict(self) -> dict:
        return {
            "task_type_counts": self._task_type_counts,
            "feature_means": self._feature_means,
            "feature_vars": self._feature_vars,
            "n_samples": self._n_samples,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UncertaintyDecomposer":
        dec = cls()
        dec._task_type_counts = data.get("task_type_counts", {})
        dec._feature_means = data.get("feature_means", [])
        dec._feature_vars = data.get("feature_vars", [])
        dec._n_samples = data.get("n_samples", 0)
        return dec


# ─── Conformalized Router ──────────────────────────────────────

class ConformalizedRouter:
    """
    Conformalized Router — 第五层决策融合。

    将前四层的点估计输出转换为带统计覆盖保证的预测集合。
    可以在以下情况覆盖 OR 融合：
    1. 抑制升级：OR 说升级，但 conformal 预测集为 {local}
    2. 强制升级：OR 说保留，但置信区间过宽
    """

    def __init__(
        self,
        cache_dir: str,
        target_coverage: float = 0.9,
        gamma: float = 0.005,
        window_size: int = 200,
        escalation_margin: float = 0.15,
    ):
        self.cache_dir = cache_dir
        self.target_coverage = target_coverage
        self.escalation_margin = escalation_margin
        os.makedirs(cache_dir, exist_ok=True)

        self.aci = AdaptiveConformalInference(target_coverage=target_coverage, gamma=gamma)
        self.calibrator = SlidingWindowCalibrator(window_size=window_size)
        self.decomposer = UncertaintyDecomposer()

        self._lock = threading.Lock()
        self._load()

    def decide(
        self,
        cascade_decision: dict,
        tqbc_decision,
        ml_prediction: dict,
        active_verify: bool,
        features: list[float],
        task_type: str,
        raw_should_escalate: bool,
        quantile_features: Any = None,
    ) -> ConformalDecision:
        """
        Conformalized 路由决策。

        参数:
            cascade_decision: CascadeDecision 输出
            tqbc_decision: TQBCDecision 输出
            ml_prediction: Meta-Learner 输出
            active_verify: Active Learner 是否请求验证
            features: 特征向量（用于 distribution_shift）
            task_type: 任务类型
            raw_should_escalate: 前四层 OR 融合的原始决策
            quantile_features: TokenQuantileFeatures 对象（原始 logprob 统计，
                               不依赖在线学习，天然满足 exchangeability）

        返回:
            ConformalDecision
        """
        # 1. 计算 nonconformity score（使用原始 logprob 特征，不使用校准置信度）
        score = self._compute_nonconformity_score(features, quantile_features=quantile_features)

        # 2-4: 快照校准器状态（锁内 O(1)），排序计算移出锁
        with self._lock:
            alpha = self.aci.alpha
            sorted_scores = self.calibrator.get_sorted_scores()

        threshold = self._compute_threshold(sorted_scores, alpha)
        prediction_set = self._build_prediction_set(score, threshold, raw_should_escalate)
        ci = self._compute_confidence_interval_from(sorted_scores, score)

        # 5. 分解不确定性（vote_entropy 只计算一次）
        vote_entropy = self._compute_vote_entropy(
            cascade_decision, tqbc_decision, ml_prediction, active_verify)
        uncertainty_sources = self.decomposer.decompose(
            tqbc_uncertainty=getattr(tqbc_decision, 'uncertainty', 0.5),
            vote_entropy=vote_entropy,
            task_type=task_type,
            features=features,
        )

        # 6. 最终决策
        if len(prediction_set) == 1:
            final_escalate = "cloud" in prediction_set
        else:
            # 双元素集合：不确定情况
            if ci[1] - ci[0] > self.escalation_margin:
                final_escalate = True  # 区间过宽 → 保守升级
            else:
                final_escalate = raw_should_escalate

        # 构建层信号
        layer_signals = {
            "cascade": {
                "escalate": cascade_decision.get("escalate", False),
                "calibrated_conf": cascade_decision.get("calibrated_confidence", 0.5),
            },
            "meta_learner": {
                "should_use_local": ml_prediction.get("should_use_local", True),
                "confidence": ml_prediction.get("confidence", 0.5),
            },
            "active_learner": {"verify": active_verify},
            "tqbc": {
                "escalate": getattr(tqbc_decision, 'should_escalate', False),
                "calibrated_conf": getattr(tqbc_decision, 'calibrated_confidence', 0.5),
                "uncertainty": getattr(tqbc_decision, 'uncertainty', 0.5),
            },
            "raw_fusion": raw_should_escalate,
        }

        # 构建原因
        reason = self._build_reason(
            score, threshold, prediction_set, ci, final_escalate, raw_should_escalate)

        return ConformalDecision(
            route="cloud" if final_escalate else "local",
            should_escalate=final_escalate,
            prediction_set=prediction_set,
            confidence_interval=ci,
            interval_width=ci[1] - ci[0],
            alpha=self.aci.alpha,
            target_coverage=self.target_coverage,
            nonconformity_score=score,
            threshold=threshold,
            uncertainty_sources=uncertainty_sources,
            layer_signals=layer_signals,
            reason=reason,
        )

    def record_outcome(
        self,
        decision: ConformalDecision,
        success: bool,
        escalated: bool,
        task_type: str,
        features: list[float] | None = None,
    ) -> None:
        """
        记录决策结果，更新 ACI 和校准器。

        参数:
            decision: ConformalDecision 输出
            success: 任务是否成功
            escalated: 是否升级到云端
            task_type: 任务类型
            features: 特征向量（用于更新 UncertaintyDecomposer）
        """
        should_save = False
        with self._lock:
            # 更新滑动窗口校准器
            correct_route = "cloud" if escalated else "local"
            was_correct = correct_route in decision.prediction_set
            self.calibrator.add_score(decision.nonconformity_score, was_correct)

            # 更新 ACI
            err = 0.0 if was_correct else 1.0
            self.aci.update(err)

            # 更新不确定性分解器
            if features:
                self.decomposer.update(task_type, features)

            # 定期保存（标记，锁外执行 I/O）
            if self.aci.n_steps % 10 == 0:
                should_save = True

        if should_save:
            self._save()

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "aci": {
                "alpha": round(self.aci.alpha, 4),
                "alpha_target": round(self.aci.alpha_target, 4),
                "n_steps": self.aci.n_steps,
                "empirical_coverage": round(self.aci.get_empirical_coverage(), 4),
            },
            "calibrator": {
                "window_size": self.calibrator.window_size,
                "n_scores": len(self.calibrator.scores),
                "threshold": round(self.calibrator.get_threshold(self.aci.alpha), 4),
            },
            "decomposer": {
                "n_samples": self.decomposer._n_samples,
                "n_task_types": len(self.decomposer._task_type_counts),
            },
            "config": {
                "target_coverage": self.target_coverage,
                "escalation_margin": self.escalation_margin,
            },
        }

    # ─── 内部方法 ──────────────────────────────────────────────

    def _compute_nonconformity_score(
        self,
        features: list[float],
        quantile_features: Any = None,
    ) -> float:
        """
        计算 nonconformity score。

        使用原始 logprob 统计量（TokenQuantileFeatures），不使用校准置信度。
        原始特征不依赖在线学习循环，天然满足 exchangeability 假设。

        s(x) = 0.40 * s_entropy  (中位数熵，归一化到 [0,1])
             + 0.30 * s_margin   (中位数边际，归一化到 [0,1])
             + 0.15 * s_variance (熵方差，归一化到 [0,1])
             + 0.15 * s_shift    (分布偏移)
        """
        if quantile_features is not None:
            s_entropy = 1.0 / (1.0 + math.exp(-2.0 * (quantile_features.q50_entropy - 1.5)))
            s_margin = 1.0 - max(0.0, min(1.0, quantile_features.q50_margin))
            s_variance = 1.0 / (1.0 + math.exp(-5.0 * (quantile_features.entropy_variance - 0.5)))
        else:
            # Fallback: 中性默认值
            s_entropy = 0.5
            s_margin = 0.5
            s_variance = 0.5

        s_shift = self.decomposer._compute_distribution_shift(features)

        score = 0.40 * s_entropy + 0.30 * s_margin + 0.15 * s_variance + 0.15 * s_shift
        return max(0.0, min(1.0, score))

    def _build_prediction_set(self, score: float, threshold: float, raw_should_escalate: bool) -> list[str]:
        """
        构建预测集。

        score <= threshold → 样本 conforming → 包含推荐路由
        score > threshold → 样本 nonconforming → 包含两个路由（不确定）
        """
        if score <= threshold:
            # 样本 conforming，包含推荐路由
            return ["cloud"] if raw_should_escalate else ["local"]
        else:
            # 样本 nonconforming，不确定
            return ["local", "cloud"]

    def _compute_threshold(self, sorted_scores: list[float], alpha: float) -> float:
        """从已排序的 scores 快照计算 conformal 阈值"""
        if not sorted_scores:
            return 1.0
        n = len(sorted_scores)
        idx = min(n - 1, math.ceil((n + 1) * (1 - alpha)) - 1)
        return sorted_scores[max(0, idx)]

    def _compute_confidence_interval_from(
        self, sorted_scores: list[float], score: float
    ) -> tuple[float, float]:
        """从已排序的 scores 快照计算置信区间"""
        if len(sorted_scores) < 5:
            return (0.1, 0.9)

        n = len(sorted_scores)
        rank = bisect.bisect_right(sorted_scores, score) / n
        confidence = 1.0 - rank

        q25 = sorted_scores[max(0, n // 4)]
        q75 = sorted_scores[min(n - 1, 3 * n // 4)]
        iqr = q75 - q25
        half_width = max(0.05, min(0.4, 0.5 * iqr * (1.0 + rank)))

        lower = max(0.0, confidence - half_width)
        upper = min(1.0, confidence + half_width)
        return (round(lower, 4), round(upper, 4))

    def _compute_confidence_interval(
        self, score: float
    ) -> tuple[float, float]:
        """
        计算置信区间（conformal-native 方法）。

        使用滑动窗口中全部 nonconformity scores 的经验分布构造 prediction interval。
        区间宽度由历史分数的分散程度决定。
        """
        all_scores = self.calibrator.get_sorted_scores()

        if len(all_scores) < 5:
            # 数据不足：保守估计，宽区间
            return (0.1, 0.9)

        n = len(all_scores)

        # score 在全部分数分布中的分位数位置（bisect: O(log n)）
        rank = bisect.bisect_right(all_scores, score) / n

        # rank 高（score 偏大）→ 不确定性高 → 置信度低
        # rank 低（score 偏小）→ 不确定性低 → 置信度高
        confidence = 1.0 - rank

        # 区间宽度由全部分数的 IQR 决定
        q25 = all_scores[max(0, n // 4)]
        q75 = all_scores[min(n - 1, 3 * n // 4)]
        iqr = q75 - q25

        # 半宽 = IQR 的一半，乘以 (1 + rank) 使得接近阈值时更宽
        half_width = max(0.05, min(0.4, 0.5 * iqr * (1.0 + rank)))

        lower = max(0.0, confidence - half_width)
        upper = min(1.0, confidence + half_width)

        return (round(lower, 4), round(upper, 4))

    def _compute_vote_entropy(
        self,
        cascade_decision: dict,
        tqbc_decision,
        ml_prediction: dict,
        active_verify: bool,
    ) -> float:
        """计算四层投票的信息熵"""
        votes = [
            1 if cascade_decision.get("escalate", False) else 0,
            1 if not ml_prediction.get("should_use_local", True) else 0,
            1 if active_verify else 0,
            1 if getattr(tqbc_decision, 'should_escalate', False) else 0,
        ]

        n_escalate = sum(votes)
        n_keep = 4 - n_escalate

        if n_escalate == 0 or n_keep == 0:
            return 0.0  # 完全一致

        p_esc = n_escalate / 4
        p_keep = n_keep / 4
        entropy = -(p_esc * math.log(p_esc) + p_keep * math.log(p_keep))
        return entropy

    def _build_reason(
        self,
        score: float,
        threshold: float,
        prediction_set: list[str],
        ci: tuple[float, float],
        final_escalate: bool,
        raw_escalate: bool,
    ) -> str:
        """构建决策原因字符串"""
        parts = [
            f"nonconformity={score:.3f}",
            f"threshold={threshold:.3f}",
            f"prediction_set={prediction_set}",
            f"CI=({ci[0]:.3f}, {ci[1]:.3f})",
            f"alpha={self.aci.alpha:.4f}",
        ]

        if final_escalate != raw_escalate:
            if final_escalate:
                parts.append("override: 区间过宽，强制升级")
            else:
                parts.append("override: conformal 预测集仅含 local，抑制升级")

        return " | ".join(parts)

    # ─── 持久化 ──────────────────────────────────────────────

    def _save(self) -> None:
        """保存状态到磁盘（单文件）"""
        state_file = os.path.join(self.cache_dir, "conformal_state.json")
        state = {
            "aci": self.aci.to_dict(),
            "scores": self.calibrator.to_list(),
            "decomposer": self.decomposer.to_dict(),
        }
        with open(state_file, "w") as f:
            json.dump(state, f)

    def _load(self) -> None:
        """从磁盘加载状态"""
        state_file = os.path.join(self.cache_dir, "conformal_state.json")
        if not os.path.exists(state_file):
            return
        try:
            with open(state_file) as f:
                state = json.load(f)
            if "aci" in state:
                self.aci = AdaptiveConformalInference.from_dict(state["aci"])
            if "scores" in state:
                self.calibrator = SlidingWindowCalibrator.from_list(
                    state["scores"], self.calibrator.window_size)
            if "decomposer" in state:
                self.decomposer = UncertaintyDecomposer.from_dict(state["decomposer"])
        except (json.JSONDecodeError, KeyError) as e:
            log.warning("conformal state 加载失败: %s", e)


# ─── 全局实例 ──────────────────────────────────────────────────

_conformal_router = None
_conformal_lock = threading.Lock()


def get_conformal_router(cache_dir: str = None) -> ConformalizedRouter:
    """获取全局 ConformalizedRouter 实例（线程安全）"""
    global _conformal_router
    if _conformal_router is None:
        with _conformal_lock:
            if _conformal_router is None:
                if cache_dir is None:
                    from config import get_config
                    cache_dir = get_config().cache_dir
                _conformal_router = ConformalizedRouter(cache_dir)
    return _conformal_router
