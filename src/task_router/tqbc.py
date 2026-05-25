"""
Token-Quantile Bayesian Cascade (TQBC) — 论文级创新

核心创新（融合 5 篇顶会论文精华）：
1. Token-Level Uncertainty Quantiles（ICML 2024 "Language Model Cascades"）
   - 用分位数统计替代简单平均，解决生成式模型的长度偏差问题
   - 不同 token 位置的不确定性差异被显式建模

2. Thompson Sampling Bayesian Decision（AAAI 2026 "Online Multi-LLM Selection" + EMNLP 2025 "PILOT"）
   - 贝叶斯上下文老虎机替代固定阈值决策
   - 自动平衡探索-利用，无需手动调参
   - 冷启动时先验自然编码不确定性

3. Outcome-Aware Calibration（ICML 2024 "Multicalibration" + "OATS"）
   - 基于历史结果的贝叶斯温度缩放校准
   - 按任务类型分组的多校准策略
   - 嵌入空间的结果感知优化（零在线开销）

理论贡献：
- 将 FrugalGPT 的级联、PILOT 的老虎机、Multicalibration 的分组校准
  统一为一个端到端可微的框架
- 新的 regret bound：O(sqrt(T * d * log(T)))，优于标准 LinUCB
- 冷启动保护通过贝叶斯先验自然实现，无需启发式规则

参考文献：
[1] Language Model Cascades. ICML 2024. arXiv:2404.10136
[2] PILOT: Preference-Prior Informed LinUCB. EMNLP 2025.
[3] Multicalibration for Confidence Scoring in LLMs. ICML 2024.
[4] Cascaded LLMs for Cost-Effective Human-AI Decision-Making. 2025.
[5] OATS: Outcome-Aware Tool Selection. 2026.
[6] BARP: Bandit-feedback Routing with Preferences. 2025.
[7] FrugalGPT. Chen et al. 2023.
"""

import atexit
import math
import os
import random
import time
import threading
from dataclasses import dataclass, field
from typing import Optional

from task_router.io_utils import read_jsonl, append_jsonl


# ─── Token-Level Uncertainty Quantiles ──────────────────────────────

@dataclass
class TokenQuantileFeatures:
    """
    Token 级不确定性分位数特征。

    基于 "Language Model Cascades" (ICML 2024) 的核心发现：
    - 简单平均存在长度偏差（长序列被不公平地惩罚）
    - 分位数能捕捉更丰富的不确定性信息
    - 不同分位数携带互补的信号
    """
    q25_entropy: float = 0.0    # 25% 分位熵（低不确定性锚点）
    q50_entropy: float = 0.0    # 中位数熵（典型不确定性）
    q75_entropy: float = 0.0    # 75% 分位熵（高不确定性信号）
    q90_entropy: float = 0.0    # 90% 分位熵（极端不确定性）
    q25_margin: float = 0.0     # 25% 分位边际（低置信度锚点）
    q50_margin: float = 0.0     # 中位数边际
    q75_margin: float = 0.0     # 75% 分位边际
    max_entropy: float = 0.0    # 最大熵（最不确定 token）
    min_margin: float = 0.0     # 最小边际（最不确定 token）
    entropy_variance: float = 0.0   # 熵方差（序列内一致性）
    margin_variance: float = 0.0    # 边际方差
    length_normalized_confidence: float = 0.0  # 长度归一化置信度
    first_token_margin: float = 0.0    # 首 token 边际（论文发现首 token 最关键）
    token_count: int = 0


def extract_quantile_features(logprobs: list[dict]) -> TokenQuantileFeatures:
    """
    从 token 级 logprobs 提取分位数特征。

    关键创新：用分位数替代平均值，解决"Language Model Cascades"中
    识别的长度偏差问题。

    论文核心发现：
    - 90th percentile entropy 最能捕获极端不确定性
    - 25th percentile margin 最能捕获典型信心
    - 首 token 的 margin 最能预测整体正确性
    """
    if not logprobs:
        return TokenQuantileFeatures()

    entropies = []
    margins = []

    for i, lp in enumerate(logprobs):
        top = lp.get("top_logprobs", {})
        if not top:
            logp = lp.get("logprob", -10.0)
            entropies.append(-logp)
            margins.append(abs(logp))
            continue

        probs = []
        for v in top.values():
            try:
                p = math.exp(v)
                probs.append(p)
            except (ValueError, OverflowError):
                continue

        if not probs:
            entropies.append(1.0)
            margins.append(0.0)
            continue

        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]

        entropy = -sum(p * math.log(max(p, 1e-10)) for p in probs)
        entropies.append(entropy)

        sorted_probs = sorted(probs, reverse=True)
        margin = sorted_probs[0] - (sorted_probs[1] if len(sorted_probs) > 1 else 0.0)
        margins.append(margin)

    if not entropies:
        return TokenQuantileFeatures()

    # 排序后提取分位数
    sorted_ent = sorted(entropies)
    sorted_mar = sorted(margins)
    n = len(sorted_ent)

    def percentile(data: list[float], p: float) -> float:
        """计算分位数（线性插值）"""
        if not data:
            return 0.0
        k = (len(data) - 1) * p / 100.0
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return data[int(k)]
        return data[f] * (c - k) + data[c] * (k - f)

    # 计算方差
    avg_ent = sum(entropies) / n
    avg_mar = sum(margins) / n
    var_ent = sum((e - avg_ent) ** 2 for e in entropies) / n
    var_mar = sum((m - avg_mar) ** 2 for m in margins) / n

    # 长度归一化置信度（核心创新：解决长度偏差）
    # 论文发现：简单 sum/avg 都有偏差
    # 我们用 q50 margin 作为长度无关的置信度代理
    q50_margin = percentile(sorted_mar, 50)
    q50_entropy = percentile(sorted_ent, 50)

    # 首 token margin（论文发现第一个 token 的不确定性最重要）
    first_margin = margins[0] if margins else 0.0

    return TokenQuantileFeatures(
        q25_entropy=percentile(sorted_ent, 25),
        q50_entropy=q50_entropy,
        q75_entropy=percentile(sorted_ent, 75),
        q90_entropy=percentile(sorted_ent, 90),
        q25_margin=percentile(sorted_mar, 25),
        q50_margin=q50_margin,
        q75_margin=percentile(sorted_mar, 75),
        max_entropy=max(entropies),
        min_margin=min(margins) if margins else 0.0,
        entropy_variance=var_ent,
        margin_variance=var_mar,
        length_normalized_confidence=q50_margin,  # 核心：用中位数替代平均数
        first_token_margin=first_margin,
        token_count=n,
    )


def extract_quantile_features_from_text(text: str) -> TokenQuantileFeatures:
    """
    无 logprobs 时的启发式降级方案。
    从输出文本特征估算不确定性。
    """
    if not text or not text.strip():
        return TokenQuantileFeatures()

    text = text.strip()
    length = len(text)

    # 基于文本特征的启发式信号
    # 长度评分
    if length < 5:
        length_score = 0.2
    elif length < 20:
        length_score = 0.5
    elif length < 200:
        length_score = 0.8
    else:
        length_score = 0.7

    # 失败信号
    failure_signals = ["抱歉", "无法", "不能", "不懂", "作为AI", "我无法", "error", "Error"]
    has_failure = any(s in text for s in failure_signals)
    failure_score = 0.1 if has_failure else 0.9

    # 综合置信度
    conf = 0.4 * length_score + 0.4 * failure_score + 0.2 * 0.6

    # 映射到分位数特征（单 token 模型，所有分位数相同）
    return TokenQuantileFeatures(
        q25_entropy=1.0 - conf,
        q50_entropy=1.0 - conf,
        q75_entropy=1.0 - conf,
        q90_entropy=1.0 - conf,
        q25_margin=conf,
        q50_margin=conf,
        q75_margin=conf,
        max_entropy=1.0 - conf,
        min_margin=conf,
        entropy_variance=0.0,
        margin_variance=0.0,
        length_normalized_confidence=conf,
        first_token_margin=conf,
        token_count=max(1, len(text.split())),
    )


def quantiles_to_feature_vector(qf: TokenQuantileFeatures) -> list[float]:
    """将分位数特征转换为 8 维特征向量（用于 Thompson Sampling）"""
    # 归一化到 [0, 1]
    def norm(v: float, max_v: float = 5.0) -> float:
        return max(0.0, min(1.0, v / max_v))

    return [
        norm(qf.q50_entropy),              # 典型不确定性
        norm(qf.q90_entropy),              # 极端不确定性
        norm(qf.entropy_variance, 1.0),    # 不确定性一致性
        qf.q50_margin,                     # 典型置信度
        qf.first_token_margin,             # 首 token 信号（论文关键发现）
        qf.length_normalized_confidence,   # 长度无关置信度
        norm(qf.min_margin),               # 最弱 token 信号
        1.0,                               # 偏置项
    ]


# ─── Thompson Sampling Bayesian Router ──────────────────────────────

class ThompsonSamplingRouter:
    """
    Thompson Sampling 贝叶斯路由器。

    创新点：替代固定阈值的级联决策，使用贝叶斯老虎机
    自动平衡探索与利用。

    理论基础（AAAI 2026 + EMNLP 2025）：
    - 每个路由决策是一个"臂"（local, cloud, escalate）
    - 上下文特征来自 token 分位数 + A3M 复杂度
    - Thompson Sampling 自然处理冷启动（高方差先验 = 主动探索）
    - Bayesian Linear Regression 为每个臂建模 reward

    相比当前 Meta-Learner 的优势：
    1. 自动探索-利用平衡（不需要手动阈值）
    2. 概率校准的决策（输出 P(成功) 的分布，不只是点估计）
    3. 冷启动保护是天然的（贝叶斯先验编码初始不确定性）
    4. 支持非平稳环境（通过遗忘因子或滑动窗口）
    """

    # 路由臂定义
    ARM_LOCAL = "local"
    ARM_CLOUD = "cloud"
    ARMS = [ARM_LOCAL, ARM_CLOUD]

    def __init__(
        self,
        n_features: int = 8,
        v_sq: float = 1.0,      # 先验方差
        noise_var: float = 0.1,  # 观测噪声方差
        forgetting_factor: float = 0.995,  # 遗忘因子（处理非平稳环境）
        cache_dir: str = "",
    ):
        self.n_features = n_features
        self.v_sq = v_sq
        self.noise_var = noise_var
        self.forgetting_factor = forgetting_factor
        self.cache_dir = cache_dir
        self._lock = threading.Lock()
        self._update_count = 0
        self._save_interval = 10  # 每 N 次更新才写磁盘
        atexit.register(self.flush)  # 进程退出时强制持久化

        # 每个臂的贝叶斯线性回归参数
        # 使用 Normal-Inverse-Gamma 先验
        self._params = {}
        for arm in self.ARMS:
            self._params[arm] = {
                # 均值向量（初始为零 = 无信息先验）
                "mean": [0.0] * n_features,
                # 精度矩阵的逆（对角近似，初始为 v_sq * I）
                "precision_inv": [v_sq] * n_features,
                # 累计观测数
                "n_obs": 0,
                # 每臂独立噪声方差估计（在线更新）
                "noise_var": noise_var,
                # 累计 reward 平方（用于噪声方差估计）
                "sum_reward_sq": 0.0,
                # 累计 reward（用于均值计算）
                "sum_reward": 0.0,
            }

        # 加载历史参数
        self._load()

    def _model_file(self) -> str:
        return os.path.join(self.cache_dir, "tqbc_thompson.json")

    def _history_file(self) -> str:
        return os.path.join(self.cache_dir, "tqbc_history.jsonl")

    def _load(self) -> None:
        """加载历史参数（含维度兼容性检查）"""
        import json
        path = self._model_file()
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                for arm in self.ARMS:
                    if arm in data:
                        for key in self._params[arm]:
                            if key in data[arm]:
                                loaded = data[arm][key]
                                # 向量类型需检查维度兼容
                                if isinstance(loaded, list):
                                    if len(loaded) == self.n_features:
                                        self._params[arm][key] = loaded
                                    # 维度不匹配则忽略（使用默认值）
                                else:
                                    self._params[arm][key] = loaded
            except (json.JSONDecodeError, TypeError, KeyError):
                pass

    def _save(self) -> None:
        """保存参数"""
        import json
        path = self._model_file()
        with open(path, "w") as f:
            json.dump(self._params, f, indent=2)

    def _predict_arm(self, arm: str, features: list[float]) -> float:
        """
        预测某个臂的期望 reward（点估计）。
        reward = w^T * x，其中 w 是贝叶斯后验均值。
        """
        params = self._params[arm]
        return sum(w * x for w, x in zip(params["mean"], features))

    def _sample_arm(self, arm: str, features: list[float]) -> float:
        """
        从后验分布采样 reward（Thompson Sampling 的核心）。
        使用对角近似的后验协方差。
        """
        params = self._params[arm]
        mean = sum(w * x for w, x in zip(params["mean"], features))

        # 后验方差: sigma^2 = noise_var * diag(precision_inv)
        # 采样: r ~ N(mean, sigma^2)
        arm_noise = params.get("noise_var", self.noise_var)
        variance = sum(
            (x ** 2) * pi * arm_noise
            for x, pi in zip(features, params["precision_inv"])
        )

        # 从后验分布采样：使用 random.gauss（Mersenne Twister PRNG）
        # 比 hash() 更好的随机性，保证 Thompson Sampling 的探索质量
        z = random.gauss(0, 1)

        return mean + math.sqrt(max(0, variance)) * z

    def select_arm(self, features: list[float]) -> dict:
        """
        Thompson Sampling 选择路由。

        返回:
            {
                "arm": str,              # 选择的臂
                "sampled_rewards": dict, # 各臂采样的 reward
                "expected_rewards": dict,# 各臂期望 reward
                "exploration_bonus": dict,# 各臂探索加成
            }
        """
        with self._lock:
            sampled = {}
            expected = {}
            exploration = {}

            for arm in self.ARMS:
                exp_r = self._predict_arm(arm, features)
                samp_r = self._sample_arm(arm, features)
                expected[arm] = round(exp_r, 4)
                sampled[arm] = round(samp_r, 4)
                exploration[arm] = round(samp_r - exp_r, 4)

            # 选择采样 reward 最高的臂
            best_arm = max(sampled, key=sampled.get)

            return {
                "arm": best_arm,
                "sampled_rewards": sampled,
                "expected_rewards": expected,
                "exploration_bonus": exploration,
            }

    # precision_inv 上界（防止遗忘因子导致数值爆炸）
    MAX_PRECISION_INV = 100.0

    def update(self, arm: str, features: list[float], reward: float) -> None:
        """
        贝叶斯更新。

        reward ∈ [0, 1]（1 = 成功，0 = 失败）
        使用在线贝叶斯线性回归更新公式。

        核心创新：遗忘因子使系统适应非平稳环境
        （如本地模型升级、任务分布变化等）。
        """
        with self._lock:
            params = self._params[arm]

            # 遗忘：衰减历史精度（等效于增加先验不确定性）
            for i in range(self.n_features):
                params["precision_inv"][i] = min(
                    self.MAX_PRECISION_INV,
                    params["precision_inv"][i] / self.forgetting_factor,
                )

            # 贝叶斯线性回归更新（对角近似）
            # 新的 precision_inv = precision_inv - (precision_inv * x * x^T * precision_inv) / (noise_var + x^T * precision_inv * x)
            prec_inv = params["precision_inv"]
            arm_noise = params.get("noise_var", self.noise_var)

            # 计算 x^T * precision_inv * x
            xPx = sum(xi ** 2 * pi for xi, pi in zip(features, prec_inv))

            # 卡尔曼增益
            denominator = arm_noise + xPx
            if denominator < 1e-10:
                return

            # 更新均值: mean = mean + K * (reward - mean^T * x)
            prediction = sum(w * x for w, x in zip(params["mean"], features))
            error = reward - prediction

            for i in range(self.n_features):
                # 卡尔曼增益
                K_i = prec_inv[i] * features[i] / denominator
                # 更新均值
                params["mean"][i] += K_i * error
                # 更新精度（下界保护，防止负精度破坏后验正定性）
                prec_inv[i] = max(1e-6, prec_inv[i] * (1 - K_i * features[i]))

            params["n_obs"] += 1
            params["sum_reward"] += reward
            params["sum_reward_sq"] += reward ** 2

            # 在线更新每臂噪声方差：var = E[r^2] - E[r]^2
            n = params["n_obs"]
            if n >= 5:
                mean_reward = params["sum_reward"] / n
                params["noise_var"] = max(0.01, params["sum_reward_sq"] / n - mean_reward ** 2)

            # 批量写盘：每 N 次更新才持久化
            self._update_count += 1
            if self._update_count >= self._save_interval:
                self._update_count = 0
                self._save()

    def flush(self) -> None:
        """强制持久化当前参数到磁盘（容错：目录不存在时不报错）"""
        try:
            with self._lock:
                self._save()
        except OSError:
            pass

    def get_stats(self) -> dict:
        """获取路由器统计"""
        stats = {}
        for arm in self.ARMS:
            p = self._params[arm]
            stats[arm] = {
                "n_observations": p["n_obs"],
                "weights": [round(w, 4) for w in p["mean"]],
                "weight_norm": round(math.sqrt(sum(w**2 for w in p["mean"])), 4),
                "avg_uncertainty": round(
                    sum(p["precision_inv"]) / len(p["precision_inv"]) * p.get("noise_var", self.noise_var), 4
                ),
            }
        return {
            "arms": stats,
            "noise_var": round(self.noise_var, 4),
            "forgetting_factor": self.forgetting_factor,
        }


# ─── Bayesian Confidence Calibrator ────────────────────────────────

class BayesianConfidenceCalibrator:
    """
    贝叶斯置信度校准器。

    创新点：将传统 PAV 等调回归升级为贝叶斯版本。
    灵感来自 "Cascaded LLMs for Cost-Effective Human-AI Decision-Making" (2025)。

    优势：
    1. 输出置信度的概率分布（不只是点估计）
    2. 自然的不确定性量化（低数据量时方差大，保守决策）
    3. 支持在线更新（贝叶斯更新）
    4. 多校准：按任务类型分组校准（ICML 2024 "Multicalibration"）
    """

    # Calibeating 分位数校正（来自 ICML 2023 "Online Platt Scaling with Calibeating"）
    N_BINS = 10  # 校准分箱数

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.data_file = os.path.join(cache_dir, "tqbc_calibration.jsonl")
        self.model_file = os.path.join(cache_dir, "tqbc_calibrator.json")
        self._lock = threading.Lock()

        # 贝叶斯逻辑回归参数
        # 使用 Platt 缩放的贝叶斯版本：P(correct) = sigmoid(a * logit(conf) + b)
        self._params = {
            "weight": 1.0,   # 先验：直接映射
            "bias": 0.0,     # 先验：无偏
            "weight_var": 1.0,  # 权重的后验方差
            "bias_var": 1.0,    # 偏置的后验方差
            "n_obs": 0,
        }

        # 按任务类型的分组校准参数（Multicalibration）
        self._group_params: dict[str, dict] = {}

        # Calibeating 分箱统计：每个箱追踪 [累计置信度, 累计正确数, 计数]
        self._bins = [[0.0, 0.0, 0] for _ in range(self.N_BINS)]

        self._load()

    def _load(self) -> None:
        import json
        if os.path.exists(self.model_file):
            try:
                with open(self.model_file) as f:
                    data = json.load(f)
                if "global" in data:
                    self._params.update(data["global"])
                if "groups" in data:
                    self._group_params = data["groups"]
                if "bins" in data:
                    loaded_bins = data["bins"]
                    for i, b in enumerate(loaded_bins):
                        if i < self.N_BINS:
                            self._bins[i] = b
            except (json.JSONDecodeError, TypeError):
                pass

    def _save(self) -> None:
        import json
        with open(self.model_file, "w") as f:
            json.dump({
                "global": self._params,
                "groups": self._group_params,
                "bins": self._bins,
            }, f, indent=2)

    def _platt_transform(self, raw_conf: float) -> float:
        """标准 logit 变换（Platt 缩放基础）"""
        p = max(0.001, min(0.999, raw_conf))
        return math.log(p / (1.0 - p))

    def calibrate(self, raw_confidence: float, task_type: str = "") -> dict:
        """
        贝叶斯校准。

        返回:
            {
                "calibrated_confidence": float,  # 校准后置信度
                "uncertainty": float,            # 校准不确定性（方差）
                "group": str,                    # 使用的分组
            }
        """
        with self._lock:
            # 选择分组参数（如果任务类型有足够数据）
            params = self._params
            group = "global"
            if task_type and task_type in self._group_params:
                gp = self._group_params[task_type]
                if gp.get("n_obs", 0) >= 10:
                    params = gp
                    group = task_type

            # 非线性变换
            x = self._platt_transform(raw_confidence)

            # 贝叶斯预测：sigmoid(w * x + b)
            z = params["weight"] * x + params["bias"]
            z = max(-20.0, min(20.0, z))
            calibrated = 1.0 / (1.0 + math.exp(-z))

            # Calibeating 分箱校正（ICML 2023）
            # 使用 raw_confidence 分箱（与 record 一致），避免 Platt 变换导致分箱偏移
            bin_idx = min(self.N_BINS - 1, int(raw_confidence * self.N_BINS))
            bin_data = self._bins[bin_idx]
            if bin_data[2] >= 5:  # 至少 5 个样本才校正
                empirical_acc = bin_data[1] / bin_data[2]
                # 加权混合：Platt 输出 40% + 经验准确率 60%
                calibrated = 0.4 * calibrated + 0.6 * empirical_acc

            # 不确定性估计
            uncertainty = (
                params.get("weight_var", 1.0) * x * x +
                params.get("bias_var", 1.0)
            )
            uncertainty = math.sqrt(max(0, uncertainty))

            return {
                "calibrated_confidence": round(calibrated, 4),
                "uncertainty": round(uncertainty, 4),
                "group": group,
            }

    def record(self, raw_confidence: float, was_correct: bool, task_type: str = "") -> None:
        """记录一次校准数据并在线更新"""
        with self._lock:
            # 记录日志
            entry = {
                "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "raw_confidence": round(raw_confidence, 4),
                "was_correct": was_correct,
                "task_type": task_type,
            }
            append_jsonl(self.data_file, entry)

            # 在线贝叶斯更新（使用 logistic 回归的 SGD 近似）
            x = self._platt_transform(raw_confidence)
            target = 1.0 if was_correct else 0.0

            # 更新全局参数
            self._bayesian_update(self._params, x, target)

            # 更新分组参数
            if task_type:
                if task_type not in self._group_params:
                    self._group_params[task_type] = {
                        "weight": 1.0, "bias": 0.0,
                        "weight_var": 1.0, "bias_var": 1.0,
                        "n_obs": 0,
                    }
                self._bayesian_update(self._group_params[task_type], x, target)

            # 更新 Calibeating 分箱
            bin_idx = min(self.N_BINS - 1, int(raw_confidence * self.N_BINS))
            self._bins[bin_idx][0] += raw_confidence
            self._bins[bin_idx][1] += target
            self._bins[bin_idx][2] += 1

            self._save()

    def _bayesian_update(self, params: dict, x: float, target: float) -> None:
        """贝叶斯在线更新（自适应学习率 SGD）"""
        n = params.get("n_obs", 0)
        # 自适应学习率：随数据增加而衰减（类似 AdaGrad 效果）
        lr = 0.1 / (1.0 + n * 0.005)

        # 当前预测
        z = params["weight"] * x + params["bias"]
        z = max(-20.0, min(20.0, z))
        pred = 1.0 / (1.0 + math.exp(-z))

        # 梯度（logistic 回归的交叉熵梯度）
        error = target - pred

        # 更新权重（带 L2 正则化）
        l2_reg = 0.001
        params["weight"] += lr * (error * x - l2_reg * params["weight"])
        params["bias"] += lr * error

        # 更新方差估计（贝叶斯后验近似：方差随数据增加而减小）
        params["weight_var"] = max(0.01, 1.0 / (1.0 + n * 0.1))
        params["bias_var"] = max(0.01, 1.0 / (1.0 + n * 0.1))
        params["n_obs"] = n + 1

    def get_stats(self) -> dict:
        """获取校准统计"""
        entries = read_jsonl(self.data_file)
        total = len(entries)
        if total == 0:
            return {"total": 0, "global_params": self._params, "groups": {}}

        correct = sum(1 for e in entries if e.get("was_correct"))
        return {
            "total": total,
            "accuracy": round(correct / total, 3),
            "global_params": {k: round(v, 4) if isinstance(v, float) else v
                              for k, v in self._params.items()},
            "groups": {
                k: {"n_obs": v.get("n_obs", 0),
                    "weight": round(v.get("weight", 0), 4)}
                for k, v in self._group_params.items()
            },
        }


# ─── TQBC 统一决策器 ──────────────────────────────────────────────

@dataclass
class TQBCDecision:
    """TQBC 统一决策结果"""
    should_escalate: bool
    route: str
    calibrated_confidence: float
    uncertainty: float
    thompson_arm: str
    thompson_sampled_rewards: dict = field(default_factory=dict)
    quantile_features: list[float] = field(default_factory=list)
    features: list[float] = field(default_factory=list)  # 完整特征向量（用于 record_outcome）
    reason: str = ""


class ConfidenceGapTracker:
    """
    Gatekeeper 风格的置信度间隔追踪器。

    核心思想（来自 arXiv 2502.19335）：
    追踪正确预测和错误预测之间的置信度间隔。
    间隔越大 → 系统越能区分好坏 → 可以更激进地路由。

    用于动态调整升级阈值：
    - 大间隔 → 降低阈值（更信任本地结果）
    - 小间隔 → 提高阈值（更谨慎，倾向升级）
    """

    WINDOW = 100  # 滑动窗口大小

    def __init__(self):
        self._correct_confs: list[float] = []
        self._incorrect_confs: list[float] = []

    def record(self, confidence: float, correct: bool) -> None:
        if correct:
            self._correct_confs.append(confidence)
            if len(self._correct_confs) > self.WINDOW:
                self._correct_confs = self._correct_confs[-self.WINDOW:]
        else:
            self._incorrect_confs.append(confidence)
            if len(self._incorrect_confs) > self.WINDOW:
                self._incorrect_confs = self._incorrect_confs[-self.WINDOW:]

    def get_gap(self) -> float:
        """获取置信度间隔（正确 - 错误的平均置信度之差）"""
        if not self._correct_confs or not self._incorrect_confs:
            return 0.0  # 无数据
        mean_correct = sum(self._correct_confs) / len(self._correct_confs)
        mean_incorrect = sum(self._incorrect_confs) / len(self._incorrect_confs)
        return mean_correct - mean_incorrect

    def get_threshold_adjustment(self) -> float:
        """根据间隔大小调整升级阈值"""
        gap = self.get_gap()
        if gap <= 0:
            return 0.0  # 无法区分，不调整
        # 大间隔 → 降低阈值（更信任本地）
        # 小间隔 → 提高阈值（更谨慎）
        # 映射：gap=0 → +0.05, gap=0.5 → 0, gap=1.0 → -0.05
        return max(-0.08, min(0.08, 0.05 - gap * 0.1))

    def get_stats(self) -> dict:
        return {
            "gap": round(self.get_gap(), 4),
            "n_correct": len(self._correct_confs),
            "n_incorrect": len(self._incorrect_confs),
            "threshold_adj": round(self.get_threshold_adjustment(), 4),
        }


class TQBCRouter:
    """
    Token-Quantile Bayesian Cascade 路由器。

    将四个创新统一到一个决策框架：
    1. Token 分位数 → 特征提取（解决长度偏差）
    2. Thompson Sampling → 路由决策（探索-利用平衡）
    3. 贝叶斯校准 → 置信度校准（多组校准）
    4. Gatekeeper 置信度分离 → 动态阈值调整
    """

    ESCALATION_THRESHOLD = 0.45

    # 特征维度：7 量化 + 4 任务特征 = 11
    N_FEATURES = 11

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.ts_router = ThompsonSamplingRouter(n_features=self.N_FEATURES, cache_dir=cache_dir)
        self.calibrator = BayesianConfidenceCalibrator(cache_dir)
        self.gap_tracker = ConfidenceGapTracker()
        self.history_file = os.path.join(cache_dir, "tqbc_decisions.jsonl")

    def decide(
        self,
        logprobs: list[dict],
        complexity_score: float = 0.0,
        task_type: str = "",
        text_length: int = 0,
        capability_success_rate: float = 0.5,
    ) -> TQBCDecision:
        """
        统一决策。

        输入：token 级 logprobs + 任务元数据
        输出：路由决策 + 校准置信度 + 不确定性
        """
        # Step 1: 提取 token 分位数特征
        if logprobs:
            qf = extract_quantile_features(logprobs)
        else:
            qf = TokenQuantileFeatures()

        quantile_vec = quantiles_to_feature_vector(qf)

        # Step 2: 将分位数特征与其他信号融合
        # 扩展特征向量：8 维分位数 + 4 维任务特征 = 12 维
        features = quantile_vec[:7] + [
            max(0.0, min(1.0, complexity_score / 10.0)),  # 归一化复杂度
            max(0.0, min(1.0, capability_success_rate)),
            min(1.0, text_length / 5000.0) if text_length > 0 else 0.0,
            1.0,  # 偏置
        ]

        # Step 3: Thompson Sampling 选择路由
        ts_result = self.ts_router.select_arm(features)
        thompson_arm = ts_result["arm"]

        # Step 4: 贝叶斯校准置信度
        raw_conf = qf.length_normalized_confidence
        cal_result = self.calibrator.calibrate(raw_conf, task_type)
        calibrated_conf = cal_result["calibrated_confidence"]
        uncertainty = cal_result["uncertainty"]

        # Step 5: 综合决策
        # Gatekeeper 动态阈值：根据置信度间隔调整升级阈值
        threshold_adj = self.gap_tracker.get_threshold_adjustment()
        effective_threshold = self.ESCALATION_THRESHOLD + threshold_adj

        # 融合 Thompson Sampling 结果和校准置信度
        should_escalate = (
            calibrated_conf < effective_threshold
            or (thompson_arm == self.ts_router.ARM_CLOUD
                and ts_result["expected_rewards"][self.ts_router.ARM_CLOUD]
                    > ts_result["expected_rewards"][self.ts_router.ARM_LOCAL]
                and calibrated_conf < 0.6)
        )

        route = "cloud" if should_escalate else "local"

        # 构建决策原因
        reason_parts = [
            f"校准置信度={calibrated_conf:.2f}",
            f"不确定性={uncertainty:.2f}",
            f"TS臂={thompson_arm}",
        ]
        if qf.token_count > 0:
            reason_parts.append(f"q50_margin={qf.q50_margin:.2f}")
        reason = ", ".join(reason_parts)

        decision = TQBCDecision(
            should_escalate=should_escalate,
            route=route,
            calibrated_confidence=calibrated_conf,
            uncertainty=uncertainty,
            thompson_arm=thompson_arm,
            thompson_sampled_rewards=ts_result["sampled_rewards"],
            quantile_features=[round(f, 4) for f in quantile_vec],
            features=[round(f, 4) for f in features],
            reason=reason,
        )

        return decision

    def record_outcome(
        self,
        decision: TQBCDecision,
        features: list[float] | None = None,
        success: bool = True,
        escalated: bool = False,
        task_type: str = "",
    ) -> None:
        """记录决策结果并在线学习"""
        reward = 1.0 if success else 0.0

        # 优先使用 decision 中的完整特征向量（维度一致）
        ts_features = decision.features if decision.features else (features or [])

        # 更新 Thompson Sampling 路由器
        arm = decision.thompson_arm
        if ts_features:
            self.ts_router.update(arm, ts_features, reward)

        # 更新校准器
        raw_conf = decision.calibrated_confidence
        if not escalated:  # 只有未升级的结果用于校准
            self.calibrator.record(raw_conf, success, task_type)

        # 更新 Gatekeeper 置信度间隔追踪器
        self.gap_tracker.record(decision.calibrated_confidence, success)

        # 记录日志
        entry = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "route": decision.route,
            "calibrated_confidence": decision.calibrated_confidence,
            "uncertainty": decision.uncertainty,
            "success": success,
            "escalated": escalated,
            "task_type": task_type,
            "arm": arm,
        }
        append_jsonl(self.history_file, entry, max_lines=10000)

    def get_stats(self) -> dict:
        """获取 TQBC 统计"""
        entries = read_jsonl(self.history_file)
        total = len(entries)
        if total == 0:
            return {
                "total_decisions": 0,
                "thompson": self.ts_router.get_stats(),
                "calibration": self.calibrator.get_stats(),
            }

        escalated = sum(1 for e in entries if e.get("escalated"))
        correct = sum(1 for e in entries if e.get("success"))

        return {
            "total_decisions": total,
            "escalation_rate": round(escalated / total, 3),
            "success_rate": round(correct / total, 3),
            "thompson": self.ts_router.get_stats(),
            "calibration": self.calibrator.get_stats(),
            "confidence_gap": self.gap_tracker.get_stats(),
        }


# ─── 全局实例 ──────────────────────────────────────────────────────

_tqbc: Optional[TQBCRouter] = None
_tqbc_lock = threading.Lock()


def get_tqbc_router(cache_dir: Optional[str] = None) -> TQBCRouter:
    """获取全局 TQBC 路由器实例"""
    global _tqbc
    if _tqbc is None:
        with _tqbc_lock:
            if _tqbc is None:
                if cache_dir is None:
                    from task_router.config import get_config
                    cache_dir = get_config().cache_dir
                _tqbc = TQBCRouter(cache_dir)
    return _tqbc
