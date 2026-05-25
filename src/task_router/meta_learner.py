"""
Meta-Learner — 统一所有路由信号的轻量级全局决策器

核心思路：
当前系统有 5+ 个独立的自适应机制（A3M 权重、置信度级联、推理策略、
能力追踪器、阈值调整），每个都有自己的学习率和启发式规则。
Meta-Learner 把所有信号统一为特征向量，用一个在线 logistic regression
做全局路由决策。

优势：
1. 信号交互：置信度 × 复杂度 × 历史成功率的组合效应被自动学习
2. 收敛快：SGD 在线学习，每次路由结果立即更新
3. 无需 GPU：参数量 < 100，纯 numpy 计算
4. 可解释：权重向量直接反映每个特征的重要性

特征维度（10 维）：
[0] complexity_score    — A3M 复杂度评分（归一化到 [0,1]）
[1] confidence          — 本地模型置信度
[2] entropy             — token 熵（归一化）
[3] margin              — top-1 vs top-2 差距
[4] text_length_norm    — 输入文本长度（归一化）
[5] file_count_norm     — 文件数量（归一化）
[6] capability_success  — 该能力的历史成功率
[7] strategy_cot        — 是否使用 CoT 策略
[8] strategy_structured — 是否使用结构化策略
[9] bias                — 偏置项（恒为 1）
"""

import math
import os
import time
import threading
from typing import Optional

from task_router.io_utils import read_jsonl, append_jsonl


# ─── 特征提取 ──────────────────────────────────────────────

def extract_routing_features(
    complexity_score: float,
    confidence_data: dict,
    text_length: int = 0,
    file_count: int = 0,
    capability_success_rate: float = 0.5,
    strategy: str = "direct",
) -> list[float]:
    """
    从所有信号中提取 10 维特征向量。

    参数:
        complexity_score: A3M 复杂度评分
        confidence_data: 置信度数据 {"confidence", "entropy", "margin"}
        text_length: 输入文本长度
        file_count: 文件数量
        capability_success_rate: 该能力的历史成功率
        strategy: 推理策略

    返回:
        10 维特征向量
    """
    # 归一化复杂度到 [0, 1]（假设范围 [0, 10]）
    c_score = max(0.0, min(1.0, complexity_score / 10.0))

    # 置信度信号
    conf = confidence_data.get("confidence", 0.5)
    entropy = max(0.0, min(1.0, confidence_data.get("entropy", 0.5) / 5.0))
    margin = max(0.0, min(1.0, confidence_data.get("margin", 0.5)))

    # 归一化文本长度（log 缩放，假设最大 10000 字符）
    text_norm = min(1.0, math.log(max(1, text_length) + 1) / math.log(10001))

    # 归一化文件数量（假设最大 100）
    file_norm = min(1.0, file_count / 100.0)

    # 能力成功率
    cap_rate = max(0.0, min(1.0, capability_success_rate))

    # 策略 one-hot
    is_cot = 1.0 if strategy in ("cot", "few_shot") else 0.0
    is_structured = 1.0 if strategy == "structured" else 0.0

    return [c_score, conf, entropy, margin, text_norm, file_norm, cap_rate, is_cot, is_structured, 1.0]


# ─── 在线 Logistic Regression ──────────────────────────────

class OnlineLogisticRegression:
    """
    在线 logistic regression — 每次路由结果立即更新权重。

    使用 SGD 更新，学习率自适应（AdaGrad 风格）。
    """

    def __init__(self, n_features: int = 10, learning_rate: float = 0.1):
        self.n_features = n_features
        self.learning_rate = learning_rate
        # 权重初始化为 0（无偏）
        self.weights = [0.0] * n_features
        # AdaGrad 累积梯度平方
        self.grad_sq_sum = [1e-8] * n_features

    def predict_proba(self, features: list[float]) -> float:
        """预测 P(route=local 成功)"""
        z = sum(w * x for w, x in zip(self.weights, features))
        # sigmoid，clip 防止溢出
        z = max(-20.0, min(20.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def update(self, features: list[float], success: bool) -> None:
        """用一次路由结果更新权重"""
        prob = self.predict_proba(features)
        target = 1.0 if success else 0.0
        error = target - prob

        # AdaGrad SGD 更新
        for i in range(self.n_features):
            grad = error * features[i]
            self.grad_sq_sum[i] += grad * grad
            adaptive_lr = self.learning_rate / math.sqrt(max(1e-8, self.grad_sq_sum[i]))
            self.weights[i] += adaptive_lr * grad

    def to_dict(self) -> dict:
        return {
            "weights": [round(w, 6) for w in self.weights],
            "grad_sq_sum": [max(1e-6, round(g, 6)) for g in self.grad_sq_sum],
            "n_features": self.n_features,
            "learning_rate": self.learning_rate,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OnlineLogisticRegression":
        model = cls(n_features=d.get("n_features", 10), learning_rate=d.get("learning_rate", 0.1))
        model.weights = d.get("weights", [0.0] * model.n_features)
        model.grad_sq_sum = d.get("grad_sq_sum", [1e-8] * model.n_features)
        return model


# ─── Meta-Learner 管理器 ──────────────────────────────────────

class MetaLearner:
    """
    Meta-Learner：统一路由决策器。

    职责：
    1. 接收所有信号，提取特征
    2. 预测 P(本地成功)
    3. 根据阈值做出路由决策
    4. 从结果中在线学习
    """

    def __init__(self, cache_dir: str, decision_threshold: float = 0.5):
        self.cache_dir = cache_dir
        self.model_file = os.path.join(cache_dir, "meta_learner.json")
        self.history_file = os.path.join(cache_dir, "meta_learner_history.jsonl")
        self.decision_threshold = decision_threshold
        self._lock = threading.Lock()
        self.model = self._load_model()

    def _load_model(self) -> OnlineLogisticRegression:
        if os.path.exists(self.model_file):
            try:
                import json
                with open(self.model_file) as f:
                    return OnlineLogisticRegression.from_dict(json.load(f))
            except (json.JSONDecodeError, TypeError, KeyError):
                pass
        return OnlineLogisticRegression()

    def _save_model(self) -> None:
        import json
        with open(self.model_file, "w") as f:
            json.dump(self.model.to_dict(), f, indent=2)

    def predict(self, features: list[float]) -> dict:
        """
        预测路由决策。

        返回:
            {
                "should_use_local": bool,
                "local_success_prob": float,
                "confidence": float,  # 决策置信度（离阈值越远越确定）
            }
        """
        prob = self.model.predict_proba(features)
        should_local = prob >= self.decision_threshold
        # 决策置信度：离阈值越远越确定
        confidence = abs(prob - self.decision_threshold) * 2  # 归一化到 [0, 1]

        return {
            "should_use_local": should_local,
            "local_success_prob": round(prob, 4),
            "confidence": round(min(1.0, confidence), 4),
        }

    def record_and_learn(
        self,
        features: list[float],
        success: bool,
        route: str,
        task_type: str = "",
    ) -> None:
        """记录结果并在线学习"""
        with self._lock:
            # 在线学习
            self.model.update(features, success)
            self._save_model()

            # 记录历史
            entry = {
                "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "features": [round(f, 4) for f in features],
                "success": success,
                "route": route,
                "task_type": task_type,
                "predicted_prob": round(self.model.predict_proba(features), 4),
            }
            append_jsonl(self.history_file, entry)

    def get_feature_importance(self) -> dict:
        """获取特征重要性（权重绝对值）"""
        feature_names = [
            "complexity", "confidence", "entropy", "margin",
            "text_length", "file_count", "capability_success",
            "strategy_cot", "strategy_structured", "bias",
        ]
        importance = {}
        for name, weight in zip(feature_names, self.model.weights):
            importance[name] = round(weight, 4)
        return importance

    def get_stats(self) -> dict:
        """获取学习统计"""
        entries = read_jsonl(self.history_file)
        if not entries:
            return {
                "total": 0,
                "accuracy": 0.0,
                "feature_importance": self.get_feature_importance(),
                "model_weights": self.model.to_dict(),
            }

        # 计算预测准确率
        correct = 0
        for e in entries:
            predicted_prob = e.get("predicted_prob", 0.5)
            predicted_success = predicted_prob >= self.decision_threshold
            if predicted_success == e.get("success"):
                correct += 1

        return {
            "total": len(entries),
            "accuracy": round(correct / len(entries), 3),
            "feature_importance": self.get_feature_importance(),
            "model_weights": self.model.to_dict(),
        }


# ─── 主动学习（不确定性采样）────────────────────────────────────

class ActiveLearner:
    """
    主动学习：识别系统最不确定的任务类型，优先收集数据。

    核心思路：
    - 追踪每个任务类型的预测方差
    - 方差越大 = 系统越不确定 = 最需要更多数据
    - 在路由决策时，对不确定的任务类型主动请求云端验证
    """

    def __init__(self, cache_dir: str):
        self.data_file = os.path.join(cache_dir, "active_learning.jsonl")
        self._lock = threading.Lock()
        self._task_stats: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        entries = read_jsonl(self.data_file)
        for e in entries:
            tt = e.get("task_type", "unknown")
            if tt not in self._task_stats:
                self._task_stats[tt] = {"predictions": [], "outcomes": []}
            self._task_stats[tt]["predictions"].append(e.get("predicted_prob", 0.5))
            self._task_stats[tt]["outcomes"].append(e.get("success", False))

    # 每个任务类型最多保留最近 N 条记录，防止内存无限增长
    MAX_STATS_PER_TYPE = 1000

    def record(self, task_type: str, predicted_prob: float, success: bool) -> None:
        """记录一次预测和结果"""
        with self._lock:
            if task_type not in self._task_stats:
                self._task_stats[task_type] = {"predictions": [], "outcomes": []}
            stats = self._task_stats[task_type]
            stats["predictions"].append(predicted_prob)
            stats["outcomes"].append(success)
            # 裁剪：保留最近的数据
            if len(stats["predictions"]) > self.MAX_STATS_PER_TYPE:
                stats["predictions"] = stats["predictions"][-self.MAX_STATS_PER_TYPE:]
                stats["outcomes"] = stats["outcomes"][-self.MAX_STATS_PER_TYPE:]

            entry = {
                "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "task_type": task_type,
                "predicted_prob": round(predicted_prob, 4),
                "success": success,
            }
            append_jsonl(self.data_file, entry)

    def get_uncertainty(self, task_type: str) -> float:
        """
        获取某任务类型的不确定性（预测方差）。

        返回 [0, 1]，越大越不确定。
        """
        stats = self._task_stats.get(task_type)
        if not stats or len(stats["predictions"]) < 3:
            return 1.0  # 无数据 = 最不确定

        preds = stats["predictions"][-20:]  # 最近 20 次
        mean = sum(preds) / len(preds)
        variance = sum((p - mean) ** 2 for p in preds) / len(preds)
        return min(1.0, variance * 4)  # 归一化

    def get_most_uncertain(self, n: int = 3) -> list[dict]:
        """获取最不确定的任务类型"""
        results = []
        for tt, stats in self._task_stats.items():
            if len(stats["predictions"]) < 3:
                continue
            uncertainty = self.get_uncertainty(tt)
            recent_accuracy = sum(stats["outcomes"][-10:]) / max(1, len(stats["outcomes"][-10:]))
            results.append({
                "task_type": tt,
                "uncertainty": round(uncertainty, 4),
                "sample_count": len(stats["predictions"]),
                "recent_accuracy": round(recent_accuracy, 3),
            })

        results.sort(key=lambda x: x["uncertainty"], reverse=True)
        return results[:n]

    def should_request_verification(
        self, task_type: str, threshold: float = 0.3, min_samples: int = 5,
    ) -> bool:
        """
        判断是否应该请求云端验证（主动学习）。

        冷启动保护：样本数不足时不触发验证，避免每种新任务都多花 3 次云端调用。
        """
        stats = self._task_stats.get(task_type)
        if not stats or len(stats["predictions"]) < min_samples:
            return False  # 冷启动保护：数据不足不触发
        return self.get_uncertainty(task_type) > threshold

    def get_stats(self) -> dict:
        """获取主动学习统计"""
        uncertain = self.get_most_uncertain()
        total_types = len(self._task_stats)
        total_records = sum(len(s["predictions"]) for s in self._task_stats.values())
        return {
            "total_task_types": total_types,
            "total_records": total_records,
            "most_uncertain": uncertain,
        }


# ─── 全局实例 ──────────────────────────────────────────────

_meta_learner: Optional[MetaLearner] = None
_active_learner: Optional[ActiveLearner] = None
_init_lock = threading.Lock()


def get_meta_learner(cache_dir: Optional[str] = None) -> MetaLearner:
    global _meta_learner
    if _meta_learner is None:
        with _init_lock:
            if _meta_learner is None:
                if cache_dir is None:
                    from task_router.config import get_config
                    cache_dir = get_config().cache_dir
                _meta_learner = MetaLearner(cache_dir)
    return _meta_learner


def get_active_learner(cache_dir: Optional[str] = None) -> ActiveLearner:
    global _active_learner
    if _active_learner is None:
        with _init_lock:
            if _active_learner is None:
                if cache_dir is None:
                    from task_router.config import get_config
                    cache_dir = get_config().cache_dir
                _active_learner = ActiveLearner(cache_dir)
    return _active_learner
