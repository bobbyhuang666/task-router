"""
模型注册表 — 自动发现、评估和管理本地/云端模型

核心功能:
- 自动发现 Ollama 已安装模型
- 按任务类型评估模型能力（基准测试）
- 维护模型能力画像（准确率、速度、成本）
- 支持动态模型上下线
"""

import atexit
import os
import json
import time
import threading
import subprocess
from dataclasses import dataclass, field, asdict
from typing import Optional

from task_router.config import get_config

# ─── 模型能力画像 ──────────────────────────────────────────────────

@dataclass
class ModelProfile:
    name: str                          # 模型名称 (ollama name)
    backend: str = "ollama"            # 后端: ollama / vllm / llama_cpp / cloud
    parameter_size: str = ""           # 参数量: "1.5B", "3B", "7B"
    size_gb: float = 0.0               # 模型文件大小 (GB)
    max_context: int = 4096            # 最大上下文长度
    supports_tools: bool = False       # 是否支持工具调用
    supports_json: bool = False        # 是否支持 JSON 输出

    # 能力评分 (0-1, 通过基准测试获得)
    capabilities: dict = field(default_factory=lambda: {
        "classification": 0.0,
        "translation": 0.0,
        "extraction": 0.0,
        "summarization": 0.0,
        "formatting": 0.0,
        "reasoning": 0.0,
        "code": 0.0,
        "creative": 0.0,
    })

    # 性能指标
    avg_latency_ms: float = 0.0        # 平均延迟
    tokens_per_second: float = 0.0     # 生成速度
    benchmark_runs: int = 0            # 基准测试次数
    last_benchmark: str = ""           # 最后基准测试时间

    # 成本 (本地模型为0，云端按 API 价格)
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0

    # 状态
    available: bool = True
    last_used: str = ""
    total_calls: int = 0
    success_rate: float = 1.0


# ─── 基准测试集 ──────────────────────────────────────────────────

BENCHMARK_TESTS = {
    "classification": [
        {"input": "苹果, 香蕉, 橘子, 白菜, 萝卜, 牛肉", "expected": "水果", "action": "分类"},
        {"input": "iPhone15, MacBook, iPad, ThinkPad", "expected": "手机", "action": "分类"},
    ],
    "translation": [
        {"input": "Hello world", "expected": "你好世界", "action": "翻译成中文"},
        {"input": "Good morning", "expected": "早上好", "action": "翻译成中文"},
    ],
    "extraction": [
        {"input": "张三在北京工作，电话13800138000", "expected": "张三", "action": "提取人名"},
        {"input": "2024年1月15日发布", "expected": "2024", "action": "提取日期"},
    ],
    "sentiment": [
        {"input": "产品质量很好", "expected": "正面", "action": "判断情感"},
        {"input": "服务太差了", "expected": "负面", "action": "判断情感"},
    ],
}


# ─── 模型注册表 ──────────────────────────────────────────────────

class ModelRegistry:
    """
    模型注册表 — 管理所有可用模型及其能力画像。

    使用方法:
        registry = ModelRegistry()
        registry.discover()           # 自动发现 Ollama 模型
        registry.benchmark()          # 运行基准测试
        best = registry.select_best("translation")  # 选择最佳模型
    """

    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or get_config().cache_dir
        self.registry_file = os.path.join(self.cache_dir, "model_registry.json")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.models: dict[str, ModelProfile] = {}
        self._lock = threading.Lock()
        self._update_count = 0
        self._save_interval = 10  # 每 N 次更新才写磁盘
        atexit.register(self.flush)  # 进程退出时强制持久化
        self._load()

    def _load(self):
        """从磁盘加载注册表"""
        if not os.path.exists(self.registry_file):
            return
        try:
            with open(self.registry_file) as f:
                data = json.load(f)
            for name, info in data.items():
                self.models[name] = ModelProfile(**info)
        except (json.JSONDecodeError, TypeError):
            pass

    def _save(self):
        """持久化注册表"""
        data = {}
        for name, profile in self.models.items():
            data[name] = asdict(profile)
        tmp = self.registry_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.registry_file)

    def discover(self) -> list[str]:
        """
        自动发现 Ollama 已安装模型。
        返回新发现的模型列表。
        """
        new_models = []
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                return new_models

            for line in result.stdout.strip().split("\n")[1:]:  # 跳过表头
                parts = line.split()
                if not parts:
                    continue
                name = parts[0]
                size_str = parts[2] if len(parts) > 2 else "0"

                # 解析大小
                size_gb = 0.0
                if "GB" in size_str:
                    try:
                        size_gb = float(size_str.replace("GB", ""))
                    except ValueError:
                        pass

                # 检测参数量
                param_size = self._detect_param_size(name, size_gb)

                # 检测工具支持
                supports_tools = self._detect_tool_support(name)

                if name not in self.models:
                    self.models[name] = ModelProfile(
                        name=name,
                        parameter_size=param_size,
                        size_gb=size_gb,
                        supports_tools=supports_tools,
                    )
                    new_models.append(name)
                else:
                    # 更新已有模型信息
                    self.models[name].size_gb = size_gb
                    self.models[name].available = True

            self._save()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        return new_models

    def _detect_param_size(self, name: str, size_gb: float) -> str:
        """根据模型名称和大小推断参数量"""
        name_lower = name.lower()
        # 从名称推断（更长的模式优先匹配，避免 "3b" 误匹配 "13b"）
        for pattern, size in [
            ("1.5b", "1.5B"), ("0.5b", "0.5B"),
            ("72b", "72B"), ("70b", "70B"), ("32b", "32B"),
            ("14b", "14B"), ("13b", "13B"), ("8b", "8B"),
            ("7b", "7B"), ("3b", "3B"), ("e4b", "4B"), ("e2b", "2B"),
        ]:
            if pattern in name_lower:
                return size

        # 从大小推断 (Q4_K_M 量化)
        if size_gb < 1.2:
            return "1.5B"
        elif size_gb < 2.5:
            return "3B"
        elif size_gb < 5.0:
            return "7B"
        elif size_gb < 11.0:
            return "13B"
        elif size_gb < 20.0:
            return "32B"
        else:
            return "70B+"

    def _detect_tool_support(self, name: str) -> bool:
        """检测模型是否支持工具调用"""
        name_lower = name.lower()
        tool_keywords = ["tool", "qwen-tool", "llama3.1", "llama3.2", "mistral-nemo",
                         "command-r", "firefunction", "gorilla", "hermes"]
        return any(kw in name_lower for kw in tool_keywords)

    def select_best(self, capability: str, max_size_gb: float = None,
                    prefer_speed: bool = False) -> Optional[ModelProfile]:
        """
        为指定能力选择最佳模型。

        参数:
            capability: 能力类型 (classification/translation/extraction 等)
            max_size_gb: 最大模型大小限制
            prefer_speed: 是否优先选择速度更快的模型

        返回:
            最佳模型的 ModelProfile，或 None
        """
        candidates = []
        for name, profile in self.models.items():
            if not profile.available:
                continue
            if max_size_gb and profile.size_gb > max_size_gb:
                continue
            score = profile.capabilities.get(capability, 0.0)
            if score <= 0:
                # 未测试的模型给一个默认分（按大小估计）
                score = self._estimate_default_score(profile, capability)
            candidates.append((profile, score))

        if not candidates:
            return None

        if prefer_speed:
            # 速度优先：速度权重 0.4 + 能力权重 0.6
            # 未测试模型（延迟=0）使用默认延迟 1000ms
            def speed_score(item):
                profile, cap_score = item
                latency = profile.avg_latency_ms if profile.avg_latency_ms > 0 else 1000
                speed = 1.0 / latency  # 越快越高
                return cap_score * 0.6 + speed * 0.4
            candidates.sort(key=speed_score, reverse=True)
        else:
            # 能力优先
            candidates.sort(key=lambda x: x[1], reverse=True)

        return candidates[0][0]

    def _estimate_default_score(self, profile: ModelProfile, capability: str) -> float:
        """根据模型大小估计默认能力分"""
        size_scores = {
            "1.5B": {"classification": 0.75, "translation": 0.70, "extraction": 0.70,
                     "sentiment": 0.80, "formatting": 0.75, "summarization": 0.50,
                     "reasoning": 0.30, "code": 0.20},
            "3B":   {"classification": 0.85, "translation": 0.80, "extraction": 0.80,
                     "sentiment": 0.90, "formatting": 0.85, "summarization": 0.65,
                     "reasoning": 0.50, "code": 0.40},
            "7B":   {"classification": 0.92, "translation": 0.88, "extraction": 0.88,
                     "sentiment": 0.95, "formatting": 0.92, "summarization": 0.80,
                     "reasoning": 0.70, "code": 0.65},
            "13B":  {"classification": 0.95, "translation": 0.92, "extraction": 0.92,
                     "sentiment": 0.97, "formatting": 0.95, "summarization": 0.88,
                     "reasoning": 0.80, "code": 0.75},
        }
        # 按参数大小查找最接近的分数
        param = profile.parameter_size
        if param in size_scores:
            return size_scores[param].get(capability, 0.5)

        # 根据大小推断
        if profile.size_gb < 1.2:
            return size_scores["1.5B"].get(capability, 0.5)
        elif profile.size_gb < 2.5:
            return size_scores["3B"].get(capability, 0.5)
        elif profile.size_gb < 6.0:
            return size_scores["7B"].get(capability, 0.5)
        else:
            return size_scores["13B"].get(capability, 0.5)

    def update_after_call(self, model_name: str, success: bool,
                          latency_ms: float = 0, tokens_in: int = 0,
                          tokens_out: int = 0):
        """调用后更新模型统计（批量写盘）"""
        with self._lock:
            if model_name not in self.models:
                return
            profile = self.models[model_name]
            profile.total_calls += 1
            profile.last_used = time.strftime("%Y-%m-%dT%H:%M:%S")

            # 更新延迟（指数移动平均）
            if latency_ms > 0:
                if profile.avg_latency_ms == 0:
                    profile.avg_latency_ms = latency_ms
                else:
                    profile.avg_latency_ms = profile.avg_latency_ms * 0.8 + latency_ms * 0.2

            # 更新 tokens/s
            if latency_ms > 0 and tokens_out > 0:
                tps = tokens_out / (latency_ms / 1000)
                if profile.tokens_per_second == 0:
                    profile.tokens_per_second = tps
                else:
                    profile.tokens_per_second = profile.tokens_per_second * 0.8 + tps * 0.2

            # 更新成功率（指数移动平均）
            success_val = 1.0 if success else 0.0
            profile.success_rate = profile.success_rate * 0.9 + success_val * 0.1

            # 批量写盘：每 N 次更新才持久化
            self._update_count += 1
            if self._update_count >= self._save_interval:
                self._update_count = 0
                self._save()

    def flush(self) -> None:
        """强制持久化到磁盘（容错：目录不存在时不报错）"""
        try:
            with self._lock:
                self._save()
        except OSError:
            pass

    def run_benchmark(self, model_name: str = None, capabilities: list = None) -> dict:
        """
        运行基准测试评估模型能力。

        参数:
            model_name: 指定模型名称，None 则测试所有
            capabilities: 指定测试的能力列表

        返回:
            测试结果字典
        """
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from task_router import call_ollama

        if model_name:
            models_to_test = [model_name] if model_name in self.models else []
        else:
            models_to_test = [n for n, p in self.models.items() if p.available]

        caps_to_test = capabilities or list(BENCHMARK_TESTS.keys())
        results = {}

        for mname in models_to_test:
            profile = self.models[mname]
            cap_scores = {}
            cap_latencies = {}

            for cap in caps_to_test:
                if cap not in BENCHMARK_TESTS:
                    continue
                tests = BENCHMARK_TESTS[cap]
                correct = 0
                total_latency = 0

                for test in tests:
                    prompt = f"{test['action']}：{test['input']}\n只输出结果："
                    try:
                        result = call_ollama(prompt, model=mname, max_tokens=32)
                        output = result.get("text", "").strip()
                        total_latency += result.get("time_ms", 0)

                        # 检查是否包含期望值
                        if test["expected"].lower() in output.lower():
                            correct += 1
                    except Exception:
                        pass

                score = correct / len(tests) if tests else 0
                cap_scores[cap] = round(score, 2)
                cap_latencies[cap] = total_latency / len(tests) if tests else 0

            # 更新模型画像
            for cap, score in cap_scores.items():
                profile.capabilities[cap] = score

            avg_latency = sum(cap_latencies.values()) / len(cap_latencies) if cap_latencies else 0
            profile.avg_latency_ms = avg_latency
            profile.benchmark_runs += 1
            profile.last_benchmark = time.strftime("%Y-%m-%dT%H:%M:%S")

            results[mname] = {
                "capabilities": cap_scores,
                "avg_latency_ms": round(avg_latency),
                "benchmark_runs": profile.benchmark_runs,
            }

        self._save()
        return results

    def get_summary(self) -> str:
        """返回模型注册表摘要"""
        if not self.models:
            return "模型注册表为空。运行 discover() 发现模型。"

        lines = ["模型注册表", "=" * 60]
        for name, p in sorted(self.models.items(), key=lambda x: x[1].size_gb):
            status = "✓" if p.available else "✗"
            caps = ", ".join(f"{k}:{v:.0%}" for k, v in p.capabilities.items() if v > 0)
            latency = f"{p.avg_latency_ms:.0f}ms" if p.avg_latency_ms > 0 else "未测试"
            speed = f"{p.tokens_per_second:.0f}t/s" if p.tokens_per_second > 0 else ""
            lines.append(
                f"  {status} {name:30} {p.parameter_size:>5} {p.size_gb:.1f}GB "
                f"延迟:{latency:>8} {speed:>8} 调用:{p.total_calls} "
                f"成功率:{p.success_rate:.0%}"
            )
            if caps:
                lines.append(f"    能力: {caps}")

        return "\n".join(lines)

    def get_status(self) -> dict:
        """返回模型状态的字典表示"""
        return {
            name: {
                "available": p.available,
                "parameter_size": p.parameter_size,
                "size_gb": p.size_gb,
                "capabilities": p.capabilities,
                "avg_latency_ms": p.avg_latency_ms,
                "tokens_per_second": p.tokens_per_second,
                "total_calls": p.total_calls,
                "success_rate": p.success_rate,
            }
            for name, p in self.models.items()
        }


# ─── CLI 入口 ──────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    registry = ModelRegistry()

    if len(sys.argv) > 1 and sys.argv[1] == "--discover":
        new = registry.discover()
        print(f"发现 {len(new)} 个新模型: {', '.join(new) if new else '(无)'}")
    elif len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        model = sys.argv[2] if len(sys.argv) > 2 else None
        results = registry.run_benchmark(model)
        for m, r in results.items():
            print(f"\n{m}:")
            for cap, score in r["capabilities"].items():
                bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
                print(f"  {cap:15} {bar} {score:.0%}")
    elif len(sys.argv) > 1 and sys.argv[1] == "--select":
        cap = sys.argv[2] if len(sys.argv) > 2 else "classification"
        best = registry.select_best(cap)
        if best:
            print(f"最佳 {cap} 模型: {best.name} (延迟:{best.avg_latency_ms:.0f}ms)")
        else:
            print(f"没有可用的 {cap} 模型")
    else:
        registry.discover()
        print(registry.get_summary())
