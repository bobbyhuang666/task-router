# Self-Reflective Routing (SRR) — 设计文档

> **核心创新**：让 LLM 路由系统具备"反思"能力 — 不仅记录路由结果，
> 还能分析"为什么路由错了"，并自动修正决策参数。

## 1. 动机

现有 TaskRouter v6.0 的学习机制是**纯统计的**：

| 组件 | 学习方式 | 问题 |
|------|---------|------|
| WeightTracker | 成功+1/失败-1 调阈值 | 不知道**为什么**成功或失败 |
| StrategyTracker | 统计 task_type×strategy 成功率 | 不知道**替换策略会不会更好** |
| ClosedLoopManager | 评估蒸馏对质量 | 只评蒸馏，**不反思路由决策** |
| CapabilityTracker | 成功率追踪 | 不知道**任务特征和路由的关系** |

所有组件都在做**标量信号**（success/fail）的增量学习。
它们无法回答这些问题：

- "这个翻译任务走了云端，但本地模型其实能搞定吗？"
- "用 CoT 跑这个分类任务，token 浪费了 3 倍，direct 就够了"
- "这类法律相关任务本地一直失败，是不是该永久升到云端？"

**Self-Reflective Routing** 用 LLM-as-Judge 替代标量信号，
让系统能做**多维度、语义级别的反思分析**。

## 2. 架构总览

```
┌─────────────────────────────────────────────────────┐
│                 TaskRouter v6.0+                     │
├─────────────────────────────────────────────────────┤
│                                                     │
│  run_task() ──→ EpisodeCollector.record()            │
│       │              ↓                              │
│       │         episodes.jsonl (完整决策快照)         │
│       │              ↓                              │
│       │      ┌───────────────────┐                  │
│       │      │  ReflectionEngine │  ← 周期性运行     │
│       │      │                   │                  │
│       │      │  1. QualityJudge  │  ← LLM-as-Judge  │
│       │      │     多维度打分     │                  │
│       │      │                   │                  │
│       │      │  2. RouteAnalyzer │  ← 路由决策分析   │
│       │      │     错误路由识别   │                  │
│       │      │                   │                  │
│       │      │  3. StrategyReflector │ ← 策略反思    │
│       │      │     策略×模型配对  │                  │
│       │      │                   │                  │
│       │      │  4. CorrectionProposer │ ← 修正建议   │
│       │      │     参数更新方案   │                  │
│       │      └───────┬───────────┘                  │
│       │              ↓                              │
│       │      corrections.json                       │
│       │              ↓                              │
│       │      CorrectionApplier.apply()              │
│       │              ↓                              │
│       └── a3m_weights.json (更新)                    │
│          strategy_params.json (新增)                 │
│          routing_policies.json (新增)                │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## 3. 数据结构

### 3.1 Episode（路由决策快照）

每次 `run_task()` 执行后自动记录的完整决策快照：

```python
@dataclass
class Episode:
    """一次路由决策的完整快照"""
    episode_id: str           # 唯一标识
    timestamp: str            # ISO 时间戳

    # 输入
    action: str               # 任务描述
    text: str                 # 输入文本（截断前 500 字符）
    task_type: str            # 检测到的任务类型
    capability: str           # 能力分类

    # 决策信号
    complexity_score: float   # A3M 复杂度评分
    confidence_data: dict     # 置信度信号 (entropy, margin, confidence)
    strategy: str             # 选择的推理策略
    strategy_reason: str      # 策略选择原因

    # 路由决策
    route: str                # local / cloud / cascade_escalated / cache(...)
    model_used: str           # 实际使用的模型
    routing_signals: dict     # 五层决策信号快照
    # {cascade: {escalate, confidence},
    #  meta_learner: {should_use_local, confidence},
    #  active_verify: bool,
    #  tqbc: {should_escalate, regret_bound},
    #  conformal: {prediction_set, interval_width}}

    # 输出
    output: str               # 输出文本（截断前 500 字符）
    tokens_input: int
    tokens_output: int
    time_ms: int
    cost_saved: float

    # 后验（反思阶段填充）
    quality_scores: dict      # LLM-as-Judge 的多维度评分
    optimal_route: str        # 反思认为的最优路由
    optimal_strategy: str     # 反思认为的最优策略
    routing_error: str        # 路由错误类型（如无错误则为空）
    reflection_notes: str     # 反思备注
```

### 3.2 Correction（修正建议）

反思引擎产出的参数修正方案：

```python
@dataclass
class Correction:
    """一次参数修正"""
    correction_id: str
    timestamp: str
    trigger_episode_ids: list[str]  # 触发修正的 episode ID 列表

    # 修正目标
    target: str        # "threshold" / "strategy_weight" / "routing_policy"
    parameter: str     # 具体参数名
    old_value: Any
    new_value: Any

    # 修正依据
    reason: str                    # 自然语言解释
    confidence: float              # 修正置信度 [0, 1]
    expected_impact: str           # 预期效果描述
    evidence_count: int            # 支持证据数量
```

## 4. 模块设计

### 4.1 EpisodeCollector（`src/task_router/episode_collector.py`）

**职责**：在 `run_task()` 执行后自动收集决策快照。

**设计原则**：
- **零侵入**：只读取 Task 对象的属性，不修改任何执行逻辑
- **延迟写入**：内存缓冲，批量 flush 到磁盘
- **PII 安全**：action/text 截断且可选脱敏

```python
class EpisodeCollector:
    def __init__(self, cache_dir: str, scrub_pii: bool = True):
        self.episodes_file = os.path.join(cache_dir, "episodes.jsonl")
        self._buffer: list[dict] = []
        self._lock = threading.Lock()

    def record(self, task: Task, routing_context: dict) -> None:
        """记录一次路由决策快照。

        routing_context 包含：
        - complexity_score: float
        - confidence_data: dict
        - strategy: str
        - strategy_reason: str
        - routing_signals: dict (五层信号)
        - task_type: str
        - capability: str
        """

    def flush(self) -> None:
        """将缓冲写入磁盘"""

    def get_recent(self, n: int = 100) -> list[dict]:
        """获取最近 N 条 episode"""
```

**集成点**（对 `task_router.py` 的改动）：

```python
# task_router.py — run_task() 末尾新增

# 在 _finalize_task(task) 之后
_episode_collector.record(task, routing_context={
    "complexity_score": routing_score,
    "confidence_data": conf_data,
    "strategy": result.get("strategy", "direct"),
    "strategy_reason": "",
    "routing_signals": routing_signals,
    "task_type": task_type,
    "capability": TASK_TO_CAPABILITY.get(task_type, ""),
})
```

### 4.2 QualityJudge（`src/task_router/quality_judge.py`）

**职责**：用 LLM-as-Judge 对 episode 做多维度质量评估。

**关键设计**：LLM Judge 不是每次都调用，而是**批量处理**已收集的 episodes。
调用者可以是：
- `ReflectionEngine`（周期性自动）
- CLI 命令 `sma --reflect`（手动触发）
- 测试代码（用 mock LLM）

```python
class QualityJudge:
    """LLM-as-Judge 多维度质量评估器。

    与 distillation.py 的 QualityEvaluator 的区别：
    - QualityEvaluator: 规则-based，评估蒸馏对质量
    - QualityJudge: LLM-based，评估路由决策 + 输出质量

    QualityJudge 评估的维度：
    1. relevance:    输出与任务的相关性 [0, 1]
    2. completeness:  输出的完整性 [0, 1]
    3. accuracy:     输出的准确性 [0, 1]
    4. efficiency:   路由决策的效率 [0, 1]（token 和成本是否合理）
    5. correctness:  路由选择是否正确 [0, 1]（该走本地/云端吗？）
    """

    JUDGE_PROMPT_TEMPLATE = """你是一个 LLM 路由系统的质量评估专家。
请评估以下路由决策和输出质量。

## 任务信息
- 任务描述: {action}
- 任务类型: {task_type}
- 路由决策: {route}
- 使用模型: {model_used}
- 推理策略: {strategy}
- Token 消耗: 输入 {tokens_input}, 输出 {tokens_output}

## 输入文本
{text}

## 系统输出
{output}

## 评估维度（每项 0-10 分）

请用 JSON 格式输出评分：
{{
    "relevance": <分数>,      // 输出是否正确回应了任务需求
    "completeness": <分数>,    // 输出是否完整（不是半成品）
    "accuracy": <分数>,        // 输出内容是否准确
    "efficiency": <分数>,      // 路由和策略选择是否经济高效
    "correctness": <分数>,     // 路由选择是否正确（local/cloud）
    "optimal_route": "<local|cloud>",           // 你认为最优的路由
    "optimal_strategy": "<direct|cod|cot|few_shot|structured>",  // 你认为最优的策略
    "routing_error": "<none|over_escalated|under_escalated|wrong_strategy|token_waste>",
    "notes": "<一句话说明>"
}}"""

    def __init__(self, cache_dir: str):
        self.judgments_file = os.path.join(cache_dir, "quality_judgments.jsonl")

    def judge_episode(self, episode: dict) -> dict:
        """评估单个 episode（调用 LLM）。

        返回与 episode_id 关联的评分 dict。
        """

    def judge_batch(self, episodes: list[dict], batch_size: int = 5) -> list[dict]:
        """批量评估（合并 prompt 以减少 LLM 调用）。

        每次调用评估 batch_size 个 episode。
        """

    def get_judgment(self, episode_id: str) -> Optional[dict]:
        """获取已有的评分"""

    def get_unjudged_episodes(self, episodes: list[dict]) -> list[dict]:
        """筛选尚未评分的 episode"""
```

**LLM 调用策略**（节省成本）：

1. **单 episode prompt**：一次 LLM 调用评估 1 个 episode（精确，用于小批量）
2. **批量 prompt**：将 3-5 个 episode 合并到一个 prompt 中评估（省 token，用于大批量）
3. **跳过已评估**：通过 episode_id 去重，避免重复评估
4. **优先级排序**：先评估"边界决策"（complexity_score 接近 threshold 的）

### 4.3 ReflectionEngine（`src/task_router/reflection.py`）

**职责**：读取 episodes + judgments，分析路由模式，生成修正建议。

这是 SRR 的核心创新模块。

```python
class ReflectionEngine:
    """自反思路由引擎。

    三层反思：
    1. 路由反思 (RouteAnalyzer): 哪些任务路由选错了？
    2. 策略反思 (StrategyReflector): 哪些策略选择浪费了 token？
    3. 联合反思 (JointReflector): (model, strategy) 配对最优吗？

    反思产出：
    - corrections: 参数修正建议列表
    - insights: 人类可读的分析报告
    """

    def __init__(self, cache_dir: str):
        self.judge = QualityJudge(cache_dir)
        self.corrections_file = os.path.join(cache_dir, "corrections.jsonl")
        self.insights_file = os.path.join(cache_dir, "reflection_insights.json")
        self._collector = EpisodeCollector(cache_dir)

    # ── 主入口 ──

    def reflect(self, n_episodes: int = 100) -> ReflectionReport:
        """运行一次完整反思。

        步骤：
        1. 读取最近 N 条 episode
        2. 对未评估的 episode 调用 QualityJudge
        3. 运行三层分析
        4. 生成修正建议
        5. 写入报告

        返回 ReflectionReport
        """

    # ── 路由反思 ──

    def _analyze_routing_errors(self, judged_episodes: list[dict]) -> list[dict]:
        """识别路由决策错误。

        分析维度：
        - over_escalated: 该走本地但走了云端（浪费钱）
        - under_escalated: 该走云端但走了本地（质量差）
        - optimal: 路由正确

        返回按任务类型聚合的错误统计。
        """

    # ── 策略反思 ──

    def _analyze_strategy_waste(self, judged_episodes: list[dict]) -> list[dict]:
        """识别策略选择的 token 浪费。

        分析维度：
        - token_waste: 策略消耗了过多 token（如用 CoT 跑简单分类）
        - under_reasoning: 策略过于简单（如用 direct 跑复杂推理）

        返回按 task_type×strategy 聚合的浪费统计。
        """

    # ── 联合反思 ──

    def _analyze_joint_pairs(self, judged_episodes: list[dict]) -> list[dict]:
        """分析 (route, strategy) 联合配对的效率。

        核心问题：是否存在某个 (local/cod) 配对，
        比当前常用的 (local/cot) 效果一样好但 token 消耗少 50%？

        返回最优配对建议。
        """

    # ── 修正生成 ──

    def _generate_corrections(self, analysis: dict) -> list[Correction]:
        """根据分析结果生成参数修正建议。

        修正类型：
        1. threshold: 调整 base_threshold（更激进/保守地路由到本地）
        2. strategy_weight: 调整策略选择偏好
        3. routing_policy: 新增硬规则（如"法律类任务强制云端"）
        """


@dataclass
class ReflectionReport:
    """反思报告"""
    timestamp: str
    episodes_analyzed: int

    # 路由分析
    routing_accuracy: float         # 路由正确率
    over_escalation_rate: float     # 过度升级率（该本地走了云端）
    under_escalation_rate: float    # 升级不足率（该云端走了本地）
    routing_errors_by_type: dict    # 按任务类型的错误统计

    # 策略分析
    avg_token_waste_ratio: float    # 平均 token 浪费率
    strategy_errors: list[dict]     # 策略选择错误列表

    # 联合分析
    joint_recommendations: list[dict]  # 最优 (route, strategy) 配对

    # 修正建议
    corrections: list[Correction]

    # 人类可读摘要
    summary: str
```

### 4.4 CorrectionApplier（`src/task_router/correction_applier.py`）

**职责**：将修正建议应用到路由参数。

```python
class CorrectionApplier:
    """将修正建议应用到系统参数。

    安全机制：
    1. 置信度过滤：只应用 confidence >= 0.6 的修正
    2. 幅度限制：单次修正不超过 ±20%
    3. 回滚能力：保存历史参数，支持回滚
    4. 审批模式：高风险修正写入待审批队列
    """

    def __init__(self, cache_dir: str):
        self.history_file = os.path.join(cache_dir, "correction_history.jsonl")
        self.pending_file = os.path.join(cache_dir, "corrections_pending.jsonl")

    def apply_corrections(self, corrections: list[Correction]) -> dict:
        """应用修正建议。

        返回: {"applied": int, "skipped": int, "pending": int, "details": list}
        """

    def _apply_threshold(self, correction: Correction) -> bool:
        """应用阈值修正 → 更新 a3m_weights.json"""

    def _apply_strategy_weight(self, correction: Correction) -> bool:
        """应用策略权重修正 → 更新 strategy_params.json（新增文件）"""

    def _apply_routing_policy(self, correction: Correction) -> bool:
        """应用路由策略修正 → 更新 routing_policies.json（新增文件）"""

    def rollback(self, correction_id: str) -> bool:
        """回滚指定修正"""

    def get_pending(self) -> list[dict]:
        """获取待审批的修正"""
```

## 5. 与现有系统的集成

### 5.1 对 `task_router.py` 的改动（最小侵入）

```python
# ── 新增 import ──
from task_router.episode_collector import get_episode_collector

# ── run_task() 中新增 2 行 ──
def run_task(task: Task, force_route: str = "", _depth: int = 0) -> Task:
    # ... 现有逻辑不变 ...

    # 5. 学习反馈（已有）
    wt.record_outcome(...)

    # 6. [新增] 收集路由快照
    get_episode_collector().record(task, routing_context={
        "complexity_score": routing_score,
        "confidence_data": ...,
        "strategy": ...,
        ...
    })

    return task
```

**总计改动：1 个 import + 约 10 行代码。**

### 5.2 对 `cli.py` 的改动（新增命令）

```python
# 新增 CLI 命令
--reflect           # 运行一次反思分析
--reflect-status    # 查看反思状态
--corrections       # 查看修正建议
--corrections-apply # 应用修正建议
```

### 5.3 对 `api_server.py` 的改动（新增端点）

```python
POST /api/reflect          # 触发反思分析
GET  /api/reflect/status   # 反思状态
GET  /api/corrections      # 修正建议
POST /api/corrections/apply # 应用修正
```

## 6. 反思周期设计

### 自动反思

```
每 50 条新 episode → 自动触发一次反思
每 24 小时 → 至少触发一次反思（即使 episode 不足）
```

### 手动反思

```bash
sma --reflect                    # 反思最近 100 条
sma --reflect --last 500         # 反思最近 500 条
sma --reflect --task-type translate  # 只反思翻译任务
```

### 反思→修正 流程

```
1. EpisodeCollector 收集 episodes
2. QualityJudge 对未评估的 episodes 评分
3. ReflectionEngine 分析评分数据
4. ReflectionEngine 生成 Correction 列表
5. CorrectionApplier:
   - confidence >= 0.8 → 自动应用
   - 0.6 <= confidence < 0.8 → 写入 pending，等待 CLI 审批
   - confidence < 0.6 → 忽略（仅记录在报告中）
6. 写入反思报告（reflection_insights.json）
```

## 7. 文件结构

```
src/task_router/
├── episode_collector.py    [新增] 路由决策快照收集
├── quality_judge.py        [新增] LLM-as-Judge 多维度评估
├── reflection.py           [新增] 三层反思引擎
├── correction_applier.py   [新增] 参数修正应用器
├── task_router.py          [改动] +1 import, +10 行
├── cli.py                  [改动] +4 命令
├── api_server.py           [改动] +4 端点
└── ... (其余文件不变)

tests/
├── test_episode_collector.py   [新增]
├── test_quality_judge.py       [新增]
├── test_reflection.py          [新增]
└── test_correction_applier.py  [新增]

数据文件（运行时自动生成）:
~/.cache/task_router/
├── episodes.jsonl           [新增] 路由快照
├── quality_judgments.jsonl  [新增] LLM 评分
├── corrections.jsonl        [新增] 修正记录
├── corrections_pending.jsonl [新增] 待审批修正
├── correction_history.jsonl [新增] 修正历史
├── reflection_insights.json [新增] 反思报告
├── strategy_params.json     [新增] 策略参数
└── routing_policies.json    [新增] 路由策略
```

## 8. 测试策略

### 不需要真实 LLM 的测试

所有模块都设计为可注入 mock LLM：

```python
# test_quality_judge.py
def test_judge_episode_with_mock():
    judge = QualityJudge(cache_dir=tmpdir)
    judge._call_llm = mock_llm  # 注入 mock
    result = judge.judge_episode(sample_episode)
    assert result["relevance"] >= 0
    assert result["routing_error"] in ("none", "over_escalated", ...)
```

### 测试用例规划

| 模块 | 测试数 | 重点 |
|------|--------|------|
| EpisodeCollector | 8 | 记录完整性、PII 截断、缓冲 flush、线程安全 |
| QualityJudge | 12 | 评分解析、批量处理、去重、mock LLM 集成 |
| ReflectionEngine | 15 | 路由错误识别、策略浪费检测、联合分析、修正生成 |
| CorrectionApplier | 12 | 阈值修正、策略修正、回滚、安全限制、审批流程 |
| **集成测试** | 5 | 端到端：收集→评估→反思→修正→验证参数更新 |
| **总计** | **52** | |

## 9. 论文叙事

**标题**：Self-Reflective Routing: LLM-as-Judge for Adaptive Cost-Optimized Inference

**核心贡献**：
1. **首次提出**"自反思路由"概念 — 路由系统能分析自身决策并自动修正
2. **LLM-as-Judge** 替代标量信号做多维度路由评估
3. **三层反思框架**：路由→策略→联合，逐层递进
4. **闭环自修正**：反思结果自动更新路由参数，无需人工干预

**与现有工作的区别**：

| 维度 | RouteLLM | PILOT/BARP | RouteNLP | **SRR (本方案)** |
|------|----------|------------|----------|-----------------|
| 学习信号 | 偏好标签 | bandit 奖励 | 失败聚类 | **LLM 语义评估** |
| 反思维度 | 无 | 无 | 失败模式 | **路由+策略+联合** |
| 修正粒度 | 无 | 无 | 蒸馏数据 | **阈值+策略+策略** |
| 人工干预 | 需标注 | 需设计 | 需分析 | **全自动** |

## 10. 实施顺序

```
Phase 1 (核心): EpisodeCollector + QualityJudge
  ↓ 验证：收集 100 条 episode，跑一次 LLM 评分
Phase 2 (创新): ReflectionEngine
  ↓ 验证：对评分数据做三层分析，输出修正建议
Phase 3 (闭环): CorrectionApplier + CLI/API
  ↓ 验证：端到端自动修正
Phase 4 (测试): 补齐 52 个测试用例
Phase 5 (文档): 更新 README + CHANGELOG
```
