# TaskRouter v6.0 — 创新研究报告

## 摘要

本报告记录了 TaskRouter 企业 LLM 网关的六大创新，将系统从关键词匹配的静态路由升级为基于 Token 级不确定性的自适应路由系统。核心创新 TQBC (Token-Quantile Bayesian Cascade) 在边界场景下的路由准确率达到 87.5%，相比基线系统 37.5% 提升 50 个百分点。

---

## 1. 创新架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                    TaskRouter v6.0 创新架构                   │
├─────────────────────────────────────────────────────────────┤
│  Layer 5: 自适应 Prompt 压缩 (置信度驱动的 Token 预算优化)      │
│  Layer 4: Gatekeeper 置信度分离 (动态阈值调整)                  │
│  Layer 3: OATS 缓存质量感知 (结果反馈驱动的缓存优先级)          │
│  Layer 2: 自适应推理策略 (Chain-of-Draft + Token 分位数)       │
│  Layer 1: TQBC 路由决策 (Thompson Sampling + 贝叶斯校准)       │
│  Layer 0: Token 分位数特征提取 (解决长度偏差问题)               │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 核心创新：TQBC (Token-Quantile Bayesian Cascade)

### 2.1 问题定义

现有 LLM 路由系统存在三个关键问题：
1. **长度偏差**：序列级不确定性估计受序列长度影响，短序列和长序列不可比
2. **探索-利用困境**：固定阈值无法适应任务分布变化
3. **校准不足**：置信度分数与实际准确率不匹配

### 2.2 解决方案

**Token 分位数特征提取**（来自 "Language Model Cascades", ICML 2024）
- 使用 q25/q50/q75/q90 分位数替代简单平均
- 解决长度偏差：中位数对序列长度鲁棒
- 8 维特征向量：熵分位数、边际分位数、方差、首 token 信号

**Thompson Sampling 路由**（来自 PILOT/EMNLP 2025）
- 贝叶斯线性回归为每个路由臂建模
- 自动平衡探索与利用
- 遗忘因子支持非平稳环境
- 冷启动保护（贝叶斯先验编码初始不确定性）

**贝叶斯置信度校准**（来自 Multicalibration, ICML 2024）
- Platt 缩放的贝叶斯版本
- 按任务类型的分组校准
- 自适应学习率 SGD 更新

### 2.3 实验结果

| 指标 | 基线系统 | TQBC 系统 | 改进 |
|------|----------|-----------|------|
| 路由准确率 | 37.5% | 87.5% | +50.0% |
| Token 节省 | 0% | 5.3% | +5.3% |
| 缓存精确度 | 94.3% | 94.1% | -0.2% |

**关键发现**：在边界场景（复杂度不能唯一决定路由）下，TQBC 的 logprobs 驱动方法显著优于纯复杂度阈值。

---

## 3. 创新二：Gatekeeper 置信度分离

### 3.1 灵感来源

Gatekeeper (arXiv 2502.19335)：通过自定义损失函数增加正确/错误预测之间的置信度间隔。

### 3.2 实现

- 追踪正确预测和错误预测的平均置信度
- 计算置信度间隔 (gap)
- 根据间隔大小动态调整升级阈值：
  - 大间隔 → 降低阈值（更信任本地）
  - 小间隔 → 提高阈值（更谨慎）

### 3.3 实验结果

- 置信度间隔：0.1934
- 阈值调整：+0.0307（略微更保守）

---

## 4. 创新三：Chain-of-Draft 极简推理

### 4.1 灵感来源

Chain of Draft 论文：仅使用 7.6% 的推理 token 达到 Chain-of-Thought 的准确率。

### 4.2 实现

- 新增 "cod" 策略，token 预算因子 0.3
- Prompt：「用最少的词写出关键推理步骤，然后给出答案。」
- 集成到自适应推理策略选择器

### 4.3 Token 节省

| 策略 | Token 预算倍数 | 相对节省 |
|------|---------------|----------|
| direct | 1.0x | 基准 |
| cod | 0.3x | 70% |
| few_shot | 1.5x | -50% |
| cot | 1.8x | -80% |
| structured | 1.3x | -30% |

---

## 5. 创新四：OATS 缓存质量感知

### 5.1 灵感来源

OATS (Outcome-Aware Tool Selection)：将结果反馈应用到缓存优先级。

### 5.2 实现

- 每个缓存条目有质量分（基于历史成功率）
- EMA (指数移动平均) 更新质量分
- 质量感知阈值调整：
  - 高质量条目 → 降低阈值（更容易命中）
  - 低质量条目 → 提高阈值（更难命中）

### 5.3 实验结果

| 指标 | 固定阈值 | 质量感知 | 改进 |
|------|----------|----------|------|
| 命中率 | 23.3% | 28.2% | +4.9% |
| 精确度 | 94.3% | 94.1% | -0.2% |

---

## 6. 创新五：自适应 Prompt 压缩

### 6.1 灵感来源

- LLM-DCP (arXiv 2504.11004): MDP 压缩，6.9x 压缩率
- Selection-p (EMNLP 2024): 自监督 token 选择，10x 压缩

### 6.2 实现

根据置信度选择压缩级别：
- confidence >= 0.8 → 激进压缩（保留 40%）
- confidence >= 0.6 → 中度压缩（保留 60%）
- confidence >= 0.4 → 轻度压缩（保留 80%）
- confidence < 0.4 → 不压缩

语义重要性评估：
- 指令性句子 → 高重要性
- 包含数字/格式要求 → 高重要性
- 普通描述 → 中等重要性

---

## 7. 创新六：策略反馈学习循环

### 7.1 设计

- 追踪每种推理策略在每种任务类型上的成功率
- 历史数据驱动的策略选择
- 滑动窗口（最近 30 次）防止旧数据影响

### 7.2 学习循环

```
1. 选择策略（基于复杂度 + 历史数据）
2. 执行任务
3. 记录结果（成功/失败）
4. 更新策略效果统计
5. 下次选择时参考历史数据
```

---

## 8. 参考文献

| 论文 | 来源 | 核心贡献 |
|------|------|----------|
| Language Model Cascades | ICML 2024 | Token 分位数解决长度偏差 |
| PILOT | EMNLP 2025 | 上下文老虎机 LLM 路由 |
| OATS | vLLM-SR 2026 | 结果感知的工具选择 |
| Chain of Draft | - | 极简推理，7.6% token |
| Multicalibration | ICML 2024 | 分组校准 |
| Gatekeeper | arXiv 2502.19335 | 置信度分离训练 |
| LACIE | NeurIPS 2024 | 感知器感知校准 |
| RouteLLM | ICLR 2025 | 偏好数据路由 |
| LLM-DCP | arXiv 2504.11004 | MDP 压缩 |
| FrugalGPT | - | 级联推理成本优化 |

---

## 9. 代码架构

```
scripts/
├── reasoning.py               # 统一推理策略选择器（关键词 + Token 分位数 + 反馈学习）
├── tqbc.py                    # TQBC 核心（Token 分位数 + Thompson Sampling + 贝叶斯校准）
├── adaptive_reasoning.py      # 向后兼容包装（已合并到 reasoning.py）
├── adaptive_compression.py    # 自适应 Prompt 压缩 + 重要性权重学习
├── outcome_cache.py           # OATS 缓存质量感知 + flush 持久化
├── task_router.py             # 主编排器（集成所有创新）
├── routing.py                 # A3M 复杂度估计
├── confidence.py              # 置信度门控级联（Cascade PAV 校准）
├── meta_learner.py            # 在线学习元学习器
├── cache.py                   # 语义缓存
├── models.py                  # 模型调用
└── ...

tests/
├── test_tqbc.py               # TQBC 单元测试（35 个）
├── test_tqbc_integration.py   # TQBC 集成测试（10 个）
├── test_adaptive_reasoning.py # 自适应推理测试（25 个）
├── test_adaptive_compression.py # 自适应压缩测试（18 个）
├── test_outcome_cache.py      # 缓存质量测试（22 个）
└── ...

benchmark_tqbc.py              # 综合基准测试
benchmark_learning_curve.py    # 学习曲线基准测试
```

### 架构简化的关键设计决策

**推理模块合并**（v6.1）：原系统有两套推理策略选择器：
- `reasoning.py`：关键词 + 复杂度启发式（无 logprobs）
- `adaptive_reasoning.py`：Token 分位数驱动（有 logprobs）

合并后 `reasoning.py` 成为唯一入口，`select_strategy()` 根据是否有 logprobs 自动选择：
- 有 logprobs → Token 分位数驱动决策树
- 无 logprobs → 关键词 + 复杂度启发式

`StrategyFeedbackTracker` 也合并进 `StrategyTracker`，减少一个全局单例。

---

## 10. 测试覆盖

- **总测试数**：435 个
- **通过率**：100%
- **覆盖模块**：所有核心创新模块

---

## 11. 后续优化方向

1. **级联路由**（Cascade Routing, arXiv 2410.10347）：统一路由+级联范式
2. **内部置信度**（Internal Confidence, OpenReview 2025）：单次前向传播的预生成不确定性
3. **温度缩放校准**：更简单的校准方法，可能更适合在线学习
4. **嵌入式语义路由**：替代关键词匹配的任务类型检测
5. **多模型级联**：本地小模型 → 本地大模型 → 云端模型的三级级联
6. **Cascade 合并 TQBC 校准**：当前 PAV + 贝叶斯两套校准并行，可考虑统一

---

## 12. 代码质量改进记录

### v6.1 架构简化（2026-05-28）

| 问题 | 修复 |
|------|------|
| Box-Muller 用 hash() 生成随机数 | 改用 `random.gauss(0, 1)` PRNG |
| 噪声方差所有臂共享 | 每臂独立噪声方差 + 在线方差估计 |
| 2 套推理策略选择器 | 合并到 `reasoning.py` 统一入口 |
| 2 个策略反馈追踪器 | 合并到 `StrategyTracker` |
| 函数内大量 import | 移至文件顶部 |
| estimate_importance 无学习 | 添加 `ImportanceLearner` 权重自适应 |
| outcome_cache 无 flush | 添加 `flush()` 持久化方法 |
| Platt 校准使用非标准 logit | 改为标准 logit 变换 |
| precision_inv 无上限 | 添加 `MAX_PRECISION_INV = 100.0` 防爆 |
| Calibeating 分桶校准 | 添加 40% Platt + 60% 经验混合 |

---

*报告生成时间：2026-05-28*
*TaskRouter v6.0 — 企业 LLM 智能路由网关*
