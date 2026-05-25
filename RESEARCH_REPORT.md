# AI 路由系统深度研究报告

> 基于对 15+ 开源项目、学术论文、商业产品的系统性调研

---

## 一、核心发现：路由决策的底层逻辑

### 1.1 五种路由范式

| 范式 | 代表 | 原理 | 延迟 | 适用场景 |
|------|------|------|------|----------|
| **规则路由** | LiteLLM cost-based | 预定义策略（价格/延迟/负载） | <1ms | 已知模型能力 |
| **向量路由** | Semantic Router | 输入编码为向量，与预定义路由向量比较 | <5ms | 意图分类 |
| **统计路由** | RouteLLM | 矩阵分解预测强模型胜率 | <5ms | 强弱模型二选一 |
| **神经路由** | Unify.ai / Martian | 训练神经网络预测模型表现 | <10ms | 多模型动态选择 |
| **信号路由** | vLLM Semantic Router | 多维信号提取→综合决策 | <20ms | 企业级全链路 |

**关键洞察：** TaskRouter 目前是"规则路由"（关键词+权重），最大的升级方向是引入"统计路由"（RouteLLM 的矩阵分解方法）——用历史偏好数据训练一个轻量分类器，预测"本地模型能否处理好这个任务"。

### 1.2 RouteLLM 的核心算法

RouteLLM 只解决一个问题：**给定 prompt，强模型的胜率是多少？**

```
如果 胜率 > 阈值 → 路由到强模型（云端）
否则 → 路由到弱模型（本地）
```

四种路由器实现：
- **mf（矩阵分解，推荐）**：将 prompt 和模型嵌入共享潜在空间，预测偏好概率。在 GPT-4 vs Mixtral 的人类偏好数据上训练，但能泛化到其他模型对
- **sw_ranking**：加权 Elo 排名
- **bert**：BERT 二分类器
- **causal_llm**：LLM 分类器

**阈值校准**：`python -m routellm.calibrate_threshold --strong-model-pct 0.5` → 输出阈值 0.11593，控制"多少比例走强模型"

**效果**：成本降低 85%，保持 95% GPT-4 性能

**对 TaskRouter 的启示：** 我们可以用类似方法训练一个"本地能否处理"的分类器，替代当前的关键词权重评分。

### 1.3 Martian 的预测性路由

Martian 的突破：**不运行模型就能预测模型表现**。

通过研究模型内部运作机制（looking inside models），预测模型对特定 prompt 的表现。这意味着：
- 不需要先跑所有模型再选最好的（太贵）
- 新模型上线后零摩擦自动纳入路由体系
- 同一任务，不同模型使用不同 prompt（自动 prompt 优化）

**效果**：金融行业 50 步工作流，成功率从 5.98% 提升到 35.99%（6 倍），成本降低 7 倍

---

## 二、成本优化的五大杠杆

### 杠杆 1：模型路由（40-70% 节省）

**业界最佳实践——任务分级：**
- 规划与分解 → 前沿模型（Claude Opus）
- 网络搜索与摘要 → 中端模型（Claude Sonnet）
- 结构化输出提取 → 小型快速模型
- 质量审查 → 再次使用前沿模型

**量化数据**：路由 70% 到 Haiku、20% 到 Sonnet、10% 到 Opus，成本从 $5.00/M 降至 $1.60/M

### 杠杆 2：上下文压缩（50-70% Token 缩减）

两大流派：
- **逐字删除**（Morph Compact）：删除低信号 token，保留每个字符原样，33,000 tok/s，零幻觉
- **语义压缩**（LLMLingua）：用小模型评估 token 重要性，最高 20x 压缩比

**Chain of Draft（CoD）**：匹配 Chain of Thought 准确率，仅使用 7.6% 的推理 token

### 杠杆 3：Prompt Caching（缓存命中时节省 90%）

| 提供商 | 读取折扣 | 写入溢价 | 最小前缀 |
|--------|---------|---------|---------|
| Anthropic | 90% | 1.25x | 1,024-4,096 token |
| OpenAI | 50-90% | 无 | 1,024 token |
| Google | 90% | 存储费 | 因模型而异 |

**架构原则**：稳定内容（系统指令、工具定义）在前，可变内容（用户输入）在后

**量化案例**：6 个产品从 $612/月降至 $227/月（63%），Agent 循环类从 $168 降至 $32（81%）

### 杠杆 4：语义缓存（20-60% 节省）

**GPTCache 架构**：
```
Adapter → Pre-Processor → Embedding Generator → Cache Manager → Similarity Evaluator
```

核心数据：31% 的企业 LLM 查询与历史查询语义相似。命中时延迟从 500ms+ 降至约 50ms

### 杠杆 5：批处理 API（50% 折扣）

32 个请求批处理可降低 85% 单 token 成本，延迟仅增加 20%

**叠加效果**：各层独立生效，组合可实现 80%+ 的总成本降低

---

## 三、蒸馏和学习闭环

### 3.1 数据蒸馏（最实用的方法）

**关键发现**：弱模型生成的数据反而比强模型更优（固定计算预算下）

- GOLD 框架：LLM 生成数据 → 训练 SLM → OOD 检测失败模式 → 反馈给 LLM 生成更有针对性的数据。10 个 NLP 任务平均超越 baseline 5%
- SKD（ICLR 2025）：借鉴推测解码，student 提出 token，teacher 替换低质量 token。翻译提升 41.8%，摘要提升 230%

### 3.2 DPO（小团队首选）

DPO vs RLHF：
- 模型数量：1 个 vs 3 个
- 计算量：约 1/3
- 训练稳定性：高 vs PPO 不稳定
- 数据量：至少 1000 对（领域特定），10000+ 对（鲁棒对齐）

**实践建议**：先 SFT 再 DPO 两阶段流程，β 参数 0.1-0.5

### 3.3 自动提示优化

**DSPy MIPROv2**（最成熟）：
- 同时优化指令和 few-shot 示例
- 三阶段：引导→接地→离散搜索
- 在 qwen2.5:0.5b 数学题上从 33.3% 提升到 55.6%

**OPRO**（Google）：
- 用 LLM 作为优化器，从历史 prompt 及其准确率中学习
- 优化的 zero-shot 指令匹配 few-shot CoT 性能

---

## 四、企业级架构模式

### 4.1 信号-决策架构（vLLM Semantic Router）

```
请求 → 信号层（任务类型/复杂度/安全风险/PII/幻觉）→ 决策层 → 模型池
```

用微调的 BERT 分类器做请求分类，速度极快。支持 LoRA 扩展动态加载领域特定路由能力。

### 4.2 分层防御模式

```
语义缓存 → 模型路由 → 前缀缓存 → 批处理
```

每层独立生效，组合叠加。

### 4.3 预算熔断模式

LiteLLM 的实现：
- 45 秒总超时
- 60 秒冷却期（失败模型）
- 429 错误即时冷却 + 自动跳过
- 优先级排序（order=1 → order=2 → fallback）

### 4.4 企业网关架构（Helicone）

```
Frontend → Worker（Edge 代理，Rust，1-5ms）→ Jawn（API 服务）→ ClickHouse（分析）
```

安全层：
- 基础层：Prompt Guard (86M) — 检测注入攻击，8 种语言
- 高级层：Llama Guard (3.8B) — 14 类威胁检测，97%+ 检测率

---

## 五、对 TaskRouter 的启示

### 5.1 当前差距

| 维度 | TaskRouter 现状 | 业界最佳 | 差距 |
|------|----------------|---------|------|
| 路由算法 | 关键词+权重 | 矩阵分解/神经网络 | 2 代 |
| 缓存 | Trigram Jaccard | 向量语义缓存 | 1 代 |
| 成本优化 | 单层路由 | 5 层叠加 | 4 层 |
| 蒸馏 | 基础 few-shot 注入 | DPO + 数据蒸馏 | 2 代 |
| 可观测性 | 基础日志 | 全链路追踪 | 1 代 |
| 安全 | PII 脱敏 | PII+注入+幻觉检测 | 2 项 |

### 5.2 可借鉴的具体方案

**立即可做（1-2 周）：**
1. 引入 RouteLLM 的矩阵分解方法替代关键词权重评分
2. 实现 LLMLingua 风格的上下文压缩
3. 添加 Prompt Caching 支持（稳定前缀模式）
4. 实现 Helicone 风格的安全检测层

**中期目标（1-2 月）：**
1. 训练"本地能否处理"的分类器（RouteLLM mf 方法）
2. 实现 DSPy MIPROv2 自动提示优化
3. 构建 GPTCache 风格的向量语义缓存
4. 实现 DPO 反馈循环

**长期愿景：**
1. Martian 风格的预测性路由
2. Unify 风格的可训练定制路由器
3. 全链路可观测性（类 LangSmith）

### 5.3 核心设计原则

1. **解耦**：模型选择与业务逻辑分离，通过配置管理路由
2. **分层**：缓存→路由→压缩→批处理，每层独立生效
3. **可训练**：路由器本身应该是可学习的，而非硬编码规则
4. **可度量**：无法度量就无法优化——成本归因到团队/用例/数据域
5. **渐进升级**：小模型先试，低置信度升级，而非一刀切

---

## 参考资源

| 资源 | 链接 | 关键点 |
|------|------|--------|
| LiteLLM | github.com/BerriAI/litellm | 47.9K stars, 最全面的 LLM 代理 |
| vLLM Semantic Router | github.com/vllm-project/semantic-router | 信号-决策架构 |
| RouteLLM | github.com/lm-sys/routellm | 矩阵分解路由，ICLR 2025 |
| Semantic Router | github.com/aurelio-labs/semantic-router | 向量语义分类 |
| DSPy | github.com/stanfordnlp/dspy | 自动提示优化 |
| FrugalGPT | github.com/stanford-futuredata/FrugalGPT | 级联推理 |
| GPTCache | github.com/zilliztech/GPTCache | 语义缓存 |
| Helicone | github.com/Helicone/helicone | 开源 AI 网关 |
| LLMLingua | 微软研究院, EMNLP 2023 | 上下文压缩 |
| MiniLLM | 微软, ICLR 2024 | 反向 KL 蒸馏 |
| SKD | ICLR 2025 | 推测蒸馏 |
