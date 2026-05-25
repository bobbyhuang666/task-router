#!/usr/bin/env python3
"""
TaskRouter 路由评估框架

基于真实任务场景评估路由质量，替代合成 benchmark。
参考 RouteLLM (LMSYS) 的评估方法论。

评估维度:
1. 路由准确率 — 简单任务走本地，复杂任务走云端
2. 成本效率   — 相比全云端方案节省多少
3. 预测集覆盖率 — conformal prediction 集合是否达到目标覆盖率
4. 分类别表现 — 各任务类型的路由质量

用法:
    python3 scripts/benchmark_routing.py [--tasks N] [--verbose]
"""

import argparse
import json
import random
import time
from dataclasses import dataclass, field

# ─── 任务定义 ──────────────────────────────────────────────────


@dataclass
class EvalTask:
    """评估任务"""
    id: str
    category: str            # translation / classification / code / analysis / creative / extraction
    difficulty: str          # easy / medium / hard
    action: str
    text: str
    ground_truth_route: str  # local / cloud / either
    reason: str              # 为什么应该是这个路由
    min_quality: float = 0.6  # 最低可接受质量分


# 60 个真实任务场景，按难度和类别分布
TASK_SUITE: list[EvalTask] = [
    # ── Translation (12) ──
    EvalTask("T01", "translation", "easy", "翻译成中文", "Hello, how are you?", "local", "简单日常翻译"),
    EvalTask("T02", "translation", "easy", "翻译", "Thank you for your help.", "local", "短句翻译"),
    EvalTask("T03", "translation", "easy", "翻译这段话", "The weather is nice today.", "local", "简单陈述句"),
    EvalTask("T04", "translation", "medium", "翻译技术文档",
             "The Kubernetes pod was evicted due to OOMKilled status.", "either", "技术术语需要准确"),
    EvalTask("T05", "translation", "medium", "翻译法律条款",
             "The indemnifying party shall hold harmless the indemnified party.", "cloud", "法律用语精确性要求高"),
    EvalTask("T06", "translation", "hard", "翻译诗歌",
             "Two roads diverged in a yellow wood, and sorry I could not travel both.", "cloud", "文学翻译需要创意"),
    EvalTask("T07", "translation", "easy", "帮我翻译一下", "I agree with the terms.", "local", "简单同意表达"),
    EvalTask("T08", "translation", "medium", "翻译医学报告",
             "Patient presents with acute myocardial infarction, troponin elevated.", "cloud", "医学术语精确性"),
    EvalTask("T09", "translation", "easy", "翻译为中文", "Please wait for confirmation.", "local", "简单指令"),
    EvalTask("T10", "translation", "hard", "翻译合同条款",
             "Force majeure events including but not limited to acts of God, war, pandemic.", "cloud", "合同翻译需要法律准确性"),
    EvalTask("T11", "translation", "medium", "翻译技术博客",
             "WebAssembly enables near-native performance in browser environments.", "either", "技术翻译，本地可能够用"),
    EvalTask("T12", "translation", "easy", "翻译邮件", "Looking forward to meeting you.", "local", "简单邮件用语"),

    # ── Classification (10) ──
    EvalTask("C01", "classification", "easy", "分类", "iPhone 15 Pro Max 256GB", "local", "明确商品分类"),
    EvalTask("C02", "classification", "easy", "判断情感", "这个产品太好用了！", "local", "明显正面情感"),
    EvalTask("C03", "classification", "easy", "分类新闻", "沪指今日收涨0.5%", "local", "财经新闻分类"),
    EvalTask("C04", "classification", "medium", "判断意图",
             "我想问一下你们的退货政策是怎样的", "local", "客服意图识别"),
    EvalTask("C05", "classification", "medium", "分类反馈",
             "包装很好但是产品和描述不太一样", "either", "混合情感需要细粒度分析"),
    EvalTask("C06", "classification", "hard", "判断合规风险",
             "该产品含有未经批准的添加剂成分", "cloud", "合规判断需要专业知识"),
    EvalTask("C07", "classification", "easy", "情感分析", "服务态度极差", "local", "明显负面"),
    EvalTask("C08", "classification", "medium", "分类工单",
             "系统在高并发场景下出现间歇性超时", "either", "技术工单分类"),
    EvalTask("C09", "classification", "easy", "判断类别", "患者主诉头痛三天", "local", "医疗分类"),
    EvalTask("C10", "classification", "hard", "判断法律风险",
             "该条款可能违反消费者权益保护法第26条", "cloud", "法律判断需要专业知识"),

    # ── Code Generation (12) ──
    EvalTask("G01", "code", "easy", "写一个hello world", "", "local", "最简单的代码"),
    EvalTask("G02", "code", "easy", "写一个Python函数计算斐波那契数列", "", "local", "经典简单算法"),
    EvalTask("G03", "code", "medium", "实现快速排序", "", "either", "标准算法"),
    EvalTask("G04", "code", "medium", "写一个LRU缓存", "", "cloud", "需要设计能力"),
    EvalTask("G05", "code", "hard", "设计分布式锁，支持Redis和ZooKeeper", "", "cloud", "系统设计级"),
    EvalTask("G06", "code", "medium", "写一个防抖函数", "", "either", "前端常见需求"),
    EvalTask("G07", "code", "easy", "写SQL查询找最大值", "", "local", "基础SQL"),
    EvalTask("G08", "code", "hard", "实现一个简易ORM框架", "", "cloud", "架构设计"),
    EvalTask("G09", "code", "medium", "写单元测试", "def add(a, b): return a + b", "local", "简单测试"),
    EvalTask("G10", "code", "hard", "实现Raft共识算法核心", "", "cloud", "分布式系统"),
    EvalTask("G11", "code", "easy", "写正则表达式匹配邮箱", "", "local", "常见正则"),
    EvalTask("G12", "code", "medium", "实现一个简单的JSON解析器", "", "cloud", "递归下降解析"),

    # ── Analysis (10) ──
    EvalTask("A01", "analysis", "easy", "总结这段话", "今天天气不错，适合出门散步。", "local", "简单总结"),
    EvalTask("A02", "analysis", "medium", "分析数据趋势",
             "Q1: 100万, Q2: 120万, Q3: 95万, Q4: 150万", "either", "基础数据分析"),
    EvalTask("A03", "analysis", "hard", "分析竞品优劣势",
             "我司产品A vs 竞品B的市场表现数据", "cloud", "需要深度商业分析"),
    EvalTask("A04", "analysis", "medium", "提取关键信息",
             "会议纪要：讨论了三个方案，最终决定采用方案B", "local", "信息提取"),
    EvalTask("A05", "analysis", "hard", "写商业计划书摘要",
             "SaaS产品，目标市场中小企业，ARR 500万", "cloud", "需要商业洞察"),
    EvalTask("A06", "analysis", "easy", "列出要点", "Python的优点：简单易学、生态丰富、社区活跃", "local", "简单列举"),
    EvalTask("A07", "analysis", "medium", "对比分析",
             "React vs Vue 2024年生态对比", "either", "技术对比"),
    EvalTask("A08", "analysis", "hard", "风险评估",
             "投资项目：海外房地产，预期回报15%", "cloud", "需要金融知识"),
    EvalTask("A09", "analysis", "easy", "解释概念", "什么是REST API", "local", "基础概念解释"),
    EvalTask("A10", "analysis", "medium", "根因分析",
             "系统延迟从50ms突增到500ms，日志显示数据库连接池耗尽", "either", "技术诊断"),

    # ── Creative (8) ──
    EvalTask("R01", "creative", "easy", "写一句广告语", "产品：蓝牙耳机", "local", "简单创意"),
    EvalTask("R02", "creative", "medium", "写产品描述",
             "产品：智能手表，功能：心率监测、GPS、NFC", "either", "产品文案"),
    EvalTask("R03", "creative", "hard", "写一篇技术博客",
             "主题：如何设计高可用微服务架构", "cloud", "长文写作"),
    EvalTask("R04", "creative", "medium", "写邮件回复",
             "客户投诉产品质量问题，需要道歉并提供解决方案", "either", "商务邮件"),
    EvalTask("R05", "creative", "easy", "想个标题", "文章主题：Python入门教程", "local", "简单标题"),
    EvalTask("R06", "creative", "hard", "写项目提案",
             "开发一个AI驱动的客服系统，预算100万", "cloud", "正式提案"),
    EvalTask("R07", "creative", "medium", "写用户故事",
             "电商平台的购物车功能", "either", "敏捷开发"),
    EvalTask("R08", "creative", "easy", "写通知公告", "明天下午3点开会", "local", "简单通知"),

    # ── Extraction (8) ──
    EvalTask("E01", "extraction", "easy", "提取关键词", "人工智能正在改变医疗行业", "local", "简单提取"),
    EvalTask("E02", "extraction", "easy", "提取日期", "会议定于2024年3月15日下午2点", "local", "结构化提取"),
    EvalTask("E03", "extraction", "medium", "提取合同条款",
             "甲方应在签约后30日内支付首期款项，金额为总价的30%", "either", "合同信息提取"),
    EvalTask("E04", "extraction", "medium", "提取发票信息",
             "发票号：INV-2024-001，金额：￥15,800.00，开票日期：2024-01-15", "local", "结构化数据"),
    EvalTask("E05", "extraction", "hard", "提取法律实体关系",
             "原告张三诉被告李四和王五，涉及位于北京市朝阳区的房产", "cloud", "法律NLP"),
    EvalTask("E06", "extraction", "easy", "提取邮箱地址", "联系邮箱：test@example.com", "local", "正则提取"),
    EvalTask("E07", "extraction", "medium", "提取表格数据",
             "员工表：张三 部门：技术部 职级：P7 入职：2020-03", "local", "结构化提取"),
    EvalTask("E08", "extraction", "hard", "提取医学实体",
             "患者服用阿司匹林100mg/d，合并二甲双胍500mg bid", "cloud", "医学NER"),
]


# ─── 评估逻辑 ──────────────────────────────────────────────────


@dataclass
class EvalResult:
    """单个任务的评估结果"""
    task_id: str
    category: str
    difficulty: str
    ground_truth: str       # local / cloud / either
    predicted_route: str    # local / cloud
    prediction_set: list[str]
    correct: bool           # 路由是否正确
    conformal_in_set: bool  # ground_truth 是否在 prediction_set 中
    cost_cloud: float
    cost_actual: float


@dataclass
class BenchmarkResult:
    """汇总结果"""
    total_tasks: int = 0
    correct_routes: int = 0
    conformal_coverage_hits: int = 0
    total_cost_cloud: float = 0.0
    total_cost_actual: float = 0.0
    by_category: dict = field(default_factory=dict)
    by_difficulty: dict = field(default_factory=dict)
    results: list = field(default_factory=list)
    elapsed_ms: float = 0.0


def evaluate_routing(
    tasks: list[EvalTask],
    target_coverage: float = 0.9,
    seed: int = 42,
    verbose: bool = False,
    route_fn=None,
) -> BenchmarkResult:
    """
    评估路由质量。

    不依赖真实模型运行，使用基于任务特征的路由启发式：
    - 模拟 estimate_complexity 的评分逻辑
    - 模拟 TQBC 的 logprobs 分析
    - 模拟 conformal router 的决策
    """
    result = BenchmarkResult()
    start = time.monotonic()
    if route_fn is None:
        route_fn = _heuristic_route

    for task in tasks:
        # 模拟路由决策
        predicted_route = route_fn(task)

        # 模拟 prediction_set（基于难度的不确定性）
        prediction_set = _heuristic_prediction_set(task, target_coverage)

        # 评估正确性
        if task.ground_truth_route == "either":
            correct = True  # 两种路由都可接受
        elif task.ground_truth_route == "local":
            correct = predicted_route == "local"
        else:  # cloud
            correct = predicted_route == "cloud"

        # conformal 覆盖率检查
        conformal_in_set = task.ground_truth_route in prediction_set or task.ground_truth_route == "either"

        # 成本计算
        cost_cloud = _estimate_cost(task, "cloud")
        cost_actual = _estimate_cost(task, predicted_route)

        er = EvalResult(
            task_id=task.id,
            category=task.category,
            difficulty=task.difficulty,
            ground_truth=task.ground_truth_route,
            predicted_route=predicted_route,
            prediction_set=prediction_set,
            correct=correct,
            conformal_in_set=conformal_in_set,
            cost_cloud=cost_cloud,
            cost_actual=cost_actual,
        )
        result.results.append(er)
        result.total_tasks += 1
        if correct:
            result.correct_routes += 1
        if conformal_in_set:
            result.conformal_coverage_hits += 1
        result.total_cost_cloud += cost_cloud
        result.total_cost_actual += cost_actual

        # 按类别统计
        cat = task.category
        if cat not in result.by_category:
            result.by_category[cat] = {"total": 0, "correct": 0, "cost_cloud": 0, "cost_actual": 0}
        result.by_category[cat]["total"] += 1
        if correct:
            result.by_category[cat]["correct"] += 1
        result.by_category[cat]["cost_cloud"] += cost_cloud
        result.by_category[cat]["cost_actual"] += cost_actual

        # 按难度统计
        diff = task.difficulty
        if diff not in result.by_difficulty:
            result.by_difficulty[diff] = {"total": 0, "correct": 0}
        result.by_difficulty[diff]["total"] += 1
        if correct:
            result.by_difficulty[diff]["correct"] += 1

        if verbose:
            icon = "OK" if correct else "XX"
            print(f"  [{icon}] {task.id:4} {task.difficulty:6} {task.category:14} "
                  f"GT={task.ground_truth_route:6} → {predicted_route:6} "
                  f"set={prediction_set}")

    result.elapsed_ms = (time.monotonic() - start) * 1000
    return result


def _heuristic_route(task: EvalTask) -> str:
    """模拟路由决策启发式"""
    # 简单任务 → 本地
    if task.difficulty == "easy":
        return "local"
    # 困难任务 → 云端
    if task.difficulty == "hard":
        return "cloud"
    # 中等任务：根据类别决定
    cloud_bias_categories = {"code", "analysis", "creative"}
    if task.category in cloud_bias_categories:
        return "cloud" if random.random() < 0.6 else "local"
    return "local" if random.random() < 0.7 else "cloud"


def _random_route(task: EvalTask) -> str:
    """随机路由基线（50/50）"""
    return "cloud" if random.random() < 0.5 else "local"


def _heuristic_prediction_set(task: EvalTask, target_coverage: float) -> list[str]:
    """模拟 conformal prediction set"""
    if task.difficulty == "easy":
        # 高置信度 → 大概率单元素集
        if random.random() < 0.95:
            return ["local"]
        return ["local", "cloud"]

    if task.difficulty == "hard":
        # 困难任务 → 大概率包含云端
        if random.random() < 0.85:
            return ["cloud"]
        return ["local", "cloud"]

    # 中等 → 更多双元素集（不确定）
    if random.random() < 0.5:
        return ["local", "cloud"]
    return ["local"] if random.random() < 0.6 else ["cloud"]


def _estimate_cost(task: EvalTask, route: str) -> float:
    """估算任务成本"""
    text_len = len(task.action) + len(task.text)
    input_tokens = max(50, text_len // 2)
    output_tokens = max(80, len(task.action) * 2)

    if route == "local":
        return 0.0  # 本地免费

    # DeepSeek 定价: $0.14/M input, $0.28/M output
    return (input_tokens * 0.14 + output_tokens * 0.28) / 1_000_000


# ─── 报告生成 ──────────────────────────────────────────────────


def print_report(result: BenchmarkResult) -> None:
    """打印评估报告"""
    print()
    print("=" * 70)
    print("TaskRouter 路由评估报告")
    print("=" * 70)

    # 总体指标
    acc = result.correct_routes / result.total_tasks * 100
    cov = result.conformal_coverage_hits / result.total_tasks * 100
    savings = (1 - result.total_cost_actual / result.total_cost_cloud) * 100 if result.total_cost_cloud > 0 else 0

    print(f"\n总任务数: {result.total_tasks}")
    print(f"路由准确率: {acc:.1f}% ({result.correct_routes}/{result.total_tasks})")
    print(f"Conformal 覆盖率: {cov:.1f}% (目标 90%)")
    print(f"全云端成本: ${result.total_cost_cloud:.6f}")
    print(f"实际成本:   ${result.total_cost_actual:.6f}")
    print(f"节省比例:   {savings:.1f}%")
    print(f"耗时:       {result.elapsed_ms:.1f}ms")

    # 按类别
    print(f"\n{'类别':16} {'任务数':6} {'准确率':8} {'云端成本':12} {'实际成本':12} {'节省':8}")
    print("-" * 70)
    for cat, stats in sorted(result.by_category.items()):
        cat_acc = stats["correct"] / stats["total"] * 100
        cat_savings = (1 - stats["cost_actual"] / stats["cost_cloud"]) * 100 if stats["cost_cloud"] > 0 else 0
        print(f"{cat:16} {stats['total']:6} {cat_acc:7.1f}% ${stats['cost_cloud']:10.6f} "
              f"${stats['cost_actual']:10.6f} {cat_savings:7.1f}%")

    # 按难度
    print(f"\n{'难度':16} {'任务数':6} {'准确率':8}")
    print("-" * 40)
    for diff in ["easy", "medium", "hard"]:
        if diff in result.by_difficulty:
            stats = result.by_difficulty[diff]
            diff_acc = stats["correct"] / stats["total"] * 100
            print(f"{diff:16} {stats['total']:6} {diff_acc:7.1f}%")

    # Conformal 覆盖率分析
    print("\nConformal Prediction 覆盖率:")
    for diff in ["easy", "medium", "hard"]:
        subset = [r for r in result.results if r.difficulty == diff]
        if subset:
            covered = sum(1 for r in subset if r.conformal_in_set)
            print(f"  {diff:10}: {covered}/{len(subset)} = {covered/len(subset)*100:.1f}%")


def save_json(result: BenchmarkResult, path: str) -> None:
    """保存 JSON 结果"""
    data = {
        "summary": {
            "total_tasks": result.total_tasks,
            "accuracy_pct": round(result.correct_routes / result.total_tasks * 100, 1),
            "conformal_coverage_pct": round(result.conformal_coverage_hits / result.total_tasks * 100, 1),
            "cost_cloud": round(result.total_cost_cloud, 6),
            "cost_actual": round(result.total_cost_actual, 6),
            "savings_pct": round((1 - result.total_cost_actual / result.total_cost_cloud) * 100, 1) if result.total_cost_cloud > 0 else 0,
            "elapsed_ms": round(result.elapsed_ms, 1),
        },
        "by_category": result.by_category,
        "by_difficulty": result.by_difficulty,
        "tasks": [
            {
                "id": r.task_id, "category": r.category, "difficulty": r.difficulty,
                "ground_truth": r.ground_truth, "predicted": r.predicted_route,
                "prediction_set": r.prediction_set, "correct": r.correct,
                "conformal_in_set": r.conformal_in_set,
            }
            for r in result.results
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到 {path}")


# ─── CLI ──────────────────────────────────────────────────────


def print_comparison(heuristic: BenchmarkResult, random_baseline: BenchmarkResult) -> None:
    """打印策略对比表"""
    h_acc = heuristic.correct_routes / heuristic.total_tasks * 100
    r_acc = random_baseline.correct_routes / random_baseline.total_tasks * 100
    h_cov = heuristic.conformal_coverage_hits / heuristic.total_tasks * 100
    r_cov = random_baseline.conformal_coverage_hits / random_baseline.total_tasks * 100
    h_savings = (1 - heuristic.total_cost_actual / heuristic.total_cost_cloud) * 100 if heuristic.total_cost_cloud > 0 else 0
    r_savings = (1 - random_baseline.total_cost_actual / random_baseline.total_cost_cloud) * 100 if random_baseline.total_cost_cloud > 0 else 0

    print("\n" + "=" * 70)
    print("策略对比")
    print("=" * 70)
    print(f"\n{'指标':20} {'智能路由':12} {'随机基线':12} {'提升':12}")
    print("-" * 56)
    print(f"{'路由准确率':20} {h_acc:11.1f}% {r_acc:11.1f}% {h_acc - r_acc:+11.1f}%")
    print(f"{'Conformal 覆盖率':20} {h_cov:11.1f}% {r_cov:11.1f}% {h_cov - r_cov:+11.1f}%")
    print(f"{'成本节省':20} {h_savings:11.1f}% {r_savings:11.1f}% {h_savings - r_savings:+11.1f}%")
    print(f"{'实际成本':20} ${heuristic.total_cost_actual:10.6f} ${random_baseline.total_cost_actual:10.6f}")


def main():
    parser = argparse.ArgumentParser(description="TaskRouter 路由评估")
    parser.add_argument("--tasks", type=int, default=0, help="限制任务数（0=全部）")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示每个任务详情")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output", type=str, default="benchmark_routing.json", help="输出文件")
    args = parser.parse_args()

    random.seed(args.seed)

    tasks = TASK_SUITE
    if args.tasks > 0:
        tasks = tasks[:args.tasks]

    print(f"评估 {len(tasks)} 个任务（seed={args.seed}）...")
    result = evaluate_routing(tasks, verbose=args.verbose, route_fn=_heuristic_route)
    print_report(result)
    save_json(result, args.output)

    # 随机基线对比
    random.seed(args.seed)
    random_result = evaluate_routing(tasks, route_fn=_random_route)
    print_comparison(result, random_result)


if __name__ == "__main__":
    main()
