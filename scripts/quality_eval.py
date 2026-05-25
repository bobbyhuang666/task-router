"""
质量评估框架 — A/B 测试和回归检测

核心功能:
- 对比不同模型/路由策略的输出质量
- 自动检测质量回归
- 生成质量报告
- 支持人工标注和自动评估
"""

import os
import json
import time
from dataclasses import dataclass, asdict, field

# ─── 评估用例 ──────────────────────────────────────────────────

@dataclass
class EvalCase:
    """单个评估用例"""
    id: str
    action: str
    text: str
    expected: str              # 期望输出（或关键词）
    task_type: str = ""        # 任务类型
    difficulty: str = "easy"   # easy / medium / hard
    tags: list = field(default_factory=list)

@dataclass
class EvalResult:
    """单次评估结果"""
    case_id: str
    model: str
    output: str
    score: float               # 0-1
    match_type: str            # exact / contains / semantic / fail
    latency_ms: int = 0
    tokens_input: int = 0
    tokens_output: int = 0
    timestamp: str = ""


# ─── 评估数据集 ──────────────────────────────────────────────────

DEFAULT_EVAL_SET = [
    # 翻译（期望包含关键词即可）
    EvalCase("trans_01", "翻译成中文", "Hello world", "你好", "translation"),
    EvalCase("trans_02", "翻译成中文", "Good morning", "早上好", "translation"),
    EvalCase("trans_03", "翻译成中文", "Thank you", "谢谢", "translation"),
    EvalCase("trans_04", "翻译成英文", "你好世界", "hello", "translation"),
    EvalCase("trans_05", "翻译成中文", "Machine learning", "机器学习", "translation", "medium"),

    # 情感分析（精确匹配）
    EvalCase("senti_01", "判断情感", "产品质量很好", "正面", "sentiment"),
    EvalCase("senti_02", "判断情感", "服务太差了", "负面", "sentiment"),
    EvalCase("senti_03", "判断情感", "发货速度快", "正面", "sentiment"),
    EvalCase("senti_04", "判断情感", "价格太贵了", "负面", "sentiment"),
    EvalCase("senti_05", "判断情感", "性价比很高", "正面", "sentiment"),

    # 提取（包含关键词即可）
    EvalCase("extr_01", "提取人名", "张三在北京工作", "张三", "extraction"),
    EvalCase("extr_02", "提取关键词", "苹果发布新款iPhone", "苹果", "extraction"),
    EvalCase("extr_03", "提取日期", "2024年1月15日发布", "2024", "extraction"),
    EvalCase("extr_04", "提取邮箱", "联系test@example.com", "test@example.com", "extraction", "medium"),

    # 分类（输出应包含项目名即可，模型可能列出项目而非类别）
    EvalCase("cls_01", "分类", "报告.pdf, 照片.jpg, 代码.py", "报告.pdf", "classification"),
    EvalCase("cls_02", "分类", "苹果, 香蕉, 白菜, 萝卜", "苹果", "classification"),
    EvalCase("cls_03", "分类", "北京, 上海, 纽约, 伦敦", "北京", "classification"),

    # 格式化（包含关键内容即可）
    EvalCase("fmt_01", "格式化", "apple,banana,orange", "apple", "formatting"),
    EvalCase("fmt_02", "去重", "apple, banana, apple, orange", "apple", "formatting"),

    # 概括（包含关键词即可）
    EvalCase("sum_01", "概括", "公司第一季度营收5000万元，同比增长20%", "营收", "summarization", "medium"),
]


# ─── 质量评估器 ──────────────────────────────────────────────────

class QualityEvaluator:
    """
    质量评估器 — 对比模型输出与期望结果。

    使用方法:
        evaluator = QualityEvaluator()
        results = evaluator.run_eval(model="qwen-tool:latest")
        report = evaluator.generate_report(results)
    """

    def __init__(self, eval_set: list[EvalCase] = None, cache_dir: str = None):
        self.eval_set = eval_set or DEFAULT_EVAL_SET
        if cache_dir is None:
            from config import get_config
            cache_dir = get_config().cache_dir
        self.cache_dir = cache_dir
        self.results_file = os.path.join(self.cache_dir, "eval_results.jsonl")
        os.makedirs(self.cache_dir, exist_ok=True)

    def score_output(self, output: str, expected: str) -> tuple[float, str]:
        """
        评估输出质量。

        返回: (score 0-1, match_type)
        """
        output_lower = output.lower().strip()
        expected_lower = expected.lower().strip()

        # 精确匹配
        if output_lower == expected_lower:
            return 1.0, "exact"

        # 包含匹配
        if expected_lower in output_lower:
            return 0.9, "contains"

        # 反向包含（输出被期望包含）
        if output_lower in expected_lower:
            return 0.7, "reverse_contains"

        # 部分匹配（期望的每个字符都在输出中）
        if len(expected) >= 2:
            char_matches = sum(1 for c in expected if c in output)
            char_score = char_matches / len(expected)
            if char_score > 0.5:
                return char_score * 0.6, "partial"

        return 0.0, "fail"

    def run_eval(self, model: str = None, cases: list[EvalCase] = None,
                 verbose: bool = False) -> list[EvalResult]:
        """
        运行评估。

        参数:
            model: 指定模型名称
            cases: 指定评估用例
            verbose: 是否打印详细信息

        返回:
            评估结果列表
        """
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from task_router import call_ollama, build_optimized_prompt, preprocess_text
        from routing import detect_task_type
        from prompts import PROMPT_TEMPLATES

        eval_cases = cases or self.eval_set
        results = []

        for case in eval_cases:
            task_type = case.task_type or detect_task_type(case.action, PROMPT_TEMPLATES)
            clean_text = preprocess_text(case.text)
            clean_action = preprocess_text(case.action, max_chars=200)
            prompt = build_optimized_prompt(task_type, clean_action, clean_text, [])

            start = time.time()
            result = None
            try:
                result = call_ollama(prompt, model=model, max_tokens=64)
                output = result["text"].strip()
                latency = int((time.time() - start) * 1000)
            except Exception as e:
                output = f"[错误] {e}"
                latency = int((time.time() - start) * 1000)

            score, match_type = self.score_output(output, case.expected)

            eval_result = EvalResult(
                case_id=case.id,
                model=model or "default",
                output=output,
                score=score,
                match_type=match_type,
                latency_ms=latency,
                tokens_input=result.get("tokens_input", 0) if result else 0,
                tokens_output=result.get("tokens_output", 0) if result else 0,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            )
            results.append(eval_result)

            if verbose:
                status = "✓" if score >= 0.7 else "✗"
                print(f"  {status} {case.id}: {case.action} → {output[:30]} "
                      f"(期望:{case.expected}, 分:{score:.1f}, {match_type})")

        return results

    def run_ab_test(self, model_a: str, model_b: str,
                    cases: list[EvalCase] = None, verbose: bool = False) -> dict:
        """
        A/B 测试：对比两个模型。

        返回:
            {
                "model_a": {"model": ..., "avg_score": ..., "win_count": ...},
                "model_b": {"model": ..., "avg_score": ..., "win_count": ...},
                "ties": ...,
                "details": [...]
            }
        """
        if verbose:
            print(f"A/B 测试: {model_a} vs {model_b}")
            print("=" * 50)

        results_a = self.run_eval(model_a, cases, verbose)
        results_b = self.run_eval(model_b, cases, verbose)

        wins_a = wins_b = ties = 0
        details = []

        for ra, rb in zip(results_a, results_b):
            if ra.score > rb.score:
                wins_a += 1
                winner = model_a
            elif rb.score > ra.score:
                wins_b += 1
                winner = model_b
            else:
                ties += 1
                winner = "tie"

            details.append({
                "case_id": ra.case_id,
                "score_a": ra.score,
                "score_b": rb.score,
                "output_a": ra.output[:50],
                "output_b": rb.output[:50],
                "latency_a": ra.latency_ms,
                "latency_b": rb.latency_ms,
                "winner": winner,
            })

        avg_a = sum(r.score for r in results_a) / len(results_a) if results_a else 0
        avg_b = sum(r.score for r in results_b) / len(results_b) if results_b else 0
        latency_a = sum(r.latency_ms for r in results_a) / len(results_a) if results_a else 0
        latency_b = sum(r.latency_ms for r in results_b) / len(results_b) if results_b else 0

        return {
            "model_a": {"model": model_a, "avg_score": round(avg_a, 3),
                        "avg_latency_ms": round(latency_a), "wins": wins_a},
            "model_b": {"model": model_b, "avg_score": round(avg_b, 3),
                        "avg_latency_ms": round(latency_b), "wins": wins_b},
            "ties": ties,
            "total_cases": len(details),
            "details": details,
        }

    def save_results(self, results: list[EvalResult]):
        """保存评估结果到文件"""
        with open(self.results_file, "a") as f:
            for r in results:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    def load_history(self, model: str = None, limit: int = 100) -> list[EvalResult]:
        """加载历史评估结果"""
        if not os.path.exists(self.results_file):
            return []
        results = []
        with open(self.results_file) as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if model and data.get("model") != model:
                        continue
                    results.append(EvalResult(**data))
                except (json.JSONDecodeError, TypeError):
                    continue
        return results[-limit:]

    def detect_regression(self, model: str, window: int = 5) -> dict:
        """
        检测质量回归：比较最近 N 次评估与历史平均。

        返回:
            {
                "regression_detected": bool,
                "current_avg": float,
                "historical_avg": float,
                "delta": float,
                "details": str
            }
        """
        history = self.load_history(model, limit=200)
        if len(history) < window * 2:
            return {"regression_detected": False, "details": "数据不足"}

        recent = history[-window:]
        older = history[:-window]

        recent_avg = sum(r.score for r in recent) / len(recent)
        older_avg = sum(r.score for r in older) / len(older)
        delta = recent_avg - older_avg

        regression = delta < -0.1  # 下降超过 10% 视为回归

        return {
            "regression_detected": regression,
            "current_avg": round(recent_avg, 3),
            "historical_avg": round(older_avg, 3),
            "delta": round(delta, 3),
            "details": f"{'⚠️ 质量回归' if regression else '✓ 质量稳定'}: "
                       f"最近{window}次平均 {recent_avg:.1%} vs 历史 {older_avg:.1%} "
                       f"({'+' if delta >= 0 else ''}{delta:.1%})",
        }

    def generate_report(self, results: list[EvalResult]) -> str:
        """生成评估报告"""
        if not results:
            return "无评估结果"

        total = len(results)
        passed = sum(1 for r in results if r.score >= 0.7)
        avg_score = sum(r.score for r in results) / total
        avg_latency = sum(r.latency_ms for r in results) / total

        # 按匹配类型统计
        match_counts = {}
        for r in results:
            match_counts[r.match_type] = match_counts.get(r.match_type, 0) + 1

        lines = [
            "质量评估报告",
            f"{'='*50}",
            f"模型: {results[0].model}",
            f"用例数: {total}",
            f"通过率: {passed}/{total} ({passed/total:.0%})",
            f"平均分: {avg_score:.2f}",
            f"平均延迟: {avg_latency:.0f}ms",
            "",
            "匹配类型分布:",
        ]
        for match_type, count in sorted(match_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {match_type}: {count} ({count/total:.0%})")

        # 失败用例
        failed = [r for r in results if r.score < 0.7]
        if failed:
            lines.append(f"\n失败用例 ({len(failed)}):")
            for r in failed:
                lines.append(f"  ✗ {r.case_id}: 输出='{r.output[:30]}' (分:{r.score:.1f})")

        return "\n".join(lines)


# ─── CLI 入口 ──────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    evaluator = QualityEvaluator()

    if len(sys.argv) > 1 and sys.argv[1] == "--eval":
        model = sys.argv[2] if len(sys.argv) > 2 else None
        results = evaluator.run_eval(model, verbose=True)
        evaluator.save_results(results)
        print()
        print(evaluator.generate_report(results))
    elif len(sys.argv) > 1 and sys.argv[1] == "--ab":
        model_a = sys.argv[2] if len(sys.argv) > 2 else "qwen-tool:latest"
        model_b = sys.argv[3] if len(sys.argv) > 3 else "qwen-tool-3b:latest"
        report = evaluator.run_ab_test(model_a, model_b, verbose=True)
        print()
        print("A/B 测试结果:")
        print(f"  {model_a}: 平均分 {report['model_a']['avg_score']}, 胜 {report['model_a']['wins']} 次")
        print(f"  {model_b}: 平均分 {report['model_b']['avg_score']}, 胜 {report['model_b']['wins']} 次")
        print(f"  平局: {report['ties']} 次")
    elif len(sys.argv) > 1 and sys.argv[1] == "--regression":
        model = sys.argv[2] if len(sys.argv) > 2 else None
        if not model:
            print("请指定模型: python3 quality_eval.py --regression model_name")
        else:
            result = evaluator.detect_regression(model)
            print(result["details"])
    else:
        print("用法:")
        print("  python3 quality_eval.py --eval [model]     # 运行评估")
        print("  python3 quality_eval.py --ab model_a model_b  # A/B 测试")
        print("  python3 quality_eval.py --regression model  # 回归检测")
