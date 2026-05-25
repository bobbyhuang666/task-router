"""
任务分解与计划执行
"""

from task_router.config import get_config
from task_router.routing import Task, estimate_complexity, detect_task_type
from task_router.prompts import PROMPT_TEMPLATES


# ─── 任务计划模板 ──────────────────────────────────────────────


DECOMPOSE_TEMPLATES: dict[str, dict] = {
    "电商数据分析": {
        "match": ["电商", "销售", "商品", "订单"],
        "subtasks": ["清洗数据（去空值、统一格式）", "按类别分类商品", "统计各品类销售额", "分析销售趋势和异常", "给出优化建议并生成报告"],
        "routes": ["local", "local", "local", "cloud", "cloud"],
    },
    "文件批量整理": {
        "match": ["文件", "桌面", "整理", "归类"],
        "subtasks": ["扫描并列出所有文件", "按扩展名分类", "建议目录结构", "生成整理脚本"],
        "routes": ["local", "local", "local", "cloud"],
    },
}


# ─── 预估和分类 ──────────────────────────────────────────────────


def estimate(task_description: str) -> dict:
    """预估路由和成本"""
    from task_router.cost import calc_cost

    config = get_config()
    dummy = Task(action=task_description)
    decision = estimate_complexity(dummy, base_threshold=config.base_threshold)
    est_input = max(50, len(task_description) // 2)
    est_output = max(50, len(task_description))
    return {
        "task": task_description[:100],
        "suggested_route": decision["route"],
        "reason": decision["reason"],
        "score": decision["score"],
        "estimated_cloud_cost": f"${calc_cost(est_input, est_output):.6f}",
        "will_save": decision["route"] == "local",
    }


def classify_task(task_desc: str, text: str = "") -> dict:
    """细粒度任务分类"""
    config = get_config()
    task = Task(action=task_desc, text=text)
    decision = estimate_complexity(task, base_threshold=config.base_threshold)
    task_type = detect_task_type(task_desc, PROMPT_TEMPLATES)

    return {
        "task": task_desc[:100],
        "task_type": task_type,
        "verdict": decision["route"],
        "score": decision["score"],
        "reason": decision["reason"],
        "confidence": "high" if abs(decision["score"] - config.base_threshold) > 2 else "medium",
        "text_length": len(text),
    }


# ─── 任务分解 ──────────────────────────────────────────────────


def decompose_task(task_description: str, text_content: str = "") -> dict:
    """将大任务拆解为子任务列表"""
    desc = task_description.lower()
    for template_name, template in DECOMPOSE_TEMPLATES.items():
        if any(kw in desc for kw in template["match"]):
            subtasks = [{"id": i + 1, "action": sa, "text": text_content, "route": r}
                        for i, (sa, r) in enumerate(zip(template["subtasks"], template["routes"]))]
            return {"task": task_description[:200], "template": template_name,
                    "total_subtasks": len(subtasks),
                    "local_count": sum(1 for s in subtasks if s["route"] == "local"),
                    "cloud_count": sum(1 for s in subtasks if s["route"] == "cloud"),
                    "subtasks": subtasks}

    # 自动检测
    cl = classify_task(task_description, text_content)
    return {"task": task_description[:200], "template": "auto", "total_subtasks": 1,
            "local_count": 1 if cl["verdict"] == "local" else 0,
            "cloud_count": 1 if cl["verdict"] == "cloud" else 0,
            "subtasks": [{"id": 1, "action": task_description[:200], "text": text_content,
                          "route": cl["verdict"], "reason": cl["reason"]}]}


# ─── 计划执行 ──────────────────────────────────────────────────


def execute_plan(plan: dict) -> dict:
    """执行任务计划"""
    from task_router.task_router import run_task

    results: list[dict] = []
    total_input = total_output = total_time = 0
    total_saved = 0.0

    for i, step in enumerate(plan.get("subtasks", [])):
        task = Task(action=step["action"], text=step.get("text", ""))
        task = run_task(task, force_route=step.get("route", ""))
        total_input += task.tokens_input
        total_output += task.tokens_output
        total_time += task.time_ms
        total_saved += task.cost_saved
        results.append({"id": step.get("id", i + 1), "action": step["action"],
                        "route": task.route, "output": task.output,
                        "tokens_input": task.tokens_input, "tokens_output": task.tokens_output,
                        "time_ms": task.time_ms, "cost_saved": task.cost_saved})

    return {"task": plan.get("task", ""), "total_steps": len(results),
            "local_steps": sum(1 for r in results if r["route"] == "local"),
            "cloud_steps": sum(1 for r in results if "cloud" in r["route"]),
            "total_tokens_input": total_input, "total_tokens_output": total_output,
            "total_time_ms": total_time, "total_cost_saved": round(total_saved, 6), "steps": results}
