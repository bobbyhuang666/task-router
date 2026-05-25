#!/usr/bin/env python3
"""真实工作任务实验 — 开发者日常场景"""

import os, sys, time, json, tempfile
from dataclasses import dataclass, asdict

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
sys.path.insert(0, SCRIPTS_DIR)

from config import get_config
from routing import Task, estimate_complexity, detect_task_type
from models import call_ollama
from reasoning import select_strategy, enhance_prompt_with_strategy
from confidence import extract_confidence
from tqbc import TQBCRouter

WORK_TASKS = [
    {
        "name": "代码审查",
        "prompt": """审查这段代码，找出潜在问题：

def process_orders(orders):
    results = []
    for order in orders:
        if order['status'] == 'pending':
            total = 0
            for item in order['items']:
                total += item['price'] * item['qty']
            if total > 1000:
                total = total * 0.9
            order['total'] = total
            order['status'] = 'processed'
            results.append(order)
    return results""",
        "check": lambda t: len(t) > 50 and any(k in t for k in ["异常", "错误处理", "error", "bug", "问题", "改进", "建议", "缺少", "没有"]),
    },
    {
        "name": "Bug诊断",
        "prompt": """线上服务报错，错误日志如下：

Traceback (most recent call last):
  File "app.py", line 45, in get_user
    user = db.query(User).filter_by(id=user_id).one()
  File "sqlalchemy/orm/query.py", line 3441, in one
    raise NoResultFound("No row was found for one()")
sqlalchemy.exc.NoResultFound: No row was found for one()

请分析原因并给出修复方案。""",
        "check": lambda t: any(k in t for k in ["one()", "first()", "存在", "找不到", "404", "异常处理", "try", "except", "NoResultFound"]),
    },
    {
        "name": "写文档",
        "prompt": """为以下函数写技术文档（参数说明、返回值、使用示例）：

def merge_sorted_lists(list1: list[int], list2: list[int]) -> list[int]:
    result = []
    i = j = 0
    while i < len(list1) and j < len(list2):
        if list1[i] <= list2[j]:
            result.append(list1[i])
            i += 1
        else:
            result.append(list2[j])
            j += 1
    result.extend(list1[i:])
    result.extend(list2[j:])
    return result""",
        "check": lambda t: len(t) > 100 and ("参数" in t or "Args" in t or "返回" in t or "Return" in t or "示例" in t),
    },
    {
        "name": "写SQL",
        "prompt": """数据库表：
- users(id, name, email, created_at)
- orders(id, user_id, amount, status, created_at)
- order_items(id, order_id, product_id, quantity, price)

写SQL：过去30天消费最多的前10个用户，显示用户名、总消费、订单数。""",
        "check": lambda t: "SELECT" in t.upper() and "JOIN" in t.upper() and ("SUM" in t.upper() or "GROUP" in t.upper()),
    },
    {
        "name": "重构建议",
        "prompt": """这段代码能工作但很乱，给出重构建议：

def do_stuff(data, mode):
    if mode == 1:
        result = []
        for d in data:
            if d > 0:
                result.append(d * 2)
        return result
    elif mode == 2:
        result = []
        for d in data:
            if d > 0:
                result.append(d * 3)
        return result
    elif mode == 3:
        result = []
        for d in data:
            if d < 0:
                result.append(d * -1)
        return result
    else:
        return data""",
        "check": lambda t: len(t) > 80 and any(k in t for k in ["提取", "复用", "函数", "DRY", "重构", "策略", "模式", "消除", "重复"]),
    },
    {
        "name": "写邮件",
        "prompt": """写一封邮件通知客户：订单因供应链问题延迟2周发货。
要求：专业礼貌、表达歉意、提供补偿（下次9折）、保持信任。""",
        "check": lambda t: len(t) > 80 and any(k in t for k in ["歉", "延迟", "补偿", "折扣", "感谢", "抱歉"]),
    },
    {
        "name": "技术方案",
        "prompt": """需求：用户上传图片后自动OCR识别文字。
技术栈：Python + FastAPI + PostgreSQL + Redis
请给出技术方案：技术选型、架构、接口设计、注意事项。""",
        "check": lambda t: len(t) > 150 and any(k in t.lower() for k in ["ocr", "tesseract", "paddle", "api", "接口", "存储", "上传"]),
    },
    {
        "name": "写测试",
        "prompt": """为以下函数写 pytest 单元测试：

def validate_password(password: str) -> dict:
    errors = []
    if len(password) < 8:
        errors.append('密码长度至少8位')
    if not any(c.isupper() for c in password):
        errors.append('需要至少一个大写字母')
    if not any(c.isdigit() for c in password):
        errors.append('需要至少一个数字')
    return {'valid': len(errors) == 0, 'errors': errors}""",
        "check": lambda t: "def test" in t and ("assert" in t or "pytest" in t.lower()),
    },
    {
        "name": "数据分析",
        "prompt": """产品最近7天日活：
周一:12500 周二:11800 周三:13200 周四:12800 周五:15600 周六:18900 周日:17200

分析：趋势、异常、可能原因、建议。""",
        "check": lambda t: len(t) > 80 and any(k in t for k in ["增长", "下降", "高峰", "周末", "趋势", "建议", "分析"]),
    },
    {
        "name": "加错误处理",
        "prompt": """给这段代码加完善的错误处理：

import requests

def fetch_user_data(user_id):
    resp = requests.get(f'https://api.example.com/users/{user_id}')
    data = resp.json()
    return {
        'name': data['name'],
        'email': data['email'],
        'active': data['is_active']
    }""",
        "check": lambda t: "try" in t and "except" in t and any(k in t for k in ["timeout", "status", "404", "500", "异常", "error", "raise"]),
    },
    {
        "name": "竞品分析",
        "prompt": """对比分析 PostgreSQL vs MongoDB，从以下维度：
1. 数据模型 2. 查询性能 3. 扩展性 4. 事务支持 5. 适用场景
给出选型建议。""",
        "check": lambda t: len(t) > 150 and "PostgreSQL" in t and "MongoDB" in t,
    },
    {
        "name": "排障指南",
        "prompt": """用户报告：网页加载很慢，有时超时。
请给出系统化的排障步骤，从前端到后端到基础设施逐层排查。""",
        "check": lambda t: len(t) > 100 and any(k in t for k in ["网络", "DNS", "数据库", "服务器", "前端", "CDN", "带宽", "排查"]),
    },
]


def main():
    print("=" * 65)
    print("  真实工作任务实验")
    print(f"  模型: {get_config().local_model}")
    print(f"  任务: {len(WORK_TASKS)} 个开发者日常场景")
    print("=" * 65)

    with tempfile.TemporaryDirectory() as tmpdir:
        tqbc = TQBCRouter(os.path.join(tmpdir, "tqbc"))
        results = []

        for i, task in enumerate(WORK_TASKS):
            task_id = i + 1
            print(f"\n  [{task_id:>2}] {task['name']:<12}", end=" ", flush=True)

            t0 = time.time()
            r = call_ollama(task["prompt"], with_logprobs=True, max_tokens=600)
            latency = int((time.time() - t0) * 1000)
            output = r["text"]
            logprobs = r.get("logprobs", [])
            conf = extract_confidence(logprobs) if logprobs else {"confidence": 0}

            td = detect_task_type(task["prompt"][:30], {})
            routing = estimate_complexity(Task(action=task["prompt"][:50], text=task["prompt"]))
            sd = select_strategy(action=task["prompt"][:100], text=task["prompt"],
                                 complexity_score=routing["score"], task_type=td, logprobs=logprobs)
            tqbc_d = tqbc.decide(logprobs=logprobs, complexity_score=routing["score"], task_type=td)
            tqbc.record_outcome(decision=tqbc_d, success=task["check"](output),
                               escalated=tqbc_d.should_escalate, task_type=td)

            quality = task["check"](output)
            qual = "V" if quality else "X"
            route = "esc" if tqbc_d.should_escalate else "loc"

            results.append({
                "id": task_id, "name": task["name"],
                "quality": quality, "strategy": sd.strategy,
                "route": route, "tqbc_conf": round(tqbc_d.calibrated_confidence, 4),
                "model_conf": round(conf["confidence"], 4),
                "tokens": r["tokens_output"], "latency": latency, "output": output,
            })

            print(f"{qual} {sd.strategy:<10} {route} tqbc={tqbc_d.calibrated_confidence:.3f} "
                  f"model={conf['confidence']:.3f} {latency:>6}ms {r['tokens_output']:>4}tok")

            if not quality:
                print(f"    输出前120字: {output[:120]}")

        # 汇总
        passed = sum(1 for r in results if r["quality"])
        total = len(results)
        avg_lat = sum(r["latency"] for r in results) / total
        avg_tok = sum(r["tokens"] for r in results) / total

        print(f"\n{'='*65}")
        print(f"  结果汇总")
        print(f"{'='*65}")
        print(f"  通过: {passed}/{total} = {passed/total:.0%}")
        print(f"  平均延迟: {avg_lat:.0f}ms")
        print(f"  平均 token: {avg_tok:.0f}")

        print(f"\n  {'#':<4} {'任务':<12} {'策略':<10} {'路由':<6} {'质量':<6} {'延迟':<8} {'token':<8} {'TQBC':<8} {'模型'}")
        print(f"  {'-'*72}")
        for r in results:
            q = "V" if r["quality"] else "X"
            print(f"  {r['id']:<4} {r['name']:<12} {r['strategy']:<10} {r['route']:<6} "
                  f"{q:<6} {r['latency']:>5}ms {r['tokens']:>5}tok {r['tqbc_conf']:.3f}  {r['model_conf']:.3f}")

        # 策略分布
        strat_dist = {}
        for r in results:
            strat_dist[r["strategy"]] = strat_dist.get(r["strategy"], 0) + 1
        print(f"\n  策略分布: {strat_dist}")

        # 保存
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment_work_results.json")
        with open(out, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n  结果已保存: {out}")


if __name__ == "__main__":
    main()
