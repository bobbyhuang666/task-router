#!/usr/bin/env python3
"""生成论文 LaTeX 表格（完整版，含 7 个表格）"""
import json
import os
import time


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def generate_tables(results_dir):
    ms = load_json(os.path.join(results_dir, "multiseed_routing.json"))
    ab = load_json(os.path.join(results_dir, "ablation.json"))
    lt = load_json(os.path.join(results_dir, "latency.json"))
    lj = load_json(os.path.join(results_dir, "llm_judge.json"))
    mc = load_json(os.path.join(results_dir, "model_comparison.json"))
    pa = load_json(os.path.join(results_dir, "pareto.json"))
    lc = load_json(os.path.join(results_dir, "learning_curve.json"))

    L = []
    L.append(r"\documentclass{article}")
    L.append(r"\usepackage{booktabs,amsmath,geometry,multirow}")
    L.append(r"\geometry{margin=1in}")
    L.append(r"\begin{document}")

    # ─── Table 1: Routing Accuracy & Ablation ───
    L.append(r"\begin{table*}[t]")
    L.append(r"\centering")
    L.append(r"\caption{Routing Accuracy Comparison (mean $\pm$ 95\% CI, $N{=}10$ seeds $\times$ 60 tasks)}")
    L.append(r"\label{tab:accuracy}")
    L.append(r"\begin{tabular}{lcccl}")
    L.append(r"\toprule")
    L.append(r"\textbf{Configuration} & \textbf{Accuracy (\%)} & \textbf{95\% CI} & \textbf{Cost Savings (\%)} & \textbf{Source} \\")
    L.append(r"\midrule")
    if ms:
        a = ms.get("accuracy", {})
        s = ms.get("savings", {})
        L.append(
            r"\textbf{Full TQBC (ours)} & "
            f"\\textbf{{{a.get('mean',0):.2f}}} & "
            f"$[{a.get('ci_lower',0):.2f},\\; {a.get('ci_upper',0):.2f}]$ & "
            f"\\textbf{{{s.get('mean',0):.2f}}} & "
            r"Multi-seed \\"
        )
    if ab:
        for name, data in ab.get("configs", {}).items():
            if name == "Full TQBC" and ms:
                continue
            a = data.get("accuracy", {})
            s = data.get("savings", {})
            dn = name.replace("TQBC - ", r"w/o ")
            L.append(
                f"{dn} & "
                f"{a.get('mean',0):.2f} & "
                f"$[{a.get('ci_lower',0):.2f},\\; {a.get('ci_upper',0):.2f}]$ & "
                f"{s.get('mean',0):.2f} & "
                r"Ablation \\"
            )
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table*}")

    # ─── Table 2: Conformal Coverage ───
    L.append(r"\begin{table*}[t]")
    L.append(r"\centering")
    L.append(r"\caption{Adaptive Conformal Coverage and Calibration}")
    L.append(r"\label{tab:coverage}")
    L.append(r"\begin{tabular}{lcccc}")
    L.append(r"\toprule")
    L.append(r"\textbf{Configuration} & \textbf{Coverage (\%)} & \textbf{95\% CI} & \textbf{ECE} & \textbf{Gap} \\")
    L.append(r"\midrule")
    if ms:
        c = ms.get("coverage", {})
        L.append(
            r"\textbf{Full TQBC (ours)} & "
            f"\\textbf{{{c.get('mean',0):.2f}}} & "
            f"$[{c.get('ci_lower',0):.2f},\\; {c.get('ci_upper',0):.2f}]$ & "
            r"-- & -- \\"
        )
    if ab:
        for name, data in ab.get("configs", {}).items():
            if name == "Full TQBC" and ms:
                continue
            c = data.get("coverage", {})
            dn = name.replace("TQBC - ", r"w/o ")
            L.append(
                f"{dn} & "
                f"{c.get('mean',0):.2f} & "
                f"$[{c.get('ci_lower',0):.2f},\\; {c.get('ci_upper',0):.2f}]$ & "
                r"-- & -- \\"
            )
    if lc:
        # Add learning curve summary
        s = lc.get("summary", {})
        gap_range = s.get("gap_range", [0, 0])
        ece_range = s.get("ece_range", [0, 0])
        L.append(r"\midrule")
        L.append(
            r"\multicolumn{5}{l}{\textit{Learning Curve (summary):}} \\"
        )
        L.append(
            f"& ECE range: $[{ece_range[0]:.4f},\\; {ece_range[1]:.4f}]$ & "
            f"Gap range: $[{gap_range[0]:.4f},\\; {gap_range[1]:.4f}]$ & "
            f"Rounds: {s.get('total_rounds',0)} & -- \\\\"
        )
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table*}")

    # ─── Table 3: Latency Microbenchmark ───
    L.append(r"\begin{table}[t]")
    L.append(r"\centering")
    L.append(r"\caption{Routing Decision Latency (10K iterations, $\mu$s)}")
    L.append(r"\label{tab:latency}")
    L.append(r"\begin{tabular}{lcccc}")
    L.append(r"\toprule")
    L.append(r"\textbf{Component} & \textbf{p50} & \textbf{p95} & \textbf{p99} & \textbf{mean} \\")
    L.append(r"\midrule")
    if lt:
        res = lt.get("results", {})
        comps = [
            ("decide_serial", "Full decide()"),
            ("quantile_extract", "Quantile Extract"),
            ("thompson_select", "Thompson Sampling"),
            ("bayesian_calibration", "Bayesian Calibration"),
            ("decide_threaded", "decide() (4-thread)"),
        ]
        for key, label in comps:
            d = res.get(key, {})
            if d:
                L.append(
                    f"{label} & "
                    f"{d.get('p50_us',0):.1f} & "
                    f"{d.get('p95_us',0):.1f} & "
                    f"{d.get('p99_us',0):.1f} & "
                    f"{d.get('mean_us',0):.1f} \\\\"
                )
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")

    # ─── Table 4: LLM-as-Judge Quality ───
    if lj:
        dims_order = ["translation", "summarization", "classification", "code", "qa", "analysis"]
        L.append(r"\begin{table}[t]")
        L.append(r"\centering")
        L.append(r"\caption{LLM-as-Judge Quality (qwen2.5:3b, 5 dimensions, 1--5)}")
        L.append(r"\label{tab:judge}")
        L.append(r"\begin{tabular}{lcc}")
        L.append(r"\toprule")
        L.append(r"\textbf{Category} & \textbf{Avg Score} & \textbf{n} \\")
        L.append(r"\midrule")
        for cat in dims_order:
            if cat not in lj.get("by_category", {}):
                continue
            data = lj["by_category"][cat]
            L.append(f"{cat.capitalize()} & {data.get('avg_quality',0):.2f} & {data.get('n',0)} \\\\")
        L.append(r"\midrule")
        s = lj.get("summary", {})
        L.append(
            r"\textbf{Overall} & "
            f"\\textbf{{{s.get('overall_quality',0):.2f}}} & "
            f"{s.get('n_tasks',0)} \\\\"
        )
        L.append(r"\midrule")
        L.append(r"\multicolumn{3}{l}{\textit{Dimension Averages (1--5):}} \\")
        da = s.get("dimension_averages", {})
        for dim, label in [("relevance", "Relevance"), ("accuracy", "Accuracy"),
                           ("fluency", "Fluency"), ("completeness", "Completeness"),
                           ("coherence", "Coherence")]:
            L.append(f"& {label}: {da.get(dim,0):.2f} & \\\\")
        L.append(r"\bottomrule")
        L.append(r"\end{tabular}")
        L.append(r"\end{table}")

    # ─── Table 5: Local Model Comparison ───
    if mc:
        L.append(r"\begin{table}[t]")
        L.append(r"\centering")
        L.append(r"\caption{Local Model Comparison (30 real tasks)}")
        L.append(r"\label{tab:model}")
        L.append(r"\begin{tabular}{lcccc}")
        L.append(r"\toprule")
        L.append(r"\textbf{Model} & \textbf{Quality} & \textbf{Latency (ms)} & \textbf{Routing Acc.} & \textbf{Avg Len} \\")
        L.append(r"\midrule")
        for m in mc.get("models", []):
            s = mc.get(m, {}).get("summary", {})
            L.append(
                f"{m} & "
                f"{s.get('avg_quality',0):.3f} & "
                f"{s.get('avg_latency_ms',0):.0f} & "
                f"{s.get('routing_accuracy_pct',0):.1f}\\% & "
                f"{s.get('avg_output_len',0)} \\\\"
            )
        L.append(r"\bottomrule")
        L.append(r"\end{tabular}")
        L.append(r"\end{table}")

    # ─── Table 6: Pareto Curve ───
    if pa:
        L.append(r"\begin{table}[t]")
        L.append(r"\centering")
        L.append(r"\caption{Cost--Quality Pareto Frontier (escalation threshold sweep)}")
        L.append(r"\label{tab:pareto}")
        L.append(r"\begin{tabular}{ccccc}")
        L.append(r"\toprule")
        L.append(r"$\tau$ & \textbf{Accuracy (\%)} & \textbf{Cost Savings (\%)} & \textbf{Quality} & \textbf{Local/Cloud} \\")
        L.append(r"\midrule")
        best_tb = pa.get("best_balance", {}).get("threshold", -1)
        for pt in pa.get("pareto_points", []):
            mark = r" $\ast$" if pt.get("threshold") == best_tb else ""
            L.append(
                f"{pt.get('threshold',0):.2f}{mark} & "
                f"{pt.get('accuracy_pct',0):.1f} & "
                f"{pt.get('cost_savings_pct',0):.1f} & "
                f"{pt.get('avg_quality',0):.3f} & "
                f"{pt.get('local_count',0)}/{pt.get('cloud_count',0)} \\\\"
            )
        L.append(r"\bottomrule")
        L.append(r"\end{tabular}")
        L.append(r"\end{table}")

    # ─── Table 7: Learning Curve ───
    if lc:
        L.append(r"\begin{table}[t]")
        L.append(r"\centering")
        L.append(r"\caption{TQBC Online Learning Curve (Accuracy, ECE, Gap)}")
        L.append(r"\label{tab:learning}")
        L.append(r"\begin{tabular}{cccc}")
        L.append(r"\toprule")
        L.append(r"\textbf{Round} & \textbf{Accuracy (\%)} & \textbf{ECE} & \textbf{Gap} \\")
        L.append(r"\midrule")
        for p in lc.get("checkpoints", []):
            L.append(
                f"{p.get('round',0):,} & "
                f"{p.get('accuracy',0)*100:.1f} & "
                f"{p.get('ece',0):.4f} & "
                f"{p.get('gap',0):.4f} \\\\"
            )
        L.append(r"\bottomrule")
        L.append(r"\end{tabular}")
        L.append(r"\end{table}")

    L.append(r"\end{document}")
    return "\n".join(L)


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rd = os.path.join(base, "results")
    print("生成 LaTeX 表格...")
    latex = generate_tables(rd)
    out = os.path.join(rd, "tables.tex")
    with open(out, "w") as f:
        f.write(latex)
    n_tables = latex.count(r"\end{table") + latex.count(r"\end{figure")
    print(f"已保存到 {out}")
    print(f"包含 {n_tables} 个表格/图形")
    print(f"文件大小: {len(latex)} bytes")
