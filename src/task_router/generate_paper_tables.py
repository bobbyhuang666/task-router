#!/usr/bin/env python3
"""
实验 5: 生成 LaTeX 表格

读取 results/ 下所有 JSON，生成 results/tables.tex，包含：
- Table 1: 路由准确率对比（多方法，含消融）
- Table 2: Conformal 覆盖率与成本节省
- Table 3: 路由决策延迟
"""

import json
import os


def load_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def fmt_ci(mean: float, ci_lower: float = None, ci_upper: float = None) -> str:
    """格式化 mean ± CI"""
    if ci_lower is not None and ci_upper is not None:
        return f"{mean:.2f} & [{ci_lower:.2f}, {ci_upper:.2f}]"
    return f"{mean:.2f}"


def generate_tables(results_dir: str) -> str:
    """生成 LaTeX 表格"""
    multiseed = load_json(os.path.join(results_dir, "multiseed_routing.json"))
    ablation = load_json(os.path.join(results_dir, "ablation.json"))
    latency = load_json(os.path.join(results_dir, "latency.json"))
    learning = load_json(os.path.join(results_dir, "learning_curve.json"))

    lines = []
    lines.append(r"\documentclass{article}")
    lines.append(r"\usepackage{booktabs}")
    lines.append(r"\usepackage{amsmath}")
    lines.append(r"\usepackage{geometry}")
    lines.append(r"\geometry{margin=1in}")
    lines.append(r"\begin{document}")
    lines.append("")

    # ─── Table 1: 路由准确率对比 ────────────────────────
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Routing Accuracy Comparison (mean $\pm$ 95\\% CI)}")
    lines.append(r"\label{tab:accuracy}")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Method} & \textbf{Accuracy (\%)} & \textbf{95\\% CI} \\")
    lines.append(r"\midrule")

    # 多种子结果
    if multiseed:
        acc = multiseed.get("accuracy", {})
        lines.append(
            f"Full TQBC (ours) & {acc.get('mean', 0):.2f} "
            f"& $[{acc.get('ci_lower', 0):.2f},\\, {acc.get('ci_upper', 0):.2f}]$ \\\\"
        )

    # 消融结果
    if ablation:
        for name, data in ablation.get("configs", {}).items():
            acc = data.get("accuracy", {})
            display_name = name.replace("TQBC - ", "w/o ")
            if name == "Full TQBC" and multiseed:
                continue  # 已有多种子结果
            lines.append(
                f"{display_name} & {acc.get('mean', 0):.2f} "
                f"& $[{acc.get('ci_lower', 0):.2f},\\, {acc.get('ci_upper', 0):.2f}]$ \\\\"
            )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    lines.append("")

    # ─── Table 2: Conformal 覆盖率与成本节省 ─────────────
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Conformal Coverage and Cost Savings}")
    lines.append(r"\label{tab:coverage_cost}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Method} & \textbf{Coverage (\%)} & \textbf{Cost Savings (\%)} & \textbf{CI (Cov.)} \\")
    lines.append(r"\midrule")

    if multiseed:
        cov = multiseed.get("coverage", {})
        sav = multiseed.get("savings", {})
        lines.append(
            f"Full TQBC (ours) & {cov.get('mean', 0):.2f} & {sav.get('mean', 0):.2f} "
            f"& $[{cov.get('ci_lower', 0):.2f},\\, {cov.get('ci_upper', 0):.2f}]$ \\\\"
        )

    if ablation:
        for name, data in ablation.get("configs", {}).items():
            cov = data.get("coverage", {})
            sav = data.get("savings", {})
            display_name = name.replace("TQBC - ", "w/o ")
            if name == "Full TQBC" and multiseed:
                continue
            lines.append(
                f"{display_name} & {cov.get('mean', 0):.2f} & {sav.get('mean', 0):.2f} "
                f"& $[{cov.get('ci_lower', 0):.2f},\\, {cov.get('ci_upper', 0):.2f}]$ \\\\"
            )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    lines.append("")

    # ─── Table 3: 路由决策延迟 ────────────────────────
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Routing Decision Latency ($\mu$s)}")
    lines.append(r"\label{tab:latency}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Component} & \textbf{p50} & \textbf{p95} & \textbf{p99} \\")
    lines.append(r"\midrule")

    if latency:
        results = latency.get("results", {})
        components = [
            ("decide_serial", "Full decide()"),
            ("quantile_extract", "Quantile Extract"),
            ("thompson_select", "Thompson Sampling"),
            ("bayesian_calibration", "Calibration"),
            ("decide_threaded", "decide() (4-thread)"),
        ]
        for key, label in components:
            data = results.get(key, {})
            if data:
                lines.append(
                    f"{label} & {data.get('p50_us', 0):.0f} "
                    f"& {data.get('p95_us', 0):.0f} "
                    f"& {data.get('p99_us', 0):.0f} \\\\"
                )

        # 吞吐量行
        serial = results.get("decide_serial", {})
        threaded = results.get("decide_threaded", {})
        if serial and serial.get("mean_us", 0) > 0:
            throughput_serial = 1_000_000 / serial["mean_us"]
            lines.append(r"\midrule")
            lines.append(
                f"Throughput (serial) & \\multicolumn{{3}}{{c}}{{{throughput_serial:.0f} ops/s}} \\\\"
            )
            if threaded:
                lines.append(
                    f"Throughput (4-thread) & \\multicolumn{{3}}{{c}}{{{threaded.get('throughput_ops_per_sec', 0):.0f} ops/s}} \\\\"
                )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    # ─── 学习曲线数据（如果有） ────────────────────────
    if learning:
        lines.append(r"\begin{figure}[t]")
        lines.append(r"\centering")
        lines.append(r"\caption{TQBC Learning Curve}")
        lines.append(r"\label{fig:learning}")
        lines.append(r"\begin{tabular}{cccc}")
        lines.append(r"\toprule")
        lines.append(r"\textbf{Round} & \textbf{Accuracy} & \textbf{ECE} & \textbf{Gap} \\")
        lines.append(r"\midrule")
        for point in learning.get("checkpoints", []):
            lines.append(
                f"{point.get('round', 0)} & {point.get('accuracy', 0) * 100:.1f}\\% "
                f"& {point.get('ece', 0):.4f} & {point.get('gap', 0):.4f} \\\\"
            )
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{figure}")
        lines.append("")

    lines.append(r"\end{document}")

    return "\n".join(lines)


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "results")

    print("生成 LaTeX 表格...")
    print(f"  读取 {results_dir}/")

    # 检查可用文件
    for fname in ["multiseed_routing.json", "ablation.json", "latency.json", "learning_curve.json"]:
        path = os.path.join(results_dir, fname)
        status = "✓" if os.path.exists(path) else "✗"
        print(f"  {status} {fname}")

    latex = generate_tables(results_dir)

    output_path = os.path.join(results_dir, "tables.tex")
    with open(output_path, "w") as f:
        f.write(latex)
    print(f"\nLaTeX 表格已保存到 {output_path}")

    # 打印摘要
    table_count = latex.count(r"\begin{table")
    fig_count = latex.count(r"\begin{figure")
    print(f"  包含 {table_count} 个表格, {fig_count} 个图")


if __name__ == "__main__":
    main()
