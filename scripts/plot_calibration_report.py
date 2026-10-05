#!/usr/bin/env python3
"""Render a validated M7 report using headless Matplotlib; no training dependencies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from calibration_report import BUDGETS, CURVE_USERS, DEMO_WARNING, require
from calibration_sampler import digest


def build_figure(report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    require(report.get("report_digest") == digest({k: v for k, v in report.items() if k != "report_digest"}),
            "Report digest mismatch; regenerate from receipts")
    require(report.get("kind") in ("measured_m7_first_curve", "synthetic_pipeline_demo"), "Unrecognized report kind")
    demo = report["kind"] == "synthetic_pipeline_demo"
    require((report.get("warning") == DEMO_WARNING) if demo else report.get("warning") is None,
            "Report label does not match provenance")
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.7), sharey=True)
    for ax, user in zip(axes, CURVE_USERS):
        points = [r for r in report["curve"] if r["user"] == user]
        require([r["budget_minutes"] for r in points] == list(BUDGETS), "Incomplete or unordered plot points")
        x, y = [r["selected_minutes"] for r in points], [r["test_CER_percent"] for r in points]
        require(all(v > 0 for v in x), "Log axis requires positive selected minutes")
        ax.plot(x, y, "o-", color="#176b91", linewidth=1.7, markersize=5, label="Full fine-tuning")
        ax.scatter(x[-1], y[-1], marker="D", s=65, color="#176b91", zorder=4, label="Full recording budget")
        ax.axhline(report["generic_baselines"][user]["test_CER_percent"], linestyle="--",
                   color="#7b7284", linewidth=1.4, label="Generic (0 min; M4 reference)" if not demo else "Fictional generic reference")
        ax.annotate(f"full: {x[-1]:.1f} min", (x[-1], y[-1]), xytext=(-5, 10), textcoords="offset points",
                    ha="right", fontsize=8)
        ax.set_xscale("log")
        ax.set_xticks(sorted(set([1, 2, 5, 10, 30, 60, x[-1]])))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.tick_params(axis="x", labelsize=8)
        ax.set_title(f"{user} · {'tuning user' if user in ('user0', 'user1') else 'untouched user'}", fontsize=11)
        ax.set_xlabel("Labeled calibration minutes")
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("Test CER (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 0.04))
    gate = "PASS" if report["scientific_gate"]["passed"] else "FAIL — retain and diagnose"
    fig.suptitle(DEMO_WARNING if demo else "M7 · calibration data efficiency · full fine-tuning", fontsize=12,
                 color="#a13629" if demo else "#222222", fontweight="bold")
    fig.text(0.5, 0.015, f"One seed; exploratory · scientific gate: {gate}", ha="center", fontsize=9)
    if demo:
        fig.text(0.5, 0.53, "FICTIONAL VALUES", ha="center", fontsize=37, color="#a13629", alpha=0.13, rotation=12)
    fig.tight_layout(rect=(0, 0.14, 1, 0.93))
    return fig


def save_figure(report, output):
    figure = build_figure(report)
    import matplotlib.pyplot as plt
    try:
        output.mkdir(parents=True, exist_ok=False)
        for extension in ("png", "pdf", "svg"):
            figure.savefig(output / f"cer-vs-minutes.{extension}", dpi=180,
                           metadata={"Creator": "emg-calibration offline report"} if extension in ("pdf", "svg") else None)
    finally:
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    save_figure(json.loads(args.report.read_text()), args.output_dir)


if __name__ == "__main__":
    main()
