"""Render audited pilot aggregates and raw seed observations; never invent or interpolate results."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project_directory")
    args = parser.parse_args()
    root = Path(args.project_directory)
    report = json.loads((root / "report.json").read_text())
    with (root / "trials.csv").open() as handle:
        trials = list(csv.DictReader(handle))
    rows = report["comparisons"]
    if any(row["paired_runs"] != row["expected_pairs"] for row in rows):
        raise ValueError("figure requires complete comparable seed pairs; inspect report first")
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 11, "axes.labelsize": 11, "axes.titlesize": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "legend.frameon": False,
        "savefig.dpi": 300,
    })
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw={"width_ratios": [1.25, 1]})
    x = np.arange(len(rows))
    labels = [row["dataset"].replace("_", " ").title() for row in rows]
    for offset, name, color, marker, field in (
        (-0.18, "baseline", "#0072B2", "o", "baseline"),
        (0.18, "standardized", "#009E73", "s", "candidate"),
    ):
        means = [100 * row[field + "_mean"] for row in rows]
        sd = [100 * row[field + "_sd"] for row in rows]
        left.bar(x + offset, means, width=0.32, color=color, alpha=0.85,
                 label="No scaling" if name == "baseline" else "Train-fit standardization",
                 yerr=sd, capsize=4, error_kw={"lw": 1.2}, zorder=2)
        for i, row in enumerate(rows):
            values = [100 * float(trial["value"]) for trial in trials
                      if trial["dataset"] == row["dataset"] and trial["variant"] == name]
            left.scatter(x[i] + offset + np.linspace(-0.06, 0.06, len(values)), values,
                         marker=marker, s=18, facecolor="white", edgecolor="#222222", linewidth=0.6, zorder=3)
    left.set(xticks=x, xticklabels=labels, ylabel="Validation accuracy (%)", ylim=(0, 105),
             title="A  Fixed-budget MLP: mean ± sample SD")
    left.grid(axis="y", alpha=0.18, zorder=0)
    left.legend(loc="lower right", fontsize=9)
    values = np.array([100 * row["oriented_delta_mean"] for row in rows])
    intervals = np.array([row["bootstrap_95_interval"] for row in rows]) * 100
    right.errorbar(values, x, xerr=np.array([values - intervals[:, 0], intervals[:, 1] - values]),
                   fmt="D", color="#D55E00", capsize=5, markersize=6, linewidth=1.7)
    right.axvline(0, color="#555555", lw=1)
    right.axvline(report["minimum_effect"] * 100, color="#888888", lw=1, ls="--")
    right.set(yticks=x, yticklabels=labels, xlabel="Accuracy improvement (percentage points)",
              title="B  Paired mean difference with 95% interval")
    right.invert_yaxis()
    right.margins(x=0.15, y=0.35)
    right.grid(axis="x", alpha=0.18)
    fig.suptitle("TracePilot reproducible ML pilot", fontsize=16, fontweight="bold", y=1.01)
    fig.text(0.5, -0.02,
             "5 training seeds on one fixed split per dataset · 80 epochs · validation only\n"
             "Intervals: paired bootstrap over seeds (exploratory); predefined comparison, no novelty claim.",
             ha="center", va="top", fontsize=10, color="#444444")
    fig.tight_layout(w_pad=3)
    for extension in ("pdf", "png"):
        fig.savefig(root / ("fig_pilot." + extension), bbox_inches="tight")
    plt.close(fig)
    print(root / "fig_pilot.pdf")


if __name__ == "__main__":
    main()
