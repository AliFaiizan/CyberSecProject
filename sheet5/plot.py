#!/usr/bin/env python3
"""
Task 1 (CNN) — Unified plotting script

Uses the best parts of:
- plot.py (2x2 layout by (k,M) for easy comparison) :contentReference[oaicite:0]{index=0}
- plot2.py (paper-ready ΔF1 heatmap + numeric CSV) :contentReference[oaicite:1]{index=1}

Outputs (in results dir):
- fig_f1_grid.png                (2x2: k={5,10} x M={50,86}, bars = Sc2/Sc3 real/synth with error bars)
- fig_precision_recall.png       (scatter: recall vs precision, size=F1, marker=real/synth, color=scenario)
- fig_delta_f1_heatmap.png       (ΔF1 = synth - real, per scenario, per (k,M))
- task1_summary_numeric.csv      (numeric table for paper)
"""

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------
# Load JSON results
# -----------------------
def load_results_df(results_dir: str) -> pd.DataFrame:
    rows = []
    results_dir = str(results_dir)

    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    for fp in sorted(Path(results_dir).glob("*.json")):
        with open(fp, "r") as f:
            r = json.load(f)

        rows.append({
            "Scenario": int(r["scenario"]),
            "k": int(r["k_folds"]),
            "M": int(r["M"]),
            "Data": str(r["data_type"]).lower(),
            "mean_precision": float(r["mean_precision"]),
            "std_precision": float(r["std_precision"]),
            "mean_recall": float(r["mean_recall"]),
            "std_recall": float(r["std_recall"]),
            "mean_f1": float(r["mean_f1"]),
            "std_f1": float(r["std_f1"]),
            "file": fp.name,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["Data"] = df["Data"].replace({"syn": "synthetic", "synth": "synthetic"})
    df = df.sort_values(["Scenario", "k", "M", "Data"]).reset_index(drop=True)
    return df


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# -----------------------
# Plot 1: 2x2 grid F1 bars (with std error bars)
# -----------------------
def plot_f1_grid(df: pd.DataFrame, outdir: str):
    ensure_dir(outdir)

    configs = [(5, 50), (5, 86), (10, 50), (10, 86)]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Task 1 CNN — F1 by setting (mean ± std)", fontsize=16, fontweight="bold")

    for idx, (k, M) in enumerate(configs):
        ax = axes[idx // 2, idx % 2]

        sub = df[(df["k"] == k) & (df["M"] == M)].copy()
        ax.set_title(f"k={k}, M={M}", fontweight="bold")

        if sub.empty:
            ax.text(0.5, 0.5, "Missing results", ha="center", va="center")
            ax.set_axis_off()
            continue

        # fixed order for readability
        order = []
        labels = []
        colors = []
        means = []
        stds = []

        for sc in [2, 3]:
            for d in ["real", "synthetic"]:
                r = sub[(sub["Scenario"] == sc) & (sub["Data"] == d)]
                if r.empty:
                    continue
                r = r.iloc[0]
                order.append((sc, d))
                labels.append(f"Sc{sc}-{d[:4]}")
                colors.append("steelblue" if d == "real" else "coral")
                means.append(float(r["mean_f1"]))
                stds.append(float(r["std_f1"]))

        x = np.arange(len(labels))
        bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("F1")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25)

        for b, m in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, m + 0.02, f"{m:.3f}", ha="center", va="bottom", fontsize=9)

    # legend once
    from matplotlib.patches import Patch
    fig.legend(
        handles=[
            Patch(facecolor="steelblue", alpha=0.85, label="Real"),
            Patch(facecolor="coral", alpha=0.85, label="Synthetic"),
        ],
        loc="upper right",
        fontsize=11,
    )

    plt.tight_layout()
    out = os.path.join(outdir, "fig_f1_grid.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# -----------------------
# Plot 2: Precision vs Recall scatter (size = F1)
# -----------------------
def plot_precision_recall(df: pd.DataFrame, outdir: str):
    ensure_dir(outdir)

    plt.figure(figsize=(10, 8))

    for _, r in df.iterrows():
        marker = "o" if r["Data"] == "real" else "s"
        color = "blue" if int(r["Scenario"]) == 2 else "red"
        size = 220 * max(0.05, float(r["mean_f1"]))  # keep visible even if tiny F1

        label = f"Sc{r['Scenario']}-{r['Data'][:4]}-k{r['k']}-M{r['M']}"
        plt.scatter(
            float(r["mean_recall"]),
            float(r["mean_precision"]),
            s=size,
            marker=marker,
            color=color,
            alpha=0.75,
            label=label,
        )

    plt.xlabel("Recall (mean)")
    plt.ylabel("Precision (mean)")
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.25)
    plt.title("Task 1 CNN — Precision vs Recall (size = F1)", fontweight="bold")

    # Keep legend outside; still readable
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()

    out = os.path.join(outdir, "fig_precision_recall.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# -----------------------
# Plot 3: ΔF1 heatmap (Synthetic − Real)
# -----------------------
def plot_delta_f1_heatmap(df: pd.DataFrame, outdir: str):
    ensure_dir(outdir)

    real = df[df["Data"] == "real"].copy()
    syn = df[df["Data"] == "synthetic"].copy()
    if real.empty or syn.empty:
        print("Need both real and synthetic results to compute ΔF1 heatmap.")
        return

    def keyframe(x: pd.DataFrame) -> pd.DataFrame:
        x = x.copy()
        x["Cond"] = x.apply(lambda r: f"k{int(r['k'])}-M{int(r['M'])}", axis=1)
        return x[["Scenario", "Cond", "mean_f1"]]

    real_k = keyframe(real).rename(columns={"mean_f1": "f1_real"})
    syn_k = keyframe(syn).rename(columns={"mean_f1": "f1_syn"})

    merged = pd.merge(real_k, syn_k, on=["Scenario", "Cond"], how="inner")
    merged["delta_f1"] = merged["f1_syn"] - merged["f1_real"]

    pivot = merged.pivot(index="Scenario", columns="Cond", values="delta_f1")

    # stable column order: k then M
    def cond_sort(c):
        k = int(c.split("-")[0][1:])
        M = int(c.split("-")[1][1:])
        return (k, M)

    pivot = pivot.reindex(sorted(pivot.columns, key=cond_sort), axis=1).sort_index(axis=0)
    data = pivot.to_numpy(dtype=float)

    plt.figure(figsize=(10, 4))
    im = plt.imshow(data, aspect="auto")  # default colormap

    plt.title("Task 1 CNN — ΔF1 (Synthetic − Real)", fontweight="bold")
    plt.yticks(np.arange(pivot.shape[0]), [f"Sc{int(s)}" for s in pivot.index])
    plt.xticks(np.arange(pivot.shape[1]), pivot.columns, rotation=45, ha="right")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    # annotate
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            plt.text(j, i, f"{data[i, j]:+.3f}", ha="center", va="center", fontsize=10)

    plt.tight_layout()
    out = os.path.join(outdir, "fig_delta_f1_heatmap.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# -----------------------
# Main
# -----------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="task1_results", help="Folder containing Task1 JSON results")
    args = parser.parse_args()

    df = load_results_df(args.results_dir)
    if df.empty:
        print("No JSON results found.")
        return

    # save numeric table for paper
    out_csv = os.path.join(args.results_dir, "task1_summary_numeric.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    plot_f1_grid(df, args.results_dir)
    plot_precision_recall(df, args.results_dir)
    plot_delta_f1_heatmap(df, args.results_dir)

    print("Done.")


if __name__ == "__main__":
    main()
