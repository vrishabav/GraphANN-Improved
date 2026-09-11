#!/usr/bin/env python3
"""
analyze_results.py — Visualise ablation study results and degree histograms.

Run after run_ablation.sh has produced:
  - tmp/ablation_results.tsv
  - tmp/ablation_histograms/hist_*.csv
  - tmp/hard_queries.csv   (from hard_query_analysis)

Produces:
  1. Recall-Latency Pareto curves per condition
  2. Degree histograms overlay (key conditions)
  3. Hard query scatter plot (NN dist vs start dist)
  4. Ablation bar chart at L=75 (recall gain per improvement)
"""

import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

DATA_DIR = Path("tmp")
HIST_DIR  = DATA_DIR / "ablation_histograms"
RESULTS   = DATA_DIR / "ablation_results.tsv"
HQ_CSV    = DATA_DIR / "hard_queries.csv"
OUT_DIR   = DATA_DIR / "plots"
OUT_DIR.mkdir(exist_ok=True)

# ─── Colour scheme ──────────────────────────────────────────────────────────
COLORS = {
    "baseline":    "#6b7280",
    "medoid":      "#3b82f6",
    "two_pass":    "#10b981",
    "random_init": "#f59e0b",
    "strict":      "#ef4444",
    "all_four":    "#8b5cf6",
}

CONDITION_LABELS = {
    (0,0,0,0): "Baseline",
    (1,0,0,0): "Medoid only",
    (0,1,0,0): "Two-pass only",
    (0,0,1,0): "Random init only",
    (0,0,0,1): "Strict degree only",
    (1,1,0,0): "Medoid + Two-pass",
    (1,1,1,1): "All four (paper)",
}

KEY_CONDITIONS = [(0,0,0,0), (1,0,0,0), (0,1,0,0), (1,1,1,1)]

# ─── Load ablation results ────────────────────────────────────────────────────
def load_results():
    df = pd.read_csv(RESULTS, sep="\t")
    df["condition"] = list(zip(df.medoid, df.two_pass,
                               df.random_init, df.stop_degree))
    return df

# ─── Plot 1: Recall-Latency Pareto per condition ──────────────────────────────
def plot_pareto(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    for cond in KEY_CONDITIONS:
        sub = df[df.condition == cond].sort_values("avg_latency_us")
        label = CONDITION_LABELS.get(cond, str(cond))
        color = list(COLORS.values())[KEY_CONDITIONS.index(cond)]
        ax.plot(sub["avg_latency_us"], sub["recall"],
                marker="o", label=label, color=color, linewidth=2, markersize=6)
        # Annotate L values
        for _, row in sub.iterrows():
            if row["L"] in [10, 75, 200]:
                ax.annotate(f"L={int(row['L'])}",
                            (row["avg_latency_us"], row["recall"]),
                            textcoords="offset points", xytext=(6, 2),
                            fontsize=7, color=color)

    ax.set_xlabel("Average Query Latency (µs)", fontsize=12)
    ax.set_ylabel("Recall@10", fontsize=12)
    ax.set_title("Recall–Latency Trade-off by Build Configuration\n(SIFT1M, K=10)",
                 fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "pareto_curves.png", dpi=150)
    plt.close()
    print("Saved: pareto_curves.png")

# ─── Plot 2: Degree histogram overlay ────────────────────────────────────────
def plot_degree_histograms():
    fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
    tags = [
        ("m0_tp0_ri0_sd0", "Baseline (gamma=1.5)", COLORS["baseline"]),
        ("m1_tp0_ri0_sd0", "Medoid only",           COLORS["medoid"]),
        ("m0_tp1_ri0_sd0", "Two-pass only",          COLORS["two_pass"]),
        ("m1_tp1_ri1_sd1", "All four (paper)",       COLORS["all_four"]),
    ]
    for ax, (tag, label, color) in zip(axes, tags):
        path = HIST_DIR / f"hist_{tag}.csv"
        if not path.exists():
            ax.text(0.5, 0.5, "No data\n(run ablation first)",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label); continue
        hist = pd.read_csv(path)
        ax.bar(hist["degree"], hist["count"] / hist["count"].sum() * 100,
               color=color, alpha=0.8, width=0.8)
        ax.axvline(x=32, color="orange", linestyle="--", linewidth=1.5,
                   label="R=32")
        ax.axvline(x=48, color="red", linestyle=":", linewidth=1.5,
                   label="gamma*R=48")
        avg = (hist["degree"] * hist["count"]).sum() / hist["count"].sum()
        ax.axvline(x=avg, color="black", linestyle="-", linewidth=1,
                   label=f"Mean={avg:.1f}")
        ax.set_xlabel("Out-degree"); ax.set_title(label, fontsize=10)
        if ax == axes[0]: ax.set_ylabel("% of nodes")
        ax.legend(fontsize=7); ax.set_xlim(0, 55)

    fig.suptitle("Node Degree Distribution by Build Configuration (SIFT1M)",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "degree_histograms.png", dpi=150)
    plt.close()
    print("Saved: degree_histograms.png")

# ─── Plot 3: Ablation bar chart at L=75 ──────────────────────────────────────
def plot_ablation_bar(df):
    l75 = df[df.L == 75]
    baseline_recall = l75[l75.condition == (0,0,0,0)]["recall"].values
    if len(baseline_recall) == 0:
        print("No L=75 baseline data, skipping bar chart"); return
    base = baseline_recall[0]

    conds = [(0,0,0,0),(1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1),(1,1,0,0),(1,1,1,1)]
    labels, gains = [], []
    for c in conds:
        sub = l75[l75.condition == c]
        if len(sub) == 0: continue
        recall = sub["recall"].values[0]
        labels.append(CONDITION_LABELS.get(c, str(c)))
        gains.append(recall - base)

    colors_bar = [COLORS["baseline"], COLORS["medoid"], COLORS["two_pass"],
                  COLORS["random_init"], COLORS["strict"],
                  "#14b8a6", COLORS["all_four"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(labels)), gains, color=colors_bar[:len(labels)])
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Recall@10 Improvement vs Baseline")
    ax.set_title(f"Ablation Study: Recall@10 Gain at L=75\n(Baseline={base:.4f})",
                 fontsize=12)
    for bar, gain in zip(bars, gains):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
                f"+{gain:.4f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "ablation_bar.png", dpi=150)
    plt.close()
    print("Saved: ablation_bar.png")

# ─── Plot 4: P99/Mean ratio comparison ───────────────────────────────────────
def plot_p99_mean_ratio(df):
    fig, ax = plt.subplots(figsize=(9, 5))
    L_vals = sorted(df.L.unique())
    for cond in KEY_CONDITIONS:
        sub = df[df.condition == cond].sort_values("L")
        if len(sub) == 0: continue
        ratio = sub["p99_latency_us"] / sub["avg_latency_us"]
        label = CONDITION_LABELS.get(cond, str(cond))
        color = list(COLORS.values())[KEY_CONDITIONS.index(cond)]
        ax.plot(sub["L"], ratio, marker="o", label=label, color=color, linewidth=2)

    ax.set_xlabel("Search Beam Width L", fontsize=12)
    ax.set_ylabel("P99 / Mean Latency Ratio", fontsize=12)
    ax.set_title("P99/Mean Latency Ratio — Effect of Medoid Initialization\n(SIFT1M, K=10)",
                 fontsize=12)
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "p99_mean_ratio.png", dpi=150)
    plt.close()
    print("Saved: p99_mean_ratio.png")

# ─── Plot 5: Hard query scatter (NN dist vs start dist) ──────────────────────
def plot_hard_queries():
    if not HQ_CSV.exists():
        print("hard_queries.csv not found, skipping scatter plot"); return
    hq = pd.read_csv(HQ_CSV)
    fig, ax = plt.subplots(figsize=(8, 6))
    easy = hq[hq.is_hard == 0]
    hard = hq[hq.is_hard == 1]
    fixed = hq[(hq.is_hard == 1) & (hq.improved_by_new == 1)]

    ax.scatter(easy.start_dist, easy.nn1_dist, c="steelblue", alpha=0.15,
               s=5, label="Easy (all 10 found)")
    ax.scatter(hard.start_dist, hard.nn1_dist, c="tomato", alpha=0.5,
               s=15, label="Hard (≥1 missed, baseline)")
    ax.scatter(fixed.start_dist, fixed.nn1_dist, c="limegreen", alpha=0.8,
               s=20, marker="^", label="Fixed by improved index")

    ax.set_xlabel("Distance: Start Node → Query", fontsize=11)
    ax.set_ylabel("Distance: Query → True NN₁", fontsize=11)
    ax.set_title("Hard Query Characterisation (L=200, K=10)\nBaseline Index",
                 fontsize=12)
    ax.legend(fontsize=9, markerscale=2)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "hard_query_scatter.png", dpi=150)
    plt.close()
    print("Saved: hard_query_scatter.png")

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Ablation Results Visualisation ===")
    if RESULTS.exists():
        df = load_results()
        print(f"Loaded {len(df)} result rows, "
              f"{df.condition.nunique()} conditions, "
              f"{df.L.nunique()} L values")
        plot_pareto(df)
        plot_ablation_bar(df)
        plot_p99_mean_ratio(df)
    else:
        print(f"Results file not found: {RESULTS}")
        print("Run ./scripts/run_ablation.sh first")

    plot_degree_histograms()
    plot_hard_queries()
    print(f"\nAll plots saved to {OUT_DIR}/")
