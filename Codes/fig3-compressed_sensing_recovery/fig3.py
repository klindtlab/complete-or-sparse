#!/usr/bin/env python3
"""Render Figure 3 from the compressed-sensing simulation tables."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Data" / "fig3"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    cache = data_dir / ".cache" / "matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = pd.read_csv(data_dir / "curves.csv")
    summary = pd.read_csv(data_dir / "summary.csv")
    rank_check = pd.read_csv(data_dir / "rank-check.csv")
    if not rank_check["nsp_holds"].astype(bool).all():
        raise ValueError("At least one measurement matrix failed the order-2k rank check.")

    colors = plt.get_cmap("viridis")
    k_values = sorted(summary["k"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.9))
    for index, k in enumerate(k_values):
        color = colors(index / max(1, len(k_values) - 1))
        rows = curves[curves["k"] == k]
        for metric, axis in (("kl_loss", axes[0]), ("row_space_match", axes[1])):
            grouped = rows.groupby("step")[metric]
            x = grouped.mean().index.to_numpy()
            mean = grouped.mean().to_numpy()
            sem = grouped.sem().fillna(0).to_numpy()
            axis.plot(x, mean, color=color, label=f"k={k}")
            axis.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.16)
    axes[0].set_title("Prediction matching")
    axes[0].set_ylabel("KL divergence")
    axes[1].set_title("Row-space recovery")
    axes[1].set_ylabel("Subspace match")
    for axis in axes[:2]:
        axis.set_xlabel("Gradient update step")
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)

    positions = np.arange(len(k_values))
    width = 0.24
    metrics = (
        ("A_z__linear_OLS__r2", "Linear OLS"),
        ("A_z__Lasso__r2", "Supervised Lasso"),
        ("unsup_lasso__mcc", "Unsupervised MCC"),
    )
    for offset, (metric, label) in enumerate(metrics):
        grouped = summary.groupby("k")[metric]
        means = grouped.mean().reindex(k_values).to_numpy()
        sem = grouped.sem().fillna(0).reindex(k_values).to_numpy()
        axes[2].bar(positions + (offset - 1) * width, means, width, yerr=sem, label=label)
    axes[2].axhline(8 / 16, color="0.35", linestyle="--", linewidth=1, label="$n'/n$")
    axes[2].set_xticks(positions, [f"k={k}" for k in k_values])
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("Latent recovery")
    axes[2].set_ylabel("Recovery score")
    axes[2].legend(frameon=False, fontsize=7)
    axes[2].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output = data_dir / "fig3.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
