#!/usr/bin/env python3
"""Render paper Figures 2 and 9 from ``Data/fig2`` simulation tables."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Data" / "fig2"
LABELS = {"exponential": "Exponential", "gaussian": "Gaussian", "hypersphere": "Hypersphere"}
COLORS = {"linear": "#1565C0", "mlp": "#B71C1C"}


def plotting():
    cache = DATA / ".cache" / "matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def mean_sem(frame: pd.DataFrame, x: str, y: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grouped = frame.groupby(x)[y]
    return grouped.mean().index.to_numpy(), grouped.mean().to_numpy(), grouped.sem().fillna(0).to_numpy()


def plot_fig2(summary: pd.DataFrame, output: Path, scale: float) -> None:
    plt = plotting()
    data = summary[np.isclose(summary["teacher_logit_scale"], scale)]
    distributions = ("exponential", "gaussian", "hypersphere")
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.75), sharey=True)
    for index, (distribution, axis) in enumerate(zip(distributions, axes)):
        for encoder in ("linear", "mlp"):
            rows = data[(data["distribution"] == distribution) & (data["enc_type"] == encoder)]
            x, mean, sem = mean_sem(rows, "m", "final_kl")
            axis.plot(x, mean, "o-", color=COLORS[encoder], label=encoder.upper())
            axis.fill_between(x, mean - sem, mean + sem, color=COLORS[encoder], alpha=0.18)
        axis.axvline(8, color="0.35", linestyle="--", linewidth=1)
        axis.set_title(LABELS[distribution])
        axis.set_xlabel("Encoder dimension $m$")
        axis.set_xticks(sorted(data["m"].unique()))
        axis.grid(alpha=0.25)
        if index == 0:
            axis.set_ylabel("Final KL divergence")
        if index == 2:
            axis.legend(frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig9(curves: pd.DataFrame, output: Path, scale: float) -> None:
    plt = plotting()
    data = curves[np.isclose(curves["teacher_logit_scale"], scale)]
    distributions = ("exponential", "gaussian", "hypersphere")
    dimensions = sorted(data["m"].unique())
    palette = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.75), sharey=False)
    for panel, (distribution, axis) in enumerate(zip(distributions, axes)):
        for color_index, dimension in enumerate(dimensions):
            color = palette(color_index / max(1, len(dimensions) - 1))
            for encoder, linestyle in (("linear", "-"), ("mlp", "--")):
                rows = data[
                    (data["distribution"] == distribution)
                    & (data["m"] == dimension)
                    & (data["enc_type"] == encoder)
                ]
                x, mean, _ = mean_sem(rows, "step", "ce_loss")
                axis.plot(x, mean, color=color, linestyle=linestyle, linewidth=1.2)
        axis.set_title(LABELS[distribution])
        axis.set_xlabel("Gradient update step")
        axis.set_ylabel("Cross-entropy loss" if panel == 0 else "")
        axis.grid(alpha=0.25)
    from matplotlib.lines import Line2D

    encoder_legend = [
        Line2D([0], [0], color="black", linestyle="-", label="Linear"),
        Line2D([0], [0], color="black", linestyle="--", label="MLP"),
    ]
    axes[0].legend(handles=encoder_legend, frameon=False, loc="upper right")
    scalar = plt.cm.ScalarMappable(cmap=palette, norm=plt.Normalize(min(dimensions), max(dimensions)))
    fig.colorbar(scalar, ax=axes, label="Encoder dimension $m$", fraction=0.025, pad=0.02)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-scale", type=float, default=10.0)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    summary = pd.read_csv(data_dir / "summary.csv")
    curves = pd.read_csv(data_dir / "curves.csv")
    plot_fig2(summary, data_dir / "fig2.png", args.teacher_scale)
    plot_fig9(curves, data_dir / "fig9.png", args.teacher_scale)
    print(f"Saved {data_dir / 'fig2.png'} and {data_dir / 'fig9.png'}")


if __name__ == "__main__":
    main()
