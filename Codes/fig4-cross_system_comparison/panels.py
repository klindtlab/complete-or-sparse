#!/usr/bin/env python3
"""Render Figure 4A--D from the compact plot-level CSV files."""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Data" / "fig4"

DISPLAY = {
    "vgg19": "VGG19-BN",
    "resnet50": "ResNet-50",
    "vit-b16": "ViT-B/16",
    "dinov2-s14": "DINOv2-S/14",
    "gpt2": "GPT-2",
    "pythia-70m": "Pythia-70M",
    "tinyllama": "TinyLlama",
}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_plotting() -> object:
    cache = DATA / ".cache" / "matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def verify_correlations(
    points: list[dict[str, str]], correlations: list[dict[str, str]]
) -> None:
    grouped: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for row in points:
        grouped[(row["source"], row["target"])].append(
            (float(row["mis"]), float(row["mmu"]))
        )
    for row in correlations:
        values = np.asarray(grouped[(row["source"], row["target"])])
        observed = float(np.corrcoef(values[:, 0], values[:, 1])[0, 1])
        if len(values) != int(row["n"]) or not math.isclose(
            observed, float(row["r"]), rel_tol=0, abs_tol=2e-12
        ):
            raise ValueError(f"Correlation audit failed for {row['source']} -> {row['target']}")


def scatter_panel(
    plt: object,
    points: list[dict[str, str]],
    models: list[str],
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, len(models), figsize=(2.5 * len(models), 2.6), squeeze=False)
    palette = plt.get_cmap("tab10")
    for source_index, (source, axis) in enumerate(zip(models, axes[0])):
        targets = [model for model in models if model != source]
        for target_index, target in enumerate(targets):
            group = [
                row for row in points
                if row["source"] == source and row["target"] == target
            ]
            x = np.asarray([float(row["mis"]) for row in group])
            y = np.asarray([float(row["mmu"]) for row in group])
            r_value = float(np.corrcoef(x, y)[0, 1])
            axis.scatter(
                x, y, s=4, alpha=0.28, linewidths=0,
                color=palette(target_index),
                label=f"{DISPLAY[target]} (r={r_value:.2f})",
            )
        axis.set_title(DISPLAY[source])
        axis.set_xlabel("MIS")
        if source_index == 0:
            axis.set_ylabel("MMU")
        axis.grid(alpha=0.2, linewidth=0.4)
        axis.legend(frameon=False, fontsize=6, markerscale=1.6)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def correlation_panel(
    plt: object,
    rows: list[dict[str, str]],
    models: list[str],
    output: Path,
) -> None:
    matrix = np.full((len(models), len(models)), np.nan)
    for row in rows:
        matrix[models.index(row["source"]), models.index(row["target"])] = float(row["r"])
    figure, axis = plt.subplots(figsize=(3.25, 2.9))
    image = axis.imshow(matrix, vmin=0, vmax=0.65, cmap="viridis")
    for i in range(len(models)):
        for j in range(len(models)):
            if np.isfinite(matrix[i, j]):
                axis.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    labels = [DISPLAY[model] for model in models]
    axis.set_xticks(range(len(models)), labels, rotation=35, ha="right")
    axis.set_yticks(range(len(models)), labels)
    axis.set_xlabel("Target")
    axis.set_ylabel("Source")
    figure.colorbar(image, ax=axis, label="Pearson r", fraction=0.046)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def render(name: str, models: list[str], output_dir: Path) -> None:
    points = load_rows(DATA / f"{name}-points.csv")
    correlations = load_rows(DATA / f"{name}-correlations.csv")
    verify_correlations(points, correlations)
    plt = load_plotting()
    scatter_panel(plt, points, models, output_dir / f"{name}-scatter.pdf")
    correlation_panel(plt, correlations, models, output_dir / f"{name}-correlations.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DATA)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    render("vision", ["vgg19", "resnet50", "vit-b16", "dinov2-s14"], output)
    render("language", ["gpt2", "pythia-70m", "tinyllama"], output)
    print(f"Wrote Figure 4A-D panels to {output}")


if __name__ == "__main__":
    main()
