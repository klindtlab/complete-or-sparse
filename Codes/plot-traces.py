#!/usr/bin/env python3
"""Render Figures 1, 5, and 6 from traces generated under ``Data/``."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data"


def plotting() -> Any:
    cache = DATA / ".cache" / "matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def load_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    result = {
        key: np.asarray([float(row[key]) for row in rows])
        for key in rows[0]
        if key in {"step", "z_to_hz", "z2z r2", "mcc"}
    }
    if "z_to_hz" not in result and "z2z r2" in result:
        result["z_to_hz"] = result["z2z r2"]
    return result


def parse_condition(directory: Path, figure: str) -> tuple[str, str]:
    args_path = directory / "args.json"
    if args_path.exists():
        args = json.loads(args_path.read_text(encoding="utf-8"))
        expon = int(args["expon"])
        activation = int(args["relu_enc"] if figure != "fig5" else args["expon_enc"])
    else:
        name = directory.name
        expon = int(name.split("expon_", 1)[1].split("_", 1)[0])
        marker = "enc_" if figure == "fig5" else "relu_"
        activation = int(name.split(marker, 1)[1].split("_", 1)[0])
    distribution = "sparse" if expon else "hypersphere"
    representation = ("exponential" if figure == "fig5" else "relu") if activation else "linear"
    return distribution, representation


def load_runs(root: Path, figure: str) -> dict[tuple[str, str], list[tuple[Path, dict[str, np.ndarray]]]]:
    grouped: dict[tuple[str, str], list[tuple[Path, dict[str, np.ndarray]]]] = defaultdict(list)
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        logs = [path for path in directory.glob("*.csv") if path.name != "corr.csv"]
        if not logs:
            continue
        grouped[parse_condition(directory, figure)].append((directory, load_csv(logs[0])))
    return grouped


def aligned_metric(runs: list[tuple[Path, dict[str, np.ndarray]]], metric: str) -> tuple[np.ndarray, np.ndarray]:
    common = runs[0][1]["step"]
    for _, run in runs[1:]:
        common = np.intersect1d(common, run["step"])
    values = []
    for _, run in runs:
        lookup = {int(step): value for step, value in zip(run["step"], run[metric])}
        values.append([lookup[int(step)] for step in common])
    return common, np.asarray(values)


def curve(axis: Any, runs: list[tuple[Path, dict[str, np.ndarray]]], metric: str, label: str) -> None:
    steps, values = aligned_metric(runs, metric)
    mean = values.mean(axis=0)
    sem = values.std(axis=0, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else np.zeros_like(mean)
    axis.plot(steps, mean, label=label)
    axis.fill_between(steps, mean - sem, mean + sem, alpha=0.2)


def final_correlation(run_dir: Path) -> np.ndarray:
    with (run_dir / "corr.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    values = np.asarray([float(value) for key, value in rows[-1].items() if key != "step"])
    side = int(round(np.sqrt(values.size)))
    return values.reshape(side, side)


def render_grid(figure: str, output: Path) -> None:
    plt = plotting()
    grouped = load_runs(DATA / figure / "runs", figure)
    activations = ("linear", "exponential" if figure == "fig5" else "relu")
    if figure == "fig1":
        fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.8))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
    for column, distribution in enumerate(("hypersphere", "sparse")):
        for representation in activations:
            runs = grouped[(distribution, representation)]
            curve(axes[0, column], runs, "z_to_hz", representation.title())
            curve(axes[1, column], runs, "mcc", representation.title())
        axes[0, column].set_title(f"{distribution.title()} DGP")
        axes[0, column].set_ylabel("Population disentanglement")
        axes[1, column].set_ylabel("Axis alignment (MCC)")
        for axis in axes[:, column]:
            axis.set_xscale("log")
            axis.set_ylim(0, 1.05)
            axis.set_xlabel("Training step")
            axis.grid(alpha=0.25)
            axis.legend(frameon=False)
    if figure == "fig1":
        for row, representation in enumerate(activations):
            candidates = [
                directory
                for directory, _ in grouped[("sparse", representation)]
                if (directory / "corr.csv").exists()
            ]
            if not candidates:
                raise FileNotFoundError(f"No correlation trace for sparse {representation}")
            matrix = final_correlation(candidates[0])
            image = axes[row, 2].imshow(np.abs(matrix), vmin=0, vmax=1, cmap="viridis")
            axes[row, 2].set_title(f"Sparse {representation}")
            axes[row, 2].set_xlabel("Learned component")
            axes[row, 2].set_ylabel("True component")
        fig.colorbar(image, ax=axes[:, 2], label="|correlation|", fraction=0.05)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def load_tensor(path: Path) -> np.ndarray:
    import torch

    return torch.load(path, map_location="cpu", weights_only=True).numpy()


def render_fig6(output: Path) -> None:
    plt = plotting()
    relu = load_runs(DATA / "fig1" / "runs", "fig1")[("sparse", "relu")]
    exponential = load_runs(DATA / "fig5" / "runs", "fig5")[("sparse", "exponential")]
    weight_decay = load_runs(DATA / "fig6" / "weight-decay", "fig1")[("sparse", "relu")]
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.8))
    cmap = plt.get_cmap("viridis")
    normalize = plt.Normalize(0.75, 1.0)
    for axis, runs, title in zip(axes[[0, 1, 3]], (relu, exponential, weight_decay), ("ReLU", "BN + Exp", "ReLU + weight decay")):
        for _, run in runs:
            axis.plot(run["step"], run["mcc"], color=cmap(normalize(run["mcc"][-1])))
        axis.set_xscale("log")
        axis.set_ylim(0, 1)
        axis.set_title(title)
        axis.set_xlabel("Training step")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Axis alignment (MCC)")
    spectrum_runs = [(directory, run) for directory, run in relu if (directory / "hz.pt").exists()]
    if not spectrum_runs:
        raise FileNotFoundError("No hz.pt activation traces are available for Figure 6")
    for directory, run in spectrum_runs:
        values = load_tensor(directory / "hz.pt")
        singular = np.linalg.svd(values, compute_uv=False)
        variance = singular**2 / np.sum(singular**2)
        axes[2].plot(variance, ".-", color=cmap(normalize(run["mcc"][-1])))
    axes[2].set_yscale("log")
    axes[2].set_title("ReLU spectra")
    axes[2].set_xlabel("Component")
    axes[2].set_ylabel("Variance explained")
    axes[2].grid(alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figure", choices=("fig1", "fig5", "fig6"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or DATA / args.figure / f"{args.figure}.pdf"
    if args.figure == "fig6":
        render_fig6(output)
    else:
        render_grid(args.figure, output)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
