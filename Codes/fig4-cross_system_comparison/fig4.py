#!/usr/bin/env python3
"""Compute Figure 4E and compose Figure 4 from generated panel files."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from pypdf import PageObject, PdfReader, PdfWriter, Transformation
from scipy.stats import pearsonr


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Data" / "fig4"
KS = (2, 4, 8, 16, 32)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def odd_one_out_scores(
    activations: np.ndarray,
    similarity: np.ndarray,
    ks: Sequence[int],
) -> np.ndarray:
    """Return legacy-inclusive odd-one-out accuracy for each unit and k."""
    if activations.ndim != 2 or similarity.shape != (len(activations), len(activations)):
        raise ValueError("activation and similarity dimensions do not match")
    result = np.empty((activations.shape[1], len(ks)), dtype=np.float64)
    row_index = np.arange(len(activations))
    for unit in range(activations.shape[1]):
        order = np.lexsort((row_index, -activations[:, unit]))
        for k_index, k in enumerate(ks):
            values = np.asarray(similarity[np.ix_(order, order[:k])])
            threshold = float(values[:k].mean())
            result[unit, k_index] = np.mean(values[k:].mean(axis=1) < threshold)
    return result


def normalize_columns(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    centered = values - values.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=0, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("constant neural-response column encountered")
    return centered / (norms + 1e-9)


def directional_mmu(
    responses: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
) -> np.ndarray:
    source = normalize_columns(responses[:, source_mask])
    target = normalize_columns(responses[:, target_mask])
    return np.abs(source.T @ target).max(axis=1)


def plotting():
    cache = DATA / ".cache" / "matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_panel_e(
    responses: np.ndarray,
    mis: np.ndarray,
    animals: np.ndarray,
    output: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    plt = plotting()
    figure, axis = plt.subplots(figsize=(3.2, 2.8))
    colors = ("#4C78A8", "#F28E2B")
    statistics: list[dict[str, Any]] = []
    points: list[dict[str, Any]] = []
    for color, source, target in zip(colors, ("Chabo", "Tito"), ("Tito", "Chabo")):
        source_mask = animals == source
        mmu = directional_mmu(responses, source_mask, animals == target)
        x = mis[source_mask]
        correlation, p_value = pearsonr(x, mmu)
        statistics.append(
            {
                "source": source,
                "target": target,
                "n_source": int(source_mask.sum()),
                "n_candidates": int(np.sum(animals == target)),
                "r": float(correlation),
                "p": float(p_value),
            }
        )
        for neuron, mis_value, mmu_value in zip(np.flatnonzero(source_mask), x, mmu):
            points.append(
                {
                    "neuron_index": int(neuron),
                    "animal": source,
                    "target_animal": target,
                    "mis": f"{float(mis_value):.17g}",
                    "mmu": f"{float(mmu_value):.17g}",
                }
            )
        axis.scatter(x, mmu, s=15, alpha=0.6, color=color, edgecolors="none", label=f"{source} → {target}, r={correlation:.3f}")
        coefficients = np.polyfit(x, mmu, 1)
        line_x = np.linspace(float(x.min()), float(x.max()), 100)
        axis.plot(line_x, np.polyval(coefficients, line_x), "--", color=color, linewidth=1.4)
    axis.legend(loc="lower left", framealpha=0.85, fontsize=7)
    axis.grid(alpha=0.45)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_ylim(0, 0.6)
    axis.set_xlabel("Interpretability [MIS acc.]")
    axis.set_ylabel("Max matching unit [corr.]")
    figure.tight_layout()
    figure.savefig(output / "panel-e.pdf", bbox_inches="tight")
    figure.savefig(output / "panel-e.png", dpi=300, bbox_inches="tight")
    plt.close(figure)
    return statistics, points


def merge_scaled_page(target: PageObject, source_path: Path, x: float, y: float, width: float) -> None:
    source = PdfReader(str(source_path)).pages[0]
    scale = width / float(source.mediabox.width)
    target.merge_transformed_page(
        source,
        Transformation().scale(scale).translate(tx=x, ty=y),
        over=True,
        expand=False,
    )


def compose(output: Path, panels: Mapping[str, Path]) -> None:
    width, height = 497.0, 383.0
    page = PageObject.create_blank_page(width=width, height=height)
    placements = {
        "A": (0.0, 261.8, 484.8),
        "B": (60.7, 133.9, 363.5),
        "C": (4.6, 0.8, 168.7),
        "D": (181.2, 6.4, 153.4),
        "E": (353.5, 3.4, 138.5),
    }
    for label, (x, y, panel_width) in placements.items():
        merge_scaled_page(page, panels[label], x, y, panel_width)
    plt = plotting()
    with tempfile.TemporaryDirectory(prefix="figure4-labels-") as directory:
        overlay = Path(directory) / "labels.pdf"
        figure = plt.figure(figsize=(width / 72, height / 72))
        axis = figure.add_axes([0, 0, 1, 1])
        axis.set_xlim(0, width)
        axis.set_ylim(0, height)
        axis.axis("off")
        positions = {"A": (4, 366), "B": (67, 244), "C": (4, 113), "D": (184, 113), "E": (345, 113)}
        for label, (x, y) in positions.items():
            axis.text(x, y, label, fontsize=12, fontweight="bold", va="bottom")
        figure.savefig(overlay, format="pdf", transparent=True, bbox_inches=None, pad_inches=0)
        plt.close(figure)
        page.merge_page(PdfReader(str(overlay)).pages[0])
    writer = PdfWriter()
    writer.add_page(page)
    with output.open("wb") as stream:
        writer.write(stream)


def render_pdf(pdf: Path, png: Path) -> None:
    executable = shutil.which("pdftoppm")
    if executable is None:
        print("pdftoppm is unavailable; wrote fig4.pdf without a PNG preview")
        return
    with tempfile.TemporaryDirectory(prefix="figure4-render-") as directory:
        prefix = Path(directory) / "figure"
        subprocess.run([executable, "-png", "-singlefile", "-r", "240", str(pdf), str(prefix)], check=True)
        shutil.copyfile(prefix.with_suffix(".png"), png)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=DATA / "macaque" / "majaj2015-it.npz")
    parser.add_argument("--similarity", type=Path, default=DATA / "macaque" / "lpips-similarity.npy")
    parser.add_argument("--output-dir", type=Path, default=DATA)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    prepared = np.load(args.prepared)
    responses = np.asarray(prepared["responses"], dtype=np.float32)
    animals = np.char.strip(np.asarray(prepared["animals"]).astype(str).reshape(-1))
    counts = {label: int(np.sum(animals == label)) for label in np.unique(animals)}
    if responses.shape != (3200, 168) or counts != {"Chabo": 58, "Tito": 110}:
        raise ValueError(f"unexpected Majaj2015 IT data: {responses.shape}, {counts}")
    similarity = np.load(args.similarity, mmap_mode="r")
    scores = odd_one_out_scores(responses, similarity, KS)
    statistics, points = plot_panel_e(responses, scores.mean(axis=1), animals, output)
    write_csv(output / "macaque-points.csv", points)
    write_json(output / "macaque-summary.json", {"stimuli": 3200, "neurons": 168, "animals": counts, "ks": list(KS), "statistics": statistics})

    panels = {
        "A": output / "vision-scatter.pdf",
        "B": output / "language-scatter.pdf",
        "C": output / "vision-correlations.pdf",
        "D": output / "language-correlations.pdf",
        "E": output / "panel-e.pdf",
    }
    missing = [str(path) for path in panels.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Run panels.py before fig4.py; missing: " + ", ".join(missing))
    compose(output / "fig4.pdf", panels)
    render_pdf(output / "fig4.pdf", output / "fig4.png")
    print(json.dumps({"status": "pass", "statistics": statistics}, indent=2))


if __name__ == "__main__":
    main()
