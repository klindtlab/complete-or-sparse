#!/usr/bin/env python3
"""Generate submission-ready figures from the repository's audited data.

This file is the central entry point for paper figures.  Add each future
figure as a ``plot_*`` function and register it in ``PLOTTERS`` near the end of
the file.  Figure-specific training and data-audit code remains in its source
directory; this script owns only the final, paper-sized rendering layer.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
CODES_DIR = SCRIPT_PATH.parent
OUTPUT_DIR = PROJECT_ROOT / "Data" / "paper-figures"
FIG7_SOURCE = CODES_DIR / "fig7-weight_decay_reliability" / "fig7.py"
FIG7_BASELINE_ROOT = PROJECT_ROOT / "Data" / "fig7" / "baseline"
FIG7_RELIABILITY_ROOT = PROJECT_ROOT / "Data" / "fig7" / "reliability"
ABLATION_REPORT_ROOT = PROJECT_ROOT / "Data" / "fig8" / "report"
ABLATION_CONDITION_SUMMARY = ABLATION_REPORT_ROOT / "conditions.csv"
ABLATION_PAIRED_ROWS = ABLATION_REPORT_ROOT / "paired.csv"
ABLATION_PAIRED_SUMMARY = ABLATION_REPORT_ROOT / "paired-summary.csv"

LINEAR_COLOR = "#4C78A8"
RELU_COLOR = "#F58518"
SEED_TRACE_ALPHA = 0.17
FIG7_THRESHOLD = 0.90
FIG7_SEED_COUNT = 10
FIG7_WEIGHT_DECAYS = (0.0, 0.003, 0.01, 0.03)


def load_module(path: Path, name: str) -> ModuleType:
    """Load a repository script without requiring it to be a Python package."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_plotting() -> tuple[Any, Any]:
    """Import Matplotlib lazily and keep its cache below ``Data/``."""
    cache_dir = PROJECT_ROOT / "Data" / ".cache" / "paper_plots" / "matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / "Data" / ".cache" / "paper_plots"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    return plt, Line2D


def load_fig7_data(fig7: ModuleType) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Load and revalidate the complete revised data used by Figure 7."""
    baseline_candidates = fig7.load_revised_root(
        FIG7_BASELINE_ROOT, "revised_baseline"
    )
    reliability_candidates = fig7.load_revised_root(
        FIG7_RELIABILITY_ROOT, "revised_reliability"
    )
    reliability_runs = fig7.select_balanced_revised_runs(
        reliability_candidates, FIG7_WEIGHT_DECAYS
    )

    paired_seeds = {int(run["seed"]) for run in reliability_runs}
    expected_seeds = set(range(FIG7_SEED_COUNT))
    expected_run_count = FIG7_SEED_COUNT * len(FIG7_WEIGHT_DECAYS)
    if paired_seeds != expected_seeds or len(reliability_runs) != expected_run_count:
        raise ValueError(
            "Figure 7E requires a complete 10-seed by 4-weight-decay grid; "
            f"found {len(reliability_runs)} runs with seeds {sorted(paired_seeds)}."
        )

    baseline_runs = fig7.select_revised_baseline_runs(
        baseline_candidates, reliability_runs, FIG7_SEED_COUNT
    )
    runs_by_condition = {
        condition: sorted(
            [run for run in baseline_runs if run["condition"] == condition],
            key=lambda run: int(run["seed"]),
        )
        for condition in fig7.CONDITIONS
    }
    reliability_rows = [
        fig7.run_summary_row(run, FIG7_THRESHOLD) for run in reliability_runs
    ]
    if any(not row["full_budget"] for row in reliability_rows):
        raise ValueError("Figure 7E contains a run that did not complete its budget.")
    return runs_by_condition, reliability_rows


def stack_run_metric(
    runs: list[dict[str, Any]], metric: str
) -> tuple[np.ndarray, np.ndarray]:
    """Stack one metric after restricting all seeds to their common log steps."""
    ordered = sorted(runs, key=lambda run: int(run["seed"]))
    if not ordered:
        raise ValueError("No runs available for a Figure 7 condition.")
    common_steps = ordered[0]["steps"]
    for run in ordered[1:]:
        common_steps = np.intersect1d(common_steps, run["steps"])
    values = []
    for run in ordered:
        step_index = {int(step): idx for idx, step in enumerate(run["steps"])}
        values.append(
            np.asarray([run[metric][step_index[int(step)]] for step in common_steps])
        )
    return common_steps, np.stack(values)


def plot_training_curves(
    ax: Any,
    runs_by_condition: dict[str, list[dict[str, Any]]],
    conditions: tuple[str, str],
    metric: str,
    title: str,
    ylabel: str | None,
    *,
    show_legend: bool,
) -> None:
    for condition, color, label in zip(
        conditions, (LINEAR_COLOR, RELU_COLOR), ("Linear", "ReLU")
    ):
        steps, values = stack_run_metric(runs_by_condition[condition], metric)
        for trace in values:
            ax.plot(steps, trace, color=color, alpha=SEED_TRACE_ALPHA, linewidth=0.42)
        mean = values.mean(axis=0)
        # Use the sample standard deviation for the reported standard error.
        sem = values.std(axis=0, ddof=1) / math.sqrt(values.shape[0])
        ax.plot(steps, mean, color=color, linewidth=1.25, label=label)
        ax.fill_between(steps, mean - sem, mean + sem, color=color, alpha=0.18)

    ax.set_xscale("log")
    ax.set_xlim(
        left=max(
            1,
            min(run["steps"][0] for condition in conditions for run in runs_by_condition[condition]),
        )
    )
    ax.set_ylim(0, 1.03)
    ax.set_title(title)
    ax.set_xlabel("Training step")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25, linewidth=0.4)
    if show_legend:
        ax.legend(loc="lower right", frameon=False, handlelength=1.7)


def add_panel_label(ax: Any, label: str) -> None:
    ax.text(
        -0.21,
        1.09,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
    )


def draw_final_mcc_panel(
    ax: Any, reliability_rows: list[dict[str, Any]], threshold: float
) -> None:
    groups = {
        wd: [
            row
            for row in reliability_rows
            if math.isclose(float(row["weight_decay"]), wd, abs_tol=1e-12)
            and row["full_budget"]
        ]
        for wd in FIG7_WEIGHT_DECAYS
    }
    positions = np.arange(len(FIG7_WEIGHT_DECAYS))
    seed_maps = {
        wd: {int(row["seed"]): row for row in groups[wd]}
        for wd in FIG7_WEIGHT_DECAYS
    }
    common_seeds = sorted(
        set.intersection(*(set(seed_maps[wd]) for wd in FIG7_WEIGHT_DECAYS))
    )
    for seed in common_seeds:
        values = [seed_maps[wd][seed]["final_mcc"] for wd in FIG7_WEIGHT_DECAYS]
        ax.plot(positions, values, color="0.75", linewidth=0.45, zorder=1)

    all_values: list[float] = []
    for position, wd in zip(positions, FIG7_WEIGHT_DECAYS):
        group = groups[wd]
        values = np.asarray([row["final_mcc"] for row in group], dtype=float)
        all_values.extend(values.tolist())
        ax.scatter(
            np.full(len(values), position),
            values,
            s=13,
            color=RELU_COLOR,
            edgecolor="white",
            linewidth=0.3,
            zorder=3,
        )
        median = float(np.median(values))
        q25, q75 = np.percentile(values, [25, 75])
        ax.vlines(position, q25, q75, color="black", linewidth=1.45, zorder=4)
        ax.hlines(median, position - 0.14, position + 0.14, color="black", linewidth=1.3)
        successes = sum(bool(row["converged"]) for row in group)
        ax.text(
            position,
            1.008,
            f"{successes}/{len(group)}",
            ha="center",
            va="bottom",
            fontsize=5.5,
        )

    ax.axhline(threshold, color="0.35", linestyle="--", linewidth=0.6)
    ax.text(
        0.02,
        threshold + 0.006,
        "threshold",
        transform=ax.get_yaxis_transform(),
        color="0.35",
        fontsize=5.2,
        va="bottom",
    )
    ax.set_xticks(positions, [f"{wd:g}" for wd in FIG7_WEIGHT_DECAYS])
    ax.set_xlim(-0.36, len(FIG7_WEIGHT_DECAYS) - 1 + 0.36)
    ax.set_xlabel("Weight decay")
    ax.set_ylabel("Final MCC")
    # The weakest run has MCC 0.6896; keep it fully visible with padding.
    lower_limit = min(0.65, math.floor((min(all_values) - 0.02) * 20) / 20)
    ax.set_ylim(lower_limit, 1.035)
    ax.set_title("E  Optimization reliability", loc="left", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.4)


def draw_time_to_threshold_panel(
    ax: Any,
    reliability_rows: list[dict[str, Any]],
    threshold: float,
    line_2d: Any,
) -> None:
    rng = np.random.default_rng(0)
    budgets = sorted({float(row["expected_steps"]) for row in reliability_rows})
    if len(budgets) != 1:
        raise ValueError(f"Expected one shared optimization budget, found {budgets}")
    budget = budgets[0]
    ax.axhline(budget, color="0.55", linestyle=":", linewidth=0.55, zorder=0)
    ax.text(
        0.985,
        budget,
        "budget",
        transform=ax.get_yaxis_transform(),
        color="0.4",
        fontsize=5.2,
        ha="right",
        va="bottom",
    )

    for position, wd in enumerate(FIG7_WEIGHT_DECAYS):
        group = [
            row
            for row in reliability_rows
            if math.isclose(float(row["weight_decay"]), wd, abs_tol=1e-12)
        ]
        hits = [
            float(row["first_threshold_step"])
            for row in group
            if row["first_threshold_step"] is not None
        ]
        misses = [row for row in group if row["first_threshold_step"] is None]
        if hits:
            jitter = rng.uniform(-0.10, 0.10, size=len(hits))
            ax.scatter(
                position + jitter,
                hits,
                s=12,
                color=RELU_COLOR,
                edgecolor="white",
                linewidth=0.25,
                zorder=3,
            )
            q25, median, q75 = np.percentile(hits, [25, 50, 75])
            ax.vlines(position, q25, q75, color="black", linewidth=1.45, zorder=4)
            ax.hlines(median, position - 0.14, position + 0.14, color="black", linewidth=1.3)
        if misses:
            jitter = rng.uniform(-0.10, 0.10, size=len(misses))
            ax.scatter(
                position + jitter,
                [float(row["expected_steps"]) for row in misses],
                marker="x",
                s=15,
                color="#B33A3A",
                linewidth=0.9,
                zorder=4,
            )

    ax.set_yscale("log")
    ax.set_ylim(8e3, 1.35e6)
    ax.set_xticks(
        np.arange(len(FIG7_WEIGHT_DECAYS)),
        [f"{wd:g}" for wd in FIG7_WEIGHT_DECAYS],
    )
    ax.set_xlim(-0.36, len(FIG7_WEIGHT_DECAYS) - 1 + 0.36)
    ax.set_xlabel("Weight decay")
    ax.set_ylabel(f"Steps to MCC ≥ {threshold:.2f}")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.4)
    ax.legend(
        handles=[
            line_2d(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=RELU_COLOR,
                markeredgecolor="white",
                label="Converged",
            ),
            line_2d(
                [0],
                [0],
                marker="x",
                color="#B33A3A",
                linestyle="none",
                label="Not converged",
            ),
        ],
        loc="lower left",
        frameon=False,
        handlelength=1.2,
    )


def plot_fig_7(output_dir: Path) -> list[Path]:
    """Render the revised Figure 7 at the NeurIPS 5.5-inch text width."""
    fig7 = load_module(FIG7_SOURCE, "paper_plot_fig7_source")
    runs_by_condition, reliability_rows = load_fig7_data(fig7)
    plt, line_2d = load_plotting()

    plt.rcParams.update(
        {
            "font.size": 6.4,
            "axes.titlesize": 7.2,
            "axes.labelsize": 6.6,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "legend.fontsize": 5.5,
            "lines.linewidth": 0.8,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(5.5, 3.45), layout="constrained")
    grid = fig.add_gridspec(2, 3, width_ratios=(1.0, 1.0, 1.10))
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_e1 = fig.add_subplot(grid[0, 2])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])
    ax_e2 = fig.add_subplot(grid[1, 2])

    plot_training_curves(
        ax_a,
        runs_by_condition,
        ("hypersphere_linear", "hypersphere_relu"),
        "z_to_hz",
        "Hypersphere DGP",
        "Population disentanglement",
        show_legend=True,
    )
    plot_training_curves(
        ax_b,
        runs_by_condition,
        ("sparse_linear", "sparse_relu"),
        "z_to_hz",
        "Sparse DGP",
        None,
        show_legend=False,
    )
    plot_training_curves(
        ax_c,
        runs_by_condition,
        ("hypersphere_linear", "hypersphere_relu"),
        "mcc",
        "Hypersphere DGP",
        "Axis alignment (MCC)",
        show_legend=True,
    )
    plot_training_curves(
        ax_d,
        runs_by_condition,
        ("sparse_linear", "sparse_relu"),
        "mcc",
        "Sparse DGP",
        None,
        show_legend=False,
    )
    add_panel_label(ax_a, "A")
    add_panel_label(ax_b, "B")
    add_panel_label(ax_c, "C")
    add_panel_label(ax_d, "D")
    draw_final_mcc_panel(ax_e1, reliability_rows, FIG7_THRESHOLD)
    draw_time_to_threshold_panel(ax_e2, reliability_rows, FIG7_THRESHOLD, line_2d)

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "fig7.pdf"
    png_path = output_dir / "fig7.png"
    # Preserve the requested physical size exactly.  Constrained layout keeps
    # labels inside the canvas, so an additional tight-bounding-box rescale is
    # neither necessary nor desirable for a paper-width figure.
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    print(
        "Figure 7: 40 A--D condition-runs and 40 paired reliability runs; "
        "10 seeds per condition."
    )
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")
    return [pdf_path, png_path]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def load_ablation_condition_summary() -> dict[str, dict[str, str]]:
    """Load and validate the absolute-score summaries for the six ablations."""
    rows = read_csv_rows(ABLATION_CONDITION_SUMMARY)
    expected_ids = {
        "baseline",
        "entropy_sd_0p5",
        "entropy_sd_1",
        "entropy_sd_2",
        "entropy_affine_sd_0",
        "entropy_affine_sd_1",
        "entropy_affine_sd_2",
        "dependence_rho_0p1",
        "dependence_rho_0p3",
        "dependence_rho_0p5",
        "gamma_shape_2",
        "weibull_shape_1p5",
        "lognormal_matched",
        "signed_prob_0p1",
        "signed_prob_0p5",
        "activation_leaky_0p1",
        "activation_linear",
        "energy_wd_0",
        "energy_wd_0p01",
    }
    if len(rows) != 19 or {row["condition_id"] for row in rows} != expected_ids:
        raise ValueError(
            "Expected the baseline plus 18 ablation conditions in "
            f"{ABLATION_CONDITION_SUMMARY}."
        )
    if sorted(int(row["condition_order"]) for row in rows) != list(range(19)):
        raise ValueError("Ablation condition ordering is incomplete or duplicated.")
    for row in rows:
        if int(row["n_runs"]) != 10:
            raise ValueError(
                f"Expected 10 seeds for {row['condition_id']}, found {row['n_runs']}."
            )
        for metric in ("linear_h_to_z", "mcc"):
            mean = float(row[f"{metric}_mean"])
            sd = float(row[f"{metric}_sd"])
            sem = float(row[f"{metric}_sem"])
            if not (0.0 <= mean <= 1.0 and sd >= 0.0):
                raise ValueError(f"Invalid {metric} summary for {row['condition_id']}.")
            if not np.isclose(sem, sd / math.sqrt(10), atol=1e-12):
                raise ValueError(
                    f"SD/SEM mismatch for {row['condition_id']}: {metric}."
                )
    return {row["condition_id"]: row for row in rows}


def plot_ablation_summary(output_dir: Path) -> list[Path]:
    """Reproduce the original 2x3 ablation figure with mean +/- seed SD."""
    summary = load_ablation_condition_summary()
    plt, _ = load_plotting()

    panel_specs = (
        (
            "Class-dependent entropy",
            "Entropy SD across classes (nats)",
            (
                (
                    "bias-free",
                    ("baseline", "entropy_sd_0p5", "entropy_sd_1", "entropy_sd_2"),
                    (0.0, 0.5, 1.0, 2.0),
                    "o",
                    "-",
                ),
                (
                    "affine readout",
                    ("entropy_affine_sd_0", "entropy_affine_sd_1", "entropy_affine_sd_2"),
                    (0.0, 1.0, 2.0),
                    "s",
                    "--",
                ),
            ),
            None,
        ),
        (
            "Conditional dependence",
            "Gaussian-copula correlation",
            (("", ("baseline", "dependence_rho_0p1", "dependence_rho_0p3", "dependence_rho_0p5"), (0.0, 0.1, 0.3, 0.5), "o", "-"),),
            None,
        ),
        (
            "Non-exponential marginals",
            "Conditional distribution",
            (("", ("baseline", "gamma_shape_2", "weibull_shape_1p5", "lognormal_matched"), (0.0, 1.0, 2.0, 3.0), "o", "-"),),
            ("Exponential", r"Gamma $k=2$", r"Weibull $k=1.5$", "Log-normal"),
        ),
        (
            "Signed latents (boundary)",
            "Sign-flip probability",
            (("", ("baseline", "signed_prob_0p1", "signed_prob_0p5"), (0.0, 0.1, 0.5), "o", "-"),),
            None,
        ),
        (
            "Activation boundary",
            "Output activation",
            (("", ("baseline", "activation_leaky_0p1", "activation_linear"), (0.0, 1.0, 2.0), "o", "-"),),
            ("ReLU", "LeakyReLU\n0.1", "Linear"),
        ),
        (
            "Energy regularization",
            "AdamW weight decay",
            (("", ("energy_wd_0", "energy_wd_0p01", "baseline"), (0.0, 0.01, 0.03), "o", "-"),),
            None,
        ),
    )
    metric_styles = (
        ("linear_h_to_z", LINEAR_COLOR, "Linear recovery"),
        ("mcc", RELU_COLOR, "Permutation-scaling MCC"),
    )

    plt.rcParams.update(
        {
            "font.size": 6.2,
            "axes.titlesize": 7.0,
            "axes.labelsize": 6.4,
            "xtick.labelsize": 5.6,
            "ytick.labelsize": 5.6,
            "legend.fontsize": 4.9,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(5.5, 3.35),
        sharey=True,
        layout="constrained",
    )
    for panel_index, (axis, (title, xlabel, series, tick_labels)) in enumerate(
        zip(axes.flat, panel_specs)
    ):
        for series_name, condition_ids, x_values, marker, linestyle in series:
            x = np.asarray(x_values, dtype=float)
            for metric, color, metric_label in metric_styles:
                means = np.asarray(
                    [float(summary[condition_id][f"{metric}_mean"]) for condition_id in condition_ids]
                )
                standard_deviations = np.asarray(
                    [float(summary[condition_id][f"{metric}_sd"]) for condition_id in condition_ids]
                )
                label = metric_label + (f" ({series_name})" if series_name else "")
                axis.errorbar(
                    x,
                    means,
                    yerr=standard_deviations,
                    marker=marker,
                    linestyle=linestyle,
                    color=color,
                    linewidth=1.0,
                    markersize=3.0,
                    markeredgewidth=0.35,
                    markeredgecolor="white",
                    elinewidth=0.8,
                    capsize=1.8,
                    capthick=0.7,
                    label=label,
                )
        if tick_labels is not None:
            axis.set_xticks(np.asarray(series[0][2]), tick_labels, rotation=24, ha="right")
        axis.axhline(1.0, color="0.82", linewidth=0.55, zorder=0)
        axis.set_ylim(0.0, 1.04)
        axis.set_title(title, pad=2.2)
        axis.set_xlabel(xlabel, labelpad=2.0)
        axis.grid(axis="y", alpha=0.22, linewidth=0.4)
        axis.text(
            -0.18,
            1.09,
            chr(ord("A") + panel_index),
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=8.0,
            va="top",
        )
        if panel_index % 3 == 0:
            axis.set_ylabel("Final score")
        axis.legend(
            loc="lower left",
            frameon=False,
            handlelength=1.45,
            handletextpad=0.35,
            borderaxespad=0.25,
            labelspacing=0.25,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "fig8-compact.pdf"
    png_path = output_dir / "fig8-compact.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(
        "Ablation summary: baseline plus 18 conditions; absolute final scores "
        "shown as mean +/- SD across 10 seeds."
    )
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")
    return [pdf_path, png_path]


def load_ablation_paired_data() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Load and validate the paired ablation effects used by the forest plot."""
    paired_rows = read_csv_rows(ABLATION_PAIRED_ROWS)
    summary_rows = read_csv_rows(ABLATION_PAIRED_SUMMARY)
    if len(paired_rows) != 180 or len(summary_rows) != 18:
        raise ValueError(
            "Expected 180 seed-level paired effects and 18 condition summaries, "
            f"found {len(paired_rows)} and {len(summary_rows)}."
        )

    raw_by_condition: dict[str, list[dict[str, str]]] = {}
    for row in paired_rows:
        raw_by_condition.setdefault(row["condition_id"], []).append(row)
    summary_ids = {row["condition_id"] for row in summary_rows}
    if set(raw_by_condition) != summary_ids:
        raise ValueError("Seed-level and summary condition IDs do not match.")

    for summary in summary_rows:
        condition_id = summary["condition_id"]
        group = raw_by_condition[condition_id]
        seeds = {int(row["seed"]) for row in group}
        if len(group) != 10 or seeds != set(range(10)):
            raise ValueError(
                f"Expected paired seeds 0--9 for {condition_id}, found {sorted(seeds)}."
            )
        for raw_key, summary_prefix in (
            ("delta_linear_h_to_z", "delta_linear_h_to_z"),
            ("delta_mcc", "delta_mcc"),
        ):
            values = np.asarray([float(row[raw_key]) for row in group])
            expected_mean = float(summary[f"{summary_prefix}_mean"])
            expected_sem = float(summary[f"{summary_prefix}_sem"])
            observed_sem = float(values.std(ddof=1) / math.sqrt(len(values)))
            if not np.isclose(values.mean(), expected_mean, atol=1e-12):
                raise ValueError(f"Paired mean mismatch for {condition_id}: {raw_key}")
            if not np.isclose(observed_sem, expected_sem, atol=1e-12):
                raise ValueError(f"Paired SEM mismatch for {condition_id}: {raw_key}")
    return paired_rows, summary_rows


def ablation_display_label(row: dict[str, str], first_in_family: bool) -> str:
    family_prefix = {
        "entropy": "Entropy",
        "dependence": "Dependence",
        "distribution": "Marginal",
        "latent_nonnegativity": "Latent support",
        "activation_nonnegativity": "Activation",
        "energy": "Weight decay",
    }[row["family"]]
    label = row["label"]
    replacements = {
        "Entropy SD ": "SD ",
        "Affine, entropy SD ": "affine, SD ",
        "Copula rho ": r"copula $\rho=$",
        "Sign flips ": "sign flips ",
        "Weight decay ": "",
        "Linear": "linear output",
    }
    for source, target in replacements.items():
        if label.startswith(source):
            label = target + label[len(source):]
            break
    return f"{family_prefix}: {label}" if first_in_family else f"    {label}"


def plot_ablation_paired_effects(output_dir: Path) -> list[Path]:
    """Plot seed-paired ablation effects as a grouped, two-scale forest plot."""
    paired_rows, summary_rows = load_ablation_paired_data()
    plt, line_2d = load_plotting()

    family_order = (
        "entropy",
        "dependence",
        "distribution",
        "latent_nonnegativity",
        "activation_nonnegativity",
        "energy",
    )
    family_rank = {family: index for index, family in enumerate(family_order)}
    ordered_summaries = sorted(
        summary_rows,
        key=lambda row: (family_rank[row["family"]], summary_rows.index(row)),
    )
    raw_by_condition: dict[str, list[dict[str, str]]] = {}
    for row in paired_rows:
        raw_by_condition.setdefault(row["condition_id"], []).append(row)

    positions: list[float] = []
    labels: list[str] = []
    family_spans: list[tuple[str, float, float]] = []
    current_y = 0.0
    previous_family: str | None = None
    family_start = 0.0
    for row in ordered_summaries:
        family = row["family"]
        first_in_family = family != previous_family
        if first_in_family and previous_family is not None:
            family_spans.append((previous_family, family_start - 0.42, current_y - 0.58))
            current_y += 0.55
            family_start = current_y
        elif first_in_family:
            family_start = current_y
        positions.append(current_y)
        labels.append(ablation_display_label(row, first_in_family))
        current_y += 1.0
        previous_family = family
    if previous_family is not None:
        family_spans.append((previous_family, family_start - 0.42, current_y - 0.58))

    plt.rcParams.update(
        {
            "font.size": 7.0,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 6.3,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, (ax_full, ax_zoom) = plt.subplots(
        1,
        2,
        figsize=(5.5, 6.15),
        sharey=True,
        layout="constrained",
        gridspec_kw={"width_ratios": (1.08, 1.0)},
    )

    metric_styles = (
        (
            "delta_linear_h_to_z",
            "delta_linear_h_to_z_mean",
            "delta_linear_h_to_z_sem",
            LINEAR_COLOR,
            "o",
            0.13,
            r"Linear recovery ($h(z)\to z$)",
        ),
        (
            "delta_mcc",
            "delta_mcc_mean",
            "delta_mcc_sem",
            RELU_COLOR,
            "s",
            -0.13,
            "Axis alignment (MCC)",
        ),
    )
    seed_offsets = np.linspace(-0.045, 0.045, 10)
    for axis in (ax_full, ax_zoom):
        for span_index, (_, lower, upper) in enumerate(family_spans):
            if span_index % 2 == 0:
                axis.axhspan(lower, upper, color="0.965", zorder=0)
        axis.axvline(0.0, color="0.35", linestyle="--", linewidth=0.75, zorder=1)
        for y, summary in zip(positions, ordered_summaries):
            condition_rows = sorted(
                raw_by_condition[summary["condition_id"]],
                key=lambda row: int(row["seed"]),
            )
            for raw_key, mean_key, sem_key, color, marker, offset, _ in metric_styles:
                raw_values = np.asarray(
                    [float(row[raw_key]) for row in condition_rows], dtype=float
                )
                axis.scatter(
                    raw_values,
                    y + offset + seed_offsets,
                    s=7,
                    color=color,
                    alpha=0.20,
                    linewidth=0,
                    zorder=2,
                    clip_on=True,
                )
                axis.errorbar(
                    float(summary[mean_key]),
                    y + offset,
                    xerr=float(summary[sem_key]),
                    fmt=marker,
                    color=color,
                    markeredgecolor="white",
                    markeredgewidth=0.35,
                    markersize=4.0,
                    elinewidth=1.0,
                    capsize=2.0,
                    zorder=4,
                )
        axis.grid(True, axis="x", alpha=0.20, linewidth=0.45)
        axis.set_ylim(positions[-1] + 0.6, -0.65)
        axis.tick_params(axis="y", length=0)

    ax_full.set_xlim(-1.0, 0.055)
    ax_full.set_xticks((-1.0, -0.75, -0.5, -0.25, 0.0))
    ax_full.set_title("A  Full effect range", loc="left", fontweight="bold")
    ax_full.set_yticks(positions, labels)

    ax_zoom.set_xlim(-0.105, 0.012)
    ax_zoom.set_xticks((-0.10, -0.075, -0.05, -0.025, 0.0))
    ax_zoom.set_title("B  Near-baseline detail", loc="left", fontweight="bold")
    ax_zoom.tick_params(labelleft=False)
    ax_zoom.text(
        0.02,
        0.992,
        r"Effects $< -0.105$ omitted",
        transform=ax_zoom.transAxes,
        fontsize=5.7,
        color="0.35",
        ha="left",
        va="top",
    )

    fig.supxlabel("Paired change from baseline (ablation - same-seed baseline)")
    legend_handles = [
        line_2d(
            [0],
            [0],
            marker=marker,
            linestyle="none",
            markerfacecolor=color,
            markeredgecolor="white",
            label=label,
        )
        for _, _, _, color, marker, _, label in metric_styles
    ]
    fig.legend(
        handles=legend_handles,
        loc="outside upper center",
        ncol=2,
        frameon=False,
        columnspacing=1.6,
        handletextpad=0.4,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "fig8-paired.pdf"
    png_path = output_dir / "fig8-paired.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(
        "Ablation paired effects: 18 conditions x 10 paired seeds; "
        "summary means and SEMs verified against seed-level deltas."
    )
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")
    return [pdf_path, png_path]


PLOTTERS: dict[str, Callable[[Path], list[Path]]] = {
    "fig7": plot_fig_7,
    "fig8-compact": plot_ablation_summary,
    "fig8-paired": plot_ablation_paired_effects,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "figures",
        nargs="*",
        choices=sorted(PLOTTERS),
        default=None,
        help="Figures to generate (default: all registered figures).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR}).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    figure_names = args.figures or sorted(PLOTTERS)
    for figure_name in figure_names:
        PLOTTERS[figure_name](args.output_dir.resolve())


if __name__ == "__main__":
    main()
