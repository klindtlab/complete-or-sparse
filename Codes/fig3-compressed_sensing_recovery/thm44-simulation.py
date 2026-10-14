#!/usr/bin/env python3
"""Generate the compressed-sensing tables used by Figure 3."""

from __future__ import annotations

import argparse
import math
import os
import random
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import DictionaryLearning
from sklearn.linear_model import Lasso, LinearRegression


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "Data" / "fig3"


def comma_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_sparse_latents(
    count: int, dimension: int, sparsity: int, device: torch.device
) -> torch.Tensor:
    latent = torch.zeros(count, dimension, device=device)
    support = torch.rand(count, dimension, device=device).topk(sparsity, dim=1).indices
    values = torch.distributions.Exponential(1.0).sample((count, sparsity)).to(device)
    latent.scatter_(1, support, values)
    return latent


def build_teacher(
    classes: int,
    dimension: int,
    effective_dimension: int,
    logit_scale: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    measurement = torch.randn(effective_dimension, dimension, device=device)
    measurement /= measurement.norm(dim=0, keepdim=True).clamp_min(1e-12)
    readout = logit_scale * torch.randn(classes, effective_dimension, device=device)
    return (readout @ measurement).detach(), measurement.detach()


def make_sampler(
    teacher: torch.Tensor,
    dimension: int,
    sparsity: int,
    device: torch.device,
):
    def sample(count: int) -> tuple[torch.Tensor, torch.Tensor]:
        latent = sample_sparse_latents(count, dimension, sparsity, device)
        return latent, F.softmax(latent @ teacher.T, dim=1)

    return sample


def check_column_independence(
    measurement: torch.Tensor, sparsity: int, tolerance: float
) -> dict[str, float | int | bool]:
    """Exhaustively verify that every set of ``2k`` columns is independent."""
    matrix = measurement.detach().cpu().numpy()
    subset_size = min(2 * sparsity, matrix.shape[1])
    minimum = np.inf
    failures = 0
    for subset in combinations(range(matrix.shape[1]), subset_size):
        block = matrix[:, subset]
        singular_values = np.linalg.svd(block, compute_uv=False)
        smallest = singular_values[-1] if block.shape[1] <= block.shape[0] else 0.0
        minimum = min(minimum, smallest)
        failures += int(smallest <= tolerance)
    return {
        "nsp_holds": failures == 0,
        "n_subsets": math.comb(matrix.shape[1], subset_size),
        "n_failures": failures,
        "min_singular_val": float(minimum),
        "subset_size": subset_size,
    }


def soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


@torch.no_grad()
def teacher_entropy(sample, count: int, batch_size: int) -> float:
    total = 0.0
    for start in range(0, count, batch_size):
        current = min(batch_size, count - start)
        _, target = sample(current)
        total -= (target * torch.log(target.clamp_min(1e-12))).sum().item()
    return total / count


@torch.no_grad()
def prediction_kl(encoder: nn.Module, head: nn.Module, sample, count: int) -> float:
    latent, target = sample(count)
    log_prediction = F.log_softmax(head(encoder(latent)), dim=1)
    log_target = torch.log(target.clamp_min(1e-12))
    return float((target * (log_target - log_prediction)).sum(dim=1).mean().item())


def row_space_match(matrix: torch.Tensor, reference: torch.Tensor) -> float:
    first, _ = np.linalg.qr(matrix.detach().cpu().numpy().T)
    second, _ = np.linalg.qr(reference.detach().cpu().numpy().T)
    cosines = np.linalg.svd(first.T @ second, compute_uv=False)
    return float(np.clip(cosines, -1.0, 1.0).min())


def train_encoder(
    sample,
    reference: torch.Tensor,
    dimension: int,
    effective_dimension: int,
    classes: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    log_every: int,
    kl_samples: int,
    device: torch.device,
) -> tuple[list[dict[str, float]], nn.Linear]:
    encoder = nn.Linear(dimension, effective_dimension, bias=False).to(device)
    head = nn.Linear(effective_dimension, classes, bias=False).to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()), lr=learning_rate
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    curve = []
    for step in range(1, steps + 1):
        latent, target = sample(batch_size)
        loss = soft_cross_entropy(head(encoder(latent)), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step == 1 or step == steps or step % log_every == 0:
            curve.append(
                {
                    "step": step,
                    "ce_loss": float(loss.item()),
                    "kl_loss": prediction_kl(encoder, head, sample, kl_samples),
                    "row_space_match": row_space_match(encoder.weight, reference),
                }
            )
    return curve, encoder


def as_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def linear_recovery(train_measurements, train_latents, test_measurements) -> np.ndarray:
    model = LinearRegression(fit_intercept=True).fit(
        as_numpy(train_measurements), as_numpy(train_latents)
    )
    return model.predict(as_numpy(test_measurements))


def lasso_recovery(
    measurement: torch.Tensor, observations: torch.Tensor, alpha: float
) -> np.ndarray:
    matrix = as_numpy(measurement)
    observed = as_numpy(observations)
    recovered = np.zeros((len(observed), matrix.shape[1]))
    for index, target in enumerate(observed):
        model = Lasso(alpha=alpha, fit_intercept=False, positive=True, max_iter=1_000)
        model.fit(matrix, target)
        recovered[index] = model.coef_
    return recovered


def direct_r2(estimate: np.ndarray, target: np.ndarray) -> float:
    residual = np.square(target - estimate).sum()
    total = np.square(target - target.mean(axis=0, keepdims=True)).sum()
    return float(1.0 - residual / max(total, 1e-12))


def mean_correlation(estimate: np.ndarray, target: np.ndarray) -> float:
    correlations = []
    for column in range(target.shape[1]):
        first, second = estimate[:, column], target[:, column]
        correlations.append(
            0.0 if min(first.std(), second.std()) < 1e-12 else np.corrcoef(first, second)[0, 1]
        )
    return float(np.mean(correlations))


def matched_correlation(estimate: np.ndarray, target: np.ndarray) -> float:
    correlations = np.zeros((target.shape[1], target.shape[1]))
    for left in range(target.shape[1]):
        for right in range(target.shape[1]):
            first, second = estimate[:, left], target[:, right]
            if min(first.std(), second.std()) >= 1e-12:
                correlations[left, right] = abs(np.corrcoef(first, second)[0, 1])
    rows, columns = linear_sum_assignment(-correlations)
    return float(correlations[rows, columns].mean())


def support_rate(estimate: np.ndarray, target: np.ndarray, sparsity: int) -> float:
    correct = 0
    for estimated_row, target_row in zip(estimate, target):
        predicted = set(np.argsort(-np.abs(estimated_row))[:sparsity])
        actual = set(np.argsort(-np.abs(target_row))[:sparsity])
        if predicted == actual and np.all(np.abs(estimated_row[list(predicted)]) > 1e-3):
            correct += 1
    return correct / len(target)


def recovery_metrics(
    learned: torch.Tensor,
    reference: torch.Tensor,
    sample_latent,
    train_count: int,
    test_count: int,
    sparsity: int,
    alpha: float,
) -> dict[str, float]:
    train_latent = sample_latent(train_count)
    test_latent = sample_latent(test_count)
    target = as_numpy(test_latent)
    output: dict[str, float] = {}
    for name, matrix in (("Astar_z", reference), ("A_z", learned)):
        train_measurements = train_latent @ matrix.T
        test_measurements = test_latent @ matrix.T
        estimates = {
            "linear_OLS": linear_recovery(train_measurements, train_latent, test_measurements),
            "Lasso": lasso_recovery(matrix, test_measurements, alpha),
        }
        for method, estimate in estimates.items():
            prefix = f"{name}__{method}"
            output[f"{prefix}__r2"] = direct_r2(estimate, target)
            output[f"{prefix}__mean_corr"] = mean_correlation(estimate, target)
            output[f"{prefix}__mcc"] = matched_correlation(estimate, target)
            output[f"{prefix}__supp_rate"] = support_rate(estimate, target, sparsity)
    return output


def unsupervised_recovery(
    learned: torch.Tensor,
    sample_latent,
    train_count: int,
    test_count: int,
    alpha: float,
    iterations: int,
) -> dict[str, float]:
    train_latent = sample_latent(train_count)
    test_latent = sample_latent(test_count)
    matrix = as_numpy(learned)
    train_representation = as_numpy(train_latent) @ matrix.T
    test_representation = as_numpy(test_latent) @ matrix.T
    dictionary = DictionaryLearning(
        n_components=matrix.shape[1],
        alpha=alpha,
        fit_algorithm="cd",
        transform_algorithm="lasso_cd",
        positive_code=True,
        max_iter=iterations,
        random_state=0,
    )
    dictionary.fit(train_representation)
    estimate = dictionary.transform(test_representation)
    target = as_numpy(test_latent)
    return {
        "unsup_lasso__r2": direct_r2(estimate, target),
        "unsup_lasso__mcc": matched_correlation(estimate, target),
    }


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    curves: list[dict] = []
    summaries: list[dict] = []
    rank_checks: list[dict] = []

    for sparsity in comma_ints(args.k_values):
        for seed in comma_ints(args.seeds):
            set_seed(42 + seed)
            teacher, reference = build_teacher(
                args.classes,
                args.latent_dimension,
                args.effective_dimension,
                args.teacher_scale,
                device,
            )
            rank_check = check_column_independence(reference, sparsity, args.rank_tolerance)
            rank_checks.append({"k": sparsity, "seed": seed, **rank_check})
            sample = make_sampler(teacher, args.latent_dimension, sparsity, device)
            entropy = teacher_entropy(sample, args.entropy_samples, args.batch_size)

            set_seed(1_000 * seed + sparsity)
            curve, encoder = train_encoder(
                sample,
                reference,
                args.latent_dimension,
                args.effective_dimension,
                args.classes,
                args.steps,
                args.batch_size,
                args.learning_rate,
                args.log_every,
                args.kl_samples,
                device,
            )
            identity = {"k": sparsity, "seed": seed}
            curves.extend({**identity, "teacher_entropy": entropy, **row} for row in curve)
            sample_latent = lambda count, k=sparsity: sample_sparse_latents(
                count, args.latent_dimension, k, device
            )
            recovery = recovery_metrics(
                encoder.weight.detach(),
                reference,
                sample_latent,
                args.recovery_train,
                args.recovery_test,
                sparsity,
                args.lasso_alpha,
            )
            unsupervised = unsupervised_recovery(
                encoder.weight.detach(),
                sample_latent,
                args.recovery_train,
                args.recovery_test,
                args.lasso_alpha,
                args.dictionary_iterations,
            )
            summaries.append(
                {
                    **identity,
                    "n": args.latent_dimension,
                    "n_prime": args.effective_dimension,
                    "m": args.effective_dimension,
                    "C": args.classes,
                    "teacher_entropy": entropy,
                    "final_ce": curve[-1]["ce_loss"],
                    "final_kl": curve[-1]["kl_loss"],
                    "final_row_space_match": curve[-1]["row_space_match"],
                    "nsp_holds": rank_check["nsp_holds"],
                    "nsp_min_sigma": rank_check["min_singular_val"],
                    "z_type": "sparse",
                    **recovery,
                    **unsupervised,
                }
            )
            print(
                f"k={sparsity} seed={seed}: KL={curve[-1]['kl_loss']:.6f}, "
                f"row-space={curve[-1]['row_space_match']:.6f}"
            )

    pd.DataFrame(curves).to_csv(output / "curves.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
    pd.DataFrame(rank_checks).to_csv(output / "rank-check.csv", index=False)
    print(f"Saved Figure 3 tables to {output}")


def build_parser() -> argparse.ArgumentParser:
    quick = os.environ.get("COMPLETE_OR_SPARSE_QUICK") == "1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", help="PyTorch device; defaults to CUDA when available.")
    parser.add_argument("--seeds", default="0" if quick else "0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--k-values", default="1,2,3")
    parser.add_argument("--latent-dimension", type=int, default=16)
    parser.add_argument("--effective-dimension", type=int, default=8)
    parser.add_argument("--classes", type=int, default=64)
    parser.add_argument("--teacher-scale", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=10 if quick else 20_000)
    parser.add_argument("--batch-size", type=int, default=64 if quick else 512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=1 if quick else 250)
    parser.add_argument("--entropy-samples", type=int, default=256 if quick else 10_000)
    parser.add_argument("--kl-samples", type=int, default=128 if quick else 2_048)
    parser.add_argument("--recovery-train", type=int, default=256 if quick else 5_000)
    parser.add_argument("--recovery-test", type=int, default=64 if quick else 1_000)
    parser.add_argument("--lasso-alpha", type=float, default=0.005)
    parser.add_argument("--dictionary-iterations", type=int, default=10 if quick else 500)
    parser.add_argument("--rank-tolerance", type=float, default=1e-6)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
