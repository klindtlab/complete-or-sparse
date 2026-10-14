#!/usr/bin/env python3
"""Generate the capacity-simulation tables used by Figures 2 and 9."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "Data" / "fig2"


def comma_list(value: str, cast) -> list:
    return [cast(item) for item in value.split(",") if item]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_latents(distribution: str, count: int, dimension: int, device: torch.device) -> torch.Tensor:
    if distribution == "exponential":
        return torch.distributions.Exponential(1.0).sample((count, dimension)).to(device)
    if distribution == "gaussian":
        return torch.randn(count, dimension, device=device)
    if distribution == "hypersphere":
        values = torch.randn(count, dimension, device=device)
        return values / values.norm(dim=1, keepdim=True).clamp_min(1e-12)
    raise ValueError(f"Unknown latent distribution: {distribution}")


def build_teacher(
    classes: int,
    dimension: int,
    effective_dimension: int,
    logit_scale: float,
    device: torch.device,
) -> torch.Tensor:
    """Return a rank-``effective_dimension`` teacher matrix."""
    class_basis, _ = torch.linalg.qr(
        torch.randn(classes, effective_dimension, device=device), mode="reduced"
    )
    latent_basis, _ = torch.linalg.qr(
        torch.randn(dimension, effective_dimension, device=device), mode="reduced"
    )
    return (logit_scale * class_basis @ latent_basis.T).detach()


def make_sampler(teacher: torch.Tensor, distribution: str, device: torch.device):
    dimension = teacher.shape[1]

    def sample(count: int) -> tuple[torch.Tensor, torch.Tensor]:
        latent = sample_latents(distribution, count, dimension, device)
        probabilities = F.softmax(latent @ teacher.T, dim=1)
        return latent, probabilities

    return sample


def make_encoder(
    kind: str,
    input_dimension: int,
    output_dimension: int,
    hidden_dimension: int,
    depth: int,
    device: torch.device,
) -> nn.Module:
    if kind == "linear":
        return nn.Linear(input_dimension, output_dimension, bias=False).to(device)
    layers: list[nn.Module] = [nn.Linear(input_dimension, hidden_dimension), nn.ReLU()]
    for _ in range(depth - 1):
        layers.extend((nn.Linear(hidden_dimension, hidden_dimension), nn.ReLU()))
    layers.append(nn.Linear(hidden_dimension, output_dimension))
    return nn.Sequential(*layers).to(device)


def soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


@torch.no_grad()
def teacher_entropy(sample, count: int, batch_size: int) -> float:
    total = 0.0
    for start in range(0, count, batch_size):
        current = min(batch_size, count - start)
        _, probabilities = sample(current)
        total -= (
            probabilities * torch.log(probabilities.clamp_min(1e-12))
        ).sum().item()
    return total / count


@torch.no_grad()
def evaluate(
    encoder: nn.Module,
    head: nn.Module,
    sample,
    count: int,
    batch_size: int,
) -> tuple[float, float]:
    encoder.eval()
    head.eval()
    total_ce = 0.0
    total_kl = 0.0
    for start in range(0, count, batch_size):
        current = min(batch_size, count - start)
        latent, target = sample(current)
        log_prediction = F.log_softmax(head(encoder(latent)), dim=1)
        log_target = torch.log(target.clamp_min(1e-12))
        total_ce -= (target * log_prediction).sum().item()
        total_kl += (target * (log_target - log_prediction)).sum().item()
    return total_ce / count, total_kl / count


def train(
    encoder: nn.Module,
    sample,
    classes: int,
    output_dimension: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    log_every: int,
    device: torch.device,
) -> tuple[list[dict[str, float]], nn.Linear]:
    head = nn.Linear(output_dimension, classes, bias=False).to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()), lr=learning_rate
    )
    curve = []
    for step in range(1, steps + 1):
        latent, target = sample(batch_size)
        logits = head(encoder(latent))
        loss = soft_cross_entropy(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 1 or step == steps or step % log_every == 0:
            with torch.no_grad():
                log_target = torch.log(target.clamp_min(1e-12))
                log_prediction = F.log_softmax(logits, dim=1)
                kl = (target * (log_target - log_prediction)).sum(dim=1).mean()
            curve.append(
                {"step": step, "ce_loss": float(loss.item()), "kl_loss": float(kl.item())}
            )
    return curve, head


def effective_rank(matrix: np.ndarray) -> float:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    probabilities = singular_values / singular_values.sum()
    return float(np.exp(-(probabilities * np.log(probabilities + 1e-12)).sum()))


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seeds = comma_list(args.seeds, int)
    dimensions = comma_list(args.dimensions, int)
    distributions = comma_list(args.distributions, str)
    scales = comma_list(args.teacher_scales, float)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    curves: list[dict] = []
    summaries: list[dict] = []
    for distribution_index, distribution in enumerate(distributions):
        for scale in scales:
            for seed in seeds:
                set_seed(42 + seed)
                teacher = build_teacher(
                    args.classes,
                    args.latent_dimension,
                    args.effective_dimension,
                    scale,
                    device,
                )
                sample = make_sampler(teacher, distribution, device)
                entropy = teacher_entropy(sample, args.entropy_samples, args.batch_size)
                for dimension in dimensions:
                    for encoder_kind in ("linear", "mlp"):
                        model_seed = (
                            100_000 * seed
                            + 1_000 * int(scale * 10)
                            + 100 * distribution_index
                            + 10 * dimension
                            + int(encoder_kind == "mlp")
                        )
                        set_seed(model_seed)
                        encoder = make_encoder(
                            encoder_kind,
                            args.latent_dimension,
                            dimension,
                            args.mlp_hidden,
                            args.mlp_depth,
                            device,
                        )
                        curve, head = train(
                            encoder,
                            sample,
                            args.classes,
                            dimension,
                            args.steps,
                            args.batch_size,
                            args.learning_rate,
                            args.log_every,
                            device,
                        )
                        final_ce, final_kl = evaluate(
                            encoder,
                            head,
                            sample,
                            args.final_samples,
                            args.batch_size,
                        )
                        identity = {
                            "distribution": distribution,
                            "teacher_logit_scale": scale,
                            "seed": seed,
                            "n": args.latent_dimension,
                            "n_prime": args.effective_dimension,
                            "m": dimension,
                            "enc_type": encoder_kind,
                        }
                        curves.extend({**identity, **row} for row in curve)
                        summaries.append(
                            {
                                **identity,
                                "teacher_entropy": entropy,
                                "final_ce": final_ce,
                                "final_kl": final_kl,
                                "effective_rank_U": effective_rank(
                                    head.weight.detach().cpu().numpy()
                                ),
                            }
                        )
                        print(
                            f"{distribution} scale={scale:g} seed={seed} m={dimension} "
                            f"{encoder_kind}: KL={final_kl:.6f}"
                        )

    pd.DataFrame(curves).to_csv(output / "curves.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
    print(f"Saved Figure 2/9 tables to {output}")


def build_parser() -> argparse.ArgumentParser:
    quick = os.environ.get("COMPLETE_OR_SPARSE_QUICK") == "1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", help="PyTorch device; defaults to CUDA when available.")
    parser.add_argument("--seeds", default="0" if quick else "0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--dimensions", default="2,8" if quick else "2,4,6,8,10")
    parser.add_argument("--distributions", default="exponential,gaussian,hypersphere")
    parser.add_argument("--teacher-scales", default="10" if quick else "5,8,10")
    parser.add_argument("--latent-dimension", type=int, default=8)
    parser.add_argument("--effective-dimension", type=int, default=8)
    parser.add_argument("--classes", type=int, default=64)
    parser.add_argument("--steps", type=int, default=10 if quick else 12_000)
    parser.add_argument("--batch-size", type=int, default=64 if quick else 512)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--log-every", type=int, default=1 if quick else 50)
    parser.add_argument("--entropy-samples", type=int, default=256 if quick else 10_000)
    parser.add_argument("--final-samples", type=int, default=256 if quick else 10_000)
    parser.add_argument("--mlp-hidden", type=int, default=16 if quick else 128)
    parser.add_argument("--mlp-depth", type=int, default=1 if quick else 3)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
