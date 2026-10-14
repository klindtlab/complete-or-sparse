#!/usr/bin/env python3
"""Assumption ablations for the sparse generative model.

The workflow has three stages:

1. ``prepare-dgp`` creates one immutable set of class scales and one fixed
   invertible mixing network shared by every condition.
2. ``simulate`` runs all or a selected subset of condition/seed pairs with
   resumable, exact-step-budget training.
3. ``aggregate`` verifies the complete grid and writes run-level summaries,
   paired contrasts, a neutral report, and a six-panel ablation figure.

Shared encoder and mixing helpers live in ``Codes/utils.py``. All
data-generation ablations, metrics, checkpointing, provenance, and
aggregation are implemented here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


SCRIPT_DIR = Path(__file__).resolve().parent
CODES_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = CODES_DIR.parent
UTILS_PATH = CODES_DIR / "utils.py"
sys.path.insert(0, str(CODES_DIR))

from utils import construct_invertible_mlp, get_mlp  # noqa: E402


FORMAT_VERSION = 1
METRIC_FIELDS = (
    "step",
    "train_loss",
    "eval_loss",
    "eval_accuracy",
    "linear_h_to_z",
    "linear_z_to_h",
    "mcc",
    "activation_energy",
    "negative_activation_fraction",
    "dead_unit_fraction",
    "encoder_parameter_l2",
    "readout_parameter_l2",
    "elapsed_seconds",
)
SUMMARY_METRICS = (
    "linear_h_to_z",
    "linear_z_to_h",
    "mcc",
    "eval_loss",
    "eval_accuracy",
    "activation_energy",
    "negative_activation_fraction",
)


@dataclass(frozen=True)
class CommonConfig:
    seeds: tuple[int, ...]
    dgp_seed: int
    train_seed: int
    eval_seed: int
    n: int
    nh: int
    nr_clusters: int
    max_steps: int
    batch_size: int
    eval_samples: int
    log_every: int
    checkpoint_every: int
    learning_rate: float
    n_mixing_layers: int
    mixing_search_steps: int
    n_hidden_layers: int
    use_hidden_batch_norm: bool
    expon_entropy: float
    expon_scale: float
    cluster_fit_steps: int
    deterministic: bool


@dataclass(frozen=True)
class Condition:
    id: str
    family: str
    label: str
    description: str
    distribution: str = "exponential"
    shape: float = 1.0
    entropy_sd: float = 0.0
    dependence_rho: float = 0.0
    sign_flip_prob: float = 0.0
    output_activation: str = "relu"
    negative_slope: float = 0.0
    readout_bias: bool = False
    weight_decay: float = 0.03


@dataclass
class BaseDGP:
    cluster_vectors: torch.Tensor
    entropy_direction: torch.Tensor
    mixing: torch.nn.Module
    artifact_sha256: str


class OutputActivation(torch.nn.Module):
    def __init__(self, base: torch.nn.Module, name: str, negative_slope: float):
        super().__init__()
        self.base = base
        self.name = name
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base(x)
        if self.name == "relu":
            return F.relu(x)
        if self.name == "leaky_relu":
            return F.leaky_relu(x, negative_slope=self.negative_slope)
        if self.name == "linear":
            return x
        raise ValueError(f"Unknown output activation: {self.name}")


def set_seed(seed: int, deterministic: bool = True) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_manifest() -> dict[str, str]:
    paths = [Path(__file__).resolve(), UTILS_PATH]
    return {
        path.relative_to(PROJECT_ROOT).as_posix(): file_sha256(path)
        for path in paths
    }


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_manifest(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": package_version("scipy"),
        "scikit_learn": package_version("scikit-learn"),
        "matplotlib": package_version("matplotlib"),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "code_sha256": code_manifest(),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        result["cuda_device_name"] = torch.cuda.get_device_name(device)
    return result


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def save_torch_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, tmp)
    os.replace(tmp, path)


def load_torch(
    path: Path,
    map_location: str | torch.device = "cpu",
    *,
    weights_only: bool = True,
) -> Any:
    try:
        return torch.load(
            path, map_location=map_location, weights_only=weights_only
        )
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_plan(path: Path) -> tuple[CommonConfig, list[Condition], dict[str, Any]]:
    raw = json.loads(path.read_text())
    if raw.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported plan format in {path}")
    common_raw = dict(raw["common"])
    common_raw["seeds"] = tuple(int(value) for value in common_raw["seeds"])
    common = CommonConfig(**common_raw)
    conditions = [Condition(**item) for item in raw["conditions"]]
    validate_plan(common, conditions)
    return common, conditions, raw


def validate_plan(common: CommonConfig, conditions: list[Condition]) -> None:
    if len(set(common.seeds)) != len(common.seeds) or not common.seeds:
        raise ValueError("Plan seeds must be non-empty and unique")
    ids = [condition.id for condition in conditions]
    if not conditions or len(set(ids)) != len(ids):
        raise ValueError("Condition IDs must be non-empty and unique")
    if "baseline" not in ids:
        raise ValueError("Plan must contain a condition with id='baseline'")
    if common.n <= 0 or common.nh != common.n:
        raise ValueError("These full-capacity ablations require n == nh > 0")
    if common.nr_clusters < common.n:
        raise ValueError("nr_clusters must be at least n")
    if common.max_steps <= 0 or common.batch_size <= 0 or common.eval_samples < 4:
        raise ValueError("Invalid training or evaluation budget")
    if common.log_every <= 0 or common.checkpoint_every <= 0:
        raise ValueError("Logging/checkpoint intervals must be positive")
    valid_distributions = {"exponential", "gamma", "weibull", "lognormal"}
    valid_activations = {"relu", "leaky_relu", "linear"}
    for condition in conditions:
        if condition.distribution not in valid_distributions:
            raise ValueError(f"Invalid distribution in {condition.id}")
        if condition.output_activation not in valid_activations:
            raise ValueError(f"Invalid output activation in {condition.id}")
        if condition.shape <= 0:
            raise ValueError(f"shape must be positive in {condition.id}")
        if condition.entropy_sd < 0:
            raise ValueError(f"entropy_sd must be nonnegative in {condition.id}")
        if not 0 <= condition.dependence_rho < 1:
            raise ValueError(f"dependence_rho must be in [0,1) in {condition.id}")
        if not 0 <= condition.sign_flip_prob <= 0.5:
            raise ValueError(f"sign_flip_prob must be in [0,0.5] in {condition.id}")
        if condition.weight_decay < 0:
            raise ValueError(f"weight_decay must be nonnegative in {condition.id}")
        if condition.distribution != "exponential" and condition.dependence_rho:
            raise ValueError("Copula dependence is implemented only for exponential marginals")


def task_grid(common: CommonConfig, conditions: list[Condition]) -> list[tuple[Condition, int]]:
    return [
        (condition, seed)
        for condition in conditions
        for seed in common.seeds
    ]


def dgp_config(common: CommonConfig) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "dgp_seed": common.dgp_seed,
        "n": common.n,
        "nr_clusters": common.nr_clusters,
        "n_mixing_layers": common.n_mixing_layers,
        "mixing_search_steps": common.mixing_search_steps,
        "expon_entropy": common.expon_entropy,
        "expon_scale": common.expon_scale,
        "cluster_fit_steps": common.cluster_fit_steps,
        "deterministic": common.deterministic,
    }


def fit_cluster_vectors(common: CommonConfig, device: torch.device) -> torch.Tensor:
    logits = torch.randn(
        common.nr_clusters, common.n, device=device, requires_grad=True
    )
    optimizer = torch.optim.Adam([logits], lr=1e-3)
    if common.expon_entropy > 0:
        initial = torch.softmax(logits, dim=1)
        target_log_product = torch.log(initial).sum(dim=1).mean().detach()
    else:
        target_log_product = torch.tensor(common.expon_entropy, device=device)

    for _ in range(common.cluster_fit_steps):
        cluster_vectors = F.softplus(logits)
        log_product = torch.log(cluster_vectors).sum(dim=1)
        loss = ((log_product - target_log_product) ** 2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return F.softplus(logits).detach()


def build_mixing(common: CommonConfig, device: torch.device) -> torch.nn.Module:
    mixing = construct_invertible_mlp(
        n=common.n,
        n_layers=common.n_mixing_layers,
        act_fct="leaky_relu",
        cond_thresh_ratio=0.0,
        n_iter_cond_thresh=common.mixing_search_steps,
    ).to(device)
    mixing.eval()
    for parameter in mixing.parameters():
        parameter.requires_grad = False
    return mixing


def build_mixing_architecture(common: CommonConfig, device: torch.device) -> torch.nn.Module:
    modules: list[torch.nn.Module] = []
    for layer_index in range(common.n_mixing_layers):
        modules.append(torch.nn.Linear(common.n, common.n, bias=False))
        if layer_index < common.n_mixing_layers - 1:
            modules.append(torch.nn.LeakyReLU(negative_slope=0.2))
    mixing = torch.nn.Sequential(*modules).to(device)
    mixing.eval()
    for parameter in mixing.parameters():
        parameter.requires_grad = False
    return mixing


def prepare_dgp(common: CommonConfig, artifact_dir: Path, device_name: str) -> Path:
    artifact_dir = artifact_dir.resolve()
    artifact_path = artifact_dir / "dgp.pt"
    manifest_path = artifact_dir / "manifest.json"
    expected = dgp_config(common)
    if artifact_path.exists() or manifest_path.exists():
        if not artifact_path.exists() or not manifest_path.exists():
            raise FileExistsError(f"Incomplete DGP artifact: {artifact_dir}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("dgp_config") != expected:
            raise ValueError(f"Existing DGP config mismatch: {artifact_dir}")
        if manifest.get("artifact_sha256") != file_sha256(artifact_path):
            raise ValueError(f"Existing DGP checksum mismatch: {artifact_path}")
        print(f"DGP artifact already valid: {artifact_dir}")
        return artifact_dir

    device = resolve_device(device_name)
    set_seed(common.dgp_seed, common.deterministic)
    cluster_vectors = fit_cluster_vectors(common, device)
    mixing = build_mixing(common, device)
    direction_generator = torch.Generator(device="cpu")
    direction_generator.manual_seed(common.dgp_seed + 104_729)
    entropy_direction = torch.randn(common.nr_clusters, generator=direction_generator)
    entropy_direction = (entropy_direction - entropy_direction.mean()) / entropy_direction.std(
        unbiased=False
    )
    save_torch_atomic(
        artifact_path,
        {
            "cluster_vectors": cluster_vectors.detach().cpu(),
            "entropy_direction": entropy_direction,
            "mixing_state": {
                key: value.detach().cpu() for key, value in mixing.state_dict().items()
            },
        },
    )
    log_products = torch.log(cluster_vectors.detach().cpu()).sum(dim=1)
    manifest = {
        "dgp_config": expected,
        "artifact_sha256": file_sha256(artifact_path),
        "cluster_log_product_mean": float(log_products.mean()),
        "cluster_log_product_sd": float(log_products.std(unbiased=False)),
        "entropy_direction_mean": float(entropy_direction.mean()),
        "entropy_direction_sd": float(entropy_direction.std(unbiased=False)),
        "environment": environment_manifest(device),
    }
    write_json(manifest_path, manifest)
    print(f"Saved shared DGP artifact: {artifact_dir}")
    return artifact_dir


def load_dgp(common: CommonConfig, artifact_dir: Path, device: torch.device) -> BaseDGP:
    artifact_dir = artifact_dir.resolve()
    artifact_path = artifact_dir / "dgp.pt"
    manifest_path = artifact_dir / "manifest.json"
    if not artifact_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"Missing DGP artifact under {artifact_dir}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("dgp_config") != dgp_config(common):
        raise ValueError(f"DGP settings do not match plan: {artifact_dir}")
    actual_hash = file_sha256(artifact_path)
    if manifest.get("artifact_sha256") != actual_hash:
        raise ValueError(f"DGP checksum mismatch: {artifact_path}")
    payload = load_torch(artifact_path)
    mixing = build_mixing_architecture(common, device)
    mixing.load_state_dict(payload["mixing_state"], strict=True)
    return BaseDGP(
        cluster_vectors=payload["cluster_vectors"].to(device),
        entropy_direction=payload["entropy_direction"].to(device),
        mixing=mixing,
        artifact_sha256=actual_hash,
    )


def condition_means(
    common: CommonConfig, condition: Condition, dgp: BaseDGP
) -> torch.Tensor:
    baseline_means = dgp.cluster_vectors / common.expon_scale
    multiplier = torch.exp(
        condition.entropy_sd * dgp.entropy_direction[:, None] / common.n
    )
    return baseline_means * multiplier


def class_entropy_proxy(
    common: CommonConfig, condition: Condition, dgp: BaseDGP
) -> torch.Tensor:
    """Class-varying part of entropy; constants shared by all classes omitted."""
    means = condition_means(common, condition, dgp)
    return torch.log(means).sum(dim=1)


def sample_latents(
    common: CommonConfig,
    condition: Condition,
    dgp: BaseDGP,
    size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = torch.randint(0, common.nr_clusters, (size,), device=device)
    means = condition_means(common, condition, dgp)[labels]
    eps = torch.finfo(means.dtype).eps

    if condition.distribution == "exponential":
        if condition.dependence_rho == 0:
            uniforms = torch.rand_like(means).clamp_(eps, 1 - eps)
        else:
            independent = torch.randn_like(means)
            common_noise = torch.randn(size, 1, device=device, dtype=means.dtype)
            normal = (
                math.sqrt(condition.dependence_rho) * common_noise
                + math.sqrt(1 - condition.dependence_rho) * independent
            )
            uniforms = (0.5 * (1 + torch.erf(normal / math.sqrt(2)))).clamp_(
                eps, 1 - eps
            )
        samples = -means * torch.log1p(-uniforms)
    elif condition.distribution == "gamma":
        concentration = torch.full_like(means, condition.shape)
        samples = torch.distributions.Gamma(
            concentration=concentration,
            rate=concentration / means,
        ).sample()
    elif condition.distribution == "weibull":
        uniforms = torch.rand_like(means).clamp_(eps, 1 - eps)
        scale = means / math.gamma(1 + 1 / condition.shape)
        samples = scale * (-torch.log1p(-uniforms)).pow(1 / condition.shape)
    elif condition.distribution == "lognormal":
        # sigma^2=log(2) matches both the mean and variance of an exponential.
        sigma2 = math.log(2.0)
        mu = torch.log(means) - 0.5 * sigma2
        samples = torch.exp(mu + math.sqrt(sigma2) * torch.randn_like(means))
    else:
        raise AssertionError(condition.distribution)

    if condition.sign_flip_prob > 0:
        flips = torch.rand_like(samples) < condition.sign_flip_prob
        samples = torch.where(flips, -samples, samples)
    return samples, labels


def create_encoder(
    common: CommonConfig, condition: Condition, device: torch.device
) -> torch.nn.Module:
    hidden = (
        [common.n * 10]
        + [common.n * 50] * (common.n_hidden_layers - 2)
        + [common.n * 10]
    )
    base = get_mlp(
        n_in=common.n,
        n_out=common.nh,
        layers=hidden,
        layer_normalization="bn" if common.use_hidden_batch_norm else None,
        output_normalization=None,
    )
    return OutputActivation(
        base, condition.output_activation, condition.negative_slope
    ).to(device)


def centered_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_centered = x - x.mean(axis=0, keepdims=True)
    y_centered = y - y.mean(axis=0, keepdims=True)
    x_norm = np.linalg.norm(x_centered, axis=0, keepdims=True)
    y_norm = np.linalg.norm(y_centered, axis=0, keepdims=True)
    x_norm[x_norm == 0] = 1.0
    y_norm[y_norm == 0] = 1.0
    corr = (x_centered / x_norm).T @ (y_centered / y_norm)
    return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)


def diagonal_correlation(x: np.ndarray, y: np.ndarray) -> float:
    corr = centered_correlation(x, y)
    diagonal_length = min(corr.shape)
    return float(np.abs(np.diag(corr[:diagonal_length, :diagonal_length])).mean())


def linear_recovery(target: np.ndarray, predictors: np.ndarray) -> float:
    split = len(target) // 2
    design_train = np.column_stack([predictors[:split], np.ones(split)])
    coefficients, *_ = np.linalg.lstsq(design_train, target[:split], rcond=None)
    design_test = np.column_stack(
        [predictors[split:], np.ones(len(predictors) - split)]
    )
    predictions = design_test @ coefficients
    return diagonal_correlation(target[split:], predictions)


def permutation_scaling_mcc(z: np.ndarray, h: np.ndarray) -> tuple[float, np.ndarray]:
    corr = centered_correlation(z, h)
    rows, columns = linear_sum_assignment(-np.abs(corr))
    return float(np.abs(corr[rows, columns]).mean()), corr


def parameter_l2(module: torch.nn.Module) -> float:
    total = 0.0
    with torch.no_grad():
        for parameter in module.parameters():
            total += float((parameter.detach().float() ** 2).sum().cpu())
    return math.sqrt(total)


def evaluate(
    common: CommonConfig,
    z_eval: torch.Tensor,
    labels_eval: torch.Tensor,
    dgp: BaseDGP,
    encoder: torch.nn.Module,
    readout: torch.nn.Module,
) -> tuple[dict[str, float], np.ndarray, torch.Tensor]:
    encoder.eval()
    readout.eval()
    with torch.inference_mode():
        h_eval = encoder(dgp.mixing(z_eval))
        logits = readout(h_eval)
        eval_loss = F.cross_entropy(logits, labels_eval)
        accuracy = (logits.argmax(dim=1) == labels_eval).float().mean()
    z_np = z_eval.detach().cpu().numpy()
    h_np = h_eval.detach().cpu().numpy()
    mcc, corr = permutation_scaling_mcc(z_np, h_np)
    metrics = {
        "eval_loss": float(eval_loss.cpu()),
        "eval_accuracy": float(accuracy.cpu()),
        "linear_h_to_z": linear_recovery(z_np, h_np),
        "linear_z_to_h": linear_recovery(h_np, z_np),
        "mcc": mcc,
        "activation_energy": float((h_eval**2).sum(dim=1).mean().cpu()),
        "negative_activation_fraction": float((h_eval < 0).float().mean().cpu()),
        "dead_unit_fraction": float(
            (h_eval.detach().abs().max(dim=0).values < 1e-12).float().mean().cpu()
        ),
    }
    return metrics, corr, h_eval.detach().cpu()


def append_metric(path: Path, row: dict[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row[field] for field in METRIC_FIELDS})
        stream.flush()
        os.fsync(stream.fileno())


def truncate_metrics(path: Path, step: int) -> None:
    if not path.exists():
        return
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    retained = [row for row in rows if int(float(row["step"])) <= step]
    if len(retained) == len(rows):
        return
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(retained)
    os.replace(tmp, path)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def run_id(condition: Condition, seed: int) -> str:
    return f"{condition.id}__seed-{seed}"


def scientific_run_config(
    common: CommonConfig, condition: Condition, seed: int, dgp_hash: str
) -> dict[str, Any]:
    common_values = asdict(common)
    common_values["seeds"] = list(common.seeds)
    return {
        "format_version": FORMAT_VERSION,
        "common": common_values,
        "condition": asdict(condition),
        "init_seed": seed,
        "dgp_artifact_sha256": dgp_hash,
    }


def run_one(
    common: CommonConfig,
    condition: Condition,
    seed: int,
    dgp_path: Path,
    output_root: Path,
    device_name: str,
    resume: bool,
) -> Path:
    device = resolve_device(device_name)
    dgp = load_dgp(common, dgp_path, device)
    run_dir = output_root.resolve() / run_id(condition, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    config = scientific_run_config(common, condition, seed, dgp.artifact_sha256)
    config_path = run_dir / "config.json"
    completed_path = run_dir / "COMPLETED.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError(f"Incompatible existing run config: {run_dir}")
    if completed_path.exists():
        print(f"Already complete: {run_dir}")
        return run_dir
    if any(run_dir.iterdir()) and not resume:
        raise FileExistsError(f"Incomplete run exists; pass --resume: {run_dir}")
    write_json(config_path, config)
    write_json(run_dir / "environment.json", environment_manifest(device))
    entropy_proxy = class_entropy_proxy(common, condition, dgp).detach().cpu()
    write_json(
        run_dir / "dgp_diagnostics.json",
        {
            "artifact_sha256": dgp.artifact_sha256,
            "class_entropy_proxy_mean": float(entropy_proxy.mean()),
            "class_entropy_proxy_sd": float(entropy_proxy.std(unbiased=False)),
            "target_entropy_sd": condition.entropy_sd,
        },
    )

    set_seed(common.eval_seed, common.deterministic)
    z_eval, labels_eval = sample_latents(
        common, condition, dgp, common.eval_samples, device
    )
    save_torch_atomic(run_dir / "eval_latents.pt", z_eval.detach().cpu())
    save_torch_atomic(run_dir / "eval_labels.pt", labels_eval.detach().cpu())

    set_seed(seed, common.deterministic)
    encoder = create_encoder(common, condition, device)
    readout = torch.nn.Linear(
        common.nh,
        common.nr_clusters,
        bias=condition.readout_bias,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [*encoder.parameters(), *readout.parameters()],
        lr=common.learning_rate,
        weight_decay=condition.weight_decay,
    )

    checkpoint_path = run_dir / "checkpoint.pt"
    metrics_path = run_dir / "metrics.csv"
    start_step = 0
    elapsed_before = 0.0
    interval_loss_sum = 0.0
    interval_count = 0
    if resume and checkpoint_path.exists():
        checkpoint = load_torch(checkpoint_path, device, weights_only=False)
        encoder.load_state_dict(checkpoint["encoder"])
        readout.load_state_dict(checkpoint["readout"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        elapsed_before = float(checkpoint.get("elapsed_seconds", 0.0))
        interval_loss_sum = float(checkpoint.get("interval_loss_sum", 0.0))
        interval_count = int(checkpoint.get("interval_count", 0))
        restore_rng_state(checkpoint["rng_state"])
        truncate_metrics(metrics_path, start_step)
    else:
        set_seed(common.train_seed, common.deterministic)

    started = time.perf_counter()
    last_metrics: dict[str, float] | None = None
    last_corr: np.ndarray | None = None
    last_h: torch.Tensor | None = None
    if start_step == 0 and not metrics_path.exists():
        last_metrics, last_corr, last_h = evaluate(
            common, z_eval, labels_eval, dgp, encoder, readout
        )
        append_metric(
            metrics_path,
            {
                "step": 0,
                "train_loss": math.nan,
                **last_metrics,
                "encoder_parameter_l2": parameter_l2(encoder),
                "readout_parameter_l2": parameter_l2(readout),
                "elapsed_seconds": elapsed_before,
            },
        )

    encoder.train()
    readout.train()
    for step in range(start_step + 1, common.max_steps + 1):
        z_batch, labels = sample_latents(
            common, condition, dgp, common.batch_size, device
        )
        h_batch = encoder(dgp.mixing(z_batch))
        logits = readout(h_batch)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        interval_loss_sum += float(loss.detach().cpu())
        interval_count += 1

        should_log = step % common.log_every == 0 or step == common.max_steps
        if should_log:
            last_metrics, last_corr, last_h = evaluate(
                common, z_eval, labels_eval, dgp, encoder, readout
            )
            elapsed = elapsed_before + time.perf_counter() - started
            row = {
                "step": step,
                "train_loss": interval_loss_sum / max(interval_count, 1),
                **last_metrics,
                "encoder_parameter_l2": parameter_l2(encoder),
                "readout_parameter_l2": parameter_l2(readout),
                "elapsed_seconds": elapsed,
            }
            append_metric(metrics_path, row)
            print(
                f"{run_id(condition, seed)} step={step}/{common.max_steps} "
                f"linear={row['linear_h_to_z']:.4f} mcc={row['mcc']:.4f} "
                f"eval_loss={row['eval_loss']:.4f}"
            )
            interval_loss_sum = 0.0
            interval_count = 0
            encoder.train()
            readout.train()

        should_checkpoint = (
            step % common.checkpoint_every == 0 or step == common.max_steps
        )
        if should_checkpoint:
            elapsed = elapsed_before + time.perf_counter() - started
            save_torch_atomic(
                checkpoint_path,
                {
                    "step": step,
                    "encoder": encoder.state_dict(),
                    "readout": readout.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "interval_loss_sum": interval_loss_sum,
                    "interval_count": interval_count,
                    "elapsed_seconds": elapsed,
                    "rng_state": capture_rng_state(),
                },
            )

    if last_metrics is None or last_corr is None or last_h is None:
        last_metrics, last_corr, last_h = evaluate(
            common, z_eval, labels_eval, dgp, encoder, readout
        )
    save_torch_atomic(run_dir / "model.pt", encoder.state_dict())
    save_torch_atomic(run_dir / "readout.pt", readout.state_dict())
    save_torch_atomic(run_dir / "final_activations.pt", last_h)
    np.savetxt(run_dir / "final_correlation.csv", last_corr, delimiter=",")
    write_json(
        completed_path,
        {
            "run_id": run_id(condition, seed),
            "condition_id": condition.id,
            "init_seed": seed,
            "max_steps": common.max_steps,
            "elapsed_seconds": elapsed_before + time.perf_counter() - started,
            **last_metrics,
        },
    )
    return run_dir


def read_final_row(
    run_dir: Path,
    expected_common: CommonConfig,
    expected_condition: Condition,
    expected_seed: int,
) -> dict[str, Any]:
    config = json.loads((run_dir / "config.json").read_text())
    completed = json.loads((run_dir / "COMPLETED.json").read_text())
    environment = json.loads((run_dir / "environment.json").read_text())
    expected_common_values = asdict(expected_common)
    expected_common_values["seeds"] = list(expected_common.seeds)
    if config.get("common") != expected_common_values:
        raise ValueError(f"Run uses different common settings: {run_dir}")
    if config.get("condition") != asdict(expected_condition):
        raise ValueError(f"Run uses different condition settings: {run_dir}")
    if int(config.get("init_seed")) != expected_seed:
        raise ValueError(f"Run seed does not match its planned task: {run_dir}")
    if completed.get("run_id") != run_id(expected_condition, expected_seed):
        raise ValueError(f"Completion marker has wrong run ID: {run_dir}")
    if int(completed.get("max_steps")) != expected_common.max_steps:
        raise ValueError(f"Run did not use the planned step budget: {run_dir}")
    for metric in SUMMARY_METRICS:
        if not math.isfinite(float(completed[metric])):
            raise ValueError(f"Non-finite final {metric} in {run_dir}")
    condition = config["condition"]
    code_signature = hashlib.sha256(
        json.dumps(
            environment.get("code_sha256", {}),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return {
        "run_id": completed["run_id"],
        "condition_id": condition["id"],
        "family": condition["family"],
        "label": condition["label"],
        "description": condition["description"],
        "seed": int(config["init_seed"]),
        "dgp_artifact_sha256": config["dgp_artifact_sha256"],
        "training_code_signature": code_signature,
        "distribution": condition["distribution"],
        "shape": condition["shape"],
        "entropy_sd": condition["entropy_sd"],
        "dependence_rho": condition["dependence_rho"],
        "sign_flip_prob": condition["sign_flip_prob"],
        "output_activation": condition["output_activation"],
        "negative_slope": condition["negative_slope"],
        "readout_bias": condition["readout_bias"],
        "weight_decay": condition["weight_decay"],
        "max_steps": completed["max_steps"],
        "elapsed_seconds": completed["elapsed_seconds"],
        **{metric: completed[metric] for metric in SUMMARY_METRICS},
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def summarize_values(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    return {
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "sem": float(array.std(ddof=1) / math.sqrt(len(array))) if len(array) > 1 else 0.0,
        "median": float(np.median(array)),
        "q025": float(np.quantile(array, 0.025)),
        "q975": float(np.quantile(array, 0.975)),
    }


def condition_summaries(
    rows: list[dict[str, Any]], conditions: list[Condition]
) -> list[dict[str, Any]]:
    order = {condition.id: index for index, condition in enumerate(conditions)}
    result = []
    for condition in conditions:
        group = [row for row in rows if row["condition_id"] == condition.id]
        summary: dict[str, Any] = {
            "condition_order": order[condition.id],
            "condition_id": condition.id,
            "family": condition.family,
            "label": condition.label,
            "n_runs": len(group),
        }
        for metric in SUMMARY_METRICS:
            for statistic, value in summarize_values(row[metric] for row in group).items():
                summary[f"{metric}_{statistic}"] = value
        result.append(summary)
    return result


def paired_rows(
    rows: list[dict[str, Any]], conditions: list[Condition]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline = {
        int(row["seed"]): row for row in rows if row["condition_id"] == "baseline"
    }
    deltas: list[dict[str, Any]] = []
    for row in rows:
        if row["condition_id"] == "baseline":
            continue
        reference = baseline[int(row["seed"])]
        delta = {
            "condition_id": row["condition_id"],
            "family": row["family"],
            "label": row["label"],
            "seed": row["seed"],
        }
        for metric in SUMMARY_METRICS:
            delta[f"delta_{metric}"] = float(row[metric]) - float(reference[metric])
        deltas.append(delta)
    summaries: list[dict[str, Any]] = []
    for condition in conditions:
        if condition.id == "baseline":
            continue
        group = [row for row in deltas if row["condition_id"] == condition.id]
        summary: dict[str, Any] = {
            "condition_id": condition.id,
            "family": condition.family,
            "label": condition.label,
            "n_pairs": len(group),
        }
        for metric in SUMMARY_METRICS:
            values = [row[f"delta_{metric}"] for row in group]
            for statistic, value in summarize_values(values).items():
                summary[f"delta_{metric}_{statistic}"] = value
        summaries.append(summary)
    return deltas, summaries


def panel_conditions(
    family: str, conditions: list[Condition]
) -> list[Condition]:
    baseline = next(condition for condition in conditions if condition.id == "baseline")
    return [baseline, *[condition for condition in conditions if condition.family == family]]


def condition_x(family: str, condition: Condition) -> float | str:
    if condition.id == "baseline":
        defaults: dict[str, float | str] = {
            "entropy": 0.0,
            "dependence": 0.0,
            "distribution": "Exponential",
            "latent_nonnegativity": 0.0,
            "activation_nonnegativity": "ReLU",
            "energy": 0.03,
        }
        return defaults[family]
    if family == "entropy":
        return condition.entropy_sd
    if family == "dependence":
        return condition.dependence_rho
    if family == "distribution":
        return condition.label
    if family == "latent_nonnegativity":
        return condition.sign_flip_prob
    if family == "activation_nonnegativity":
        return condition.label
    if family == "energy":
        return condition.weight_decay
    raise AssertionError(family)


def make_figure(
    conditions: list[Condition], summaries: list[dict[str, Any]], output_dir: Path
) -> None:
    cache_dir = PROJECT_ROOT / "Data" / ".cache" / "fig8" / "matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / "Data" / ".cache" / "fig8"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary_by_id = {row["condition_id"]: row for row in summaries}
    families = (
        ("entropy", "Class-dependent entropy", "Entropy SD across classes (nats)"),
        ("dependence", "Conditional dependence", "Gaussian-copula correlation"),
        ("distribution", "Non-exponential marginals", "Conditional distribution"),
        ("latent_nonnegativity", "Signed latents (boundary)", "Sign-flip probability"),
        ("activation_nonnegativity", "Activation boundary", "Output activation"),
        ("energy", "Energy regularization", "AdamW weight decay"),
    )
    metric_styles = {
        "linear_h_to_z": ("#4C78A8", "Linear recovery"),
        "mcc": ("#F58518", "Permutation-scaling MCC"),
    }
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.1), sharey=True)
    for panel_index, (axis, (family, title, xlabel)) in enumerate(
        zip(axes.flat, families)
    ):
        selected = panel_conditions(family, conditions)
        if family == "entropy":
            series = {
                "bias-free": [
                    condition
                    for condition in selected
                    if condition.id == "baseline" or not condition.readout_bias
                ],
                "affine readout": [
                    condition for condition in selected if condition.readout_bias
                ],
            }
        else:
            series = {"": selected}
        categorical = family in {"distribution", "activation_nonnegativity"}
        for series_index, (series_name, series_conditions) in enumerate(series.items()):
            if not series_conditions:
                continue
            if categorical:
                x_values = np.arange(len(series_conditions), dtype=float)
                labels = [str(condition_x(family, item)) for item in series_conditions]
            else:
                pairs = sorted(
                    ((float(condition_x(family, item)), item) for item in series_conditions),
                    key=lambda pair: pair[0],
                )
                x_values = np.asarray([pair[0] for pair in pairs])
                series_conditions = [pair[1] for pair in pairs]
                labels = []
            for metric, (color, metric_label) in metric_styles.items():
                means = np.asarray(
                    [summary_by_id[item.id][f"{metric}_mean"] for item in series_conditions]
                )
                sems = np.asarray(
                    [summary_by_id[item.id][f"{metric}_sem"] for item in series_conditions]
                )
                linestyle = "-" if series_index == 0 else "--"
                legend_label = metric_label + (f" ({series_name})" if series_name else "")
                axis.errorbar(
                    x_values,
                    means,
                    yerr=sems,
                    marker="o" if series_index == 0 else "s",
                    linestyle=linestyle,
                    color=color,
                    linewidth=1.6,
                    markersize=4.2,
                    capsize=2.5,
                    label=legend_label,
                )
            if categorical:
                axis.set_xticks(x_values, labels, rotation=25, ha="right")
        axis.axhline(1.0, color="0.85", linewidth=0.7, zorder=0)
        axis.set_ylim(0, 1.04)
        axis.set_title(title, fontsize=11)
        axis.set_xlabel(xlabel)
        axis.grid(axis="y", alpha=0.2)
        axis.text(
            -0.13,
            1.04,
            chr(ord("A") + panel_index),
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=12,
        )
        if panel_index % 3 == 0:
            axis.set_ylabel("Final score")
        axis.legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle(
        "Assumption ablations: linear identifiability versus axis alignment",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "fig8.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "fig8.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_report(
    plan_path: Path,
    common: CommonConfig,
    conditions: list[Condition],
    summaries: list[dict[str, Any]],
    paired_summaries: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    summary_by_id = {row["condition_id"]: row for row in summaries}
    paired_by_id = {row["condition_id"]: row for row in paired_summaries}
    lines = [
        "# Assumption ablation results",
        "",
        f"Plan: `{plan_path.resolve()}`  ",
        f"Seeds per condition: {len(common.seeds)}  ",
        f"Training budget: {common.max_steps:,} gradient steps per run  ",
        f"Shared DGP seed: {common.dgp_seed}",
        "",
        "Linear recovery is the held-out Pearson score after an unrestricted linear map from learned activations to the ground-truth latents. MCC uses a one-to-one Hungarian assignment and absolute Pearson correlation, so it permits permutation and coordinate-wise scaling but not general rotations.",
        "",
        "| Family | Condition | n | Linear recovery | MCC | Paired Δ linear | Paired Δ MCC |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for condition in conditions:
        row = summary_by_id[condition.id]
        linear = f"{row['linear_h_to_z_mean']:.3f} ± {row['linear_h_to_z_sem']:.3f}"
        mcc = f"{row['mcc_mean']:.3f} ± {row['mcc_sem']:.3f}"
        if condition.id == "baseline":
            delta_linear = "—"
            delta_mcc = "—"
        else:
            paired = paired_by_id[condition.id]
            delta_linear = f"{paired['delta_linear_h_to_z_mean']:+.3f}"
            delta_mcc = f"{paired['delta_mcc_mean']:+.3f}"
        lines.append(
            f"| {condition.family} | {condition.label} | {row['n_runs']} | {linear} | {mcc} | {delta_linear} | {delta_mcc} |"
        )
    lines.extend(
        [
            "",
            "## Reporting notes",
            "",
            "- Values are mean ± SEM across initialization seeds; paired deltas subtract the same-seed baseline.",
            "- Entropy dispersion is defined exactly as the across-class standard deviation of the class-varying entropy term, in nats.",
            "- The Gaussian copula changes conditional dependence while preserving each exponential marginal.",
            "- Gamma, Weibull, and log-normal conditions match the original class-conditional means; the log-normal also matches exponential variance.",
            "- Signed-latent and non-nonnegative-output settings are boundary tests, not settings in which the theorem predicts persistence.",
            "- Interpret a selective MCC decrease with preserved linear recovery as loss of axis alignment; a decrease in both metrics indicates loss of linear identifiability under this finite-budget experiment.",
            "- Classification loss/accuracy and both linear directions are available in `runs.csv`; inspect them before attributing a low recovery score to an assumption rather than poor task optimization.",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines))
    caption = (
        "Assumption ablations for the sparse generative model. Each point shows mean ± s.e.m. "
        f"over {len(common.seeds)} paired initialization seeds after {common.max_steps:,} gradient "
        "steps. Blue denotes held-out recovery up to an arbitrary linear map; orange denotes "
        "permutation-and-scaling MCC. All conditions share class-scale vectors, nonlinear mixing, "
        "training-batch seed, and evaluation seed. Class-dependent entropy is varied in nats; "
        "conditional dependence uses a Gaussian copula with unchanged exponential marginals; "
        "non-exponential distributions match class-conditional means; sign flips and output leakage "
        "probe the boundary of nonnegativity; and the energy panel varies AdamW weight decay."
    )
    (output_dir / "caption.txt").write_text(caption + "\n")


def aggregate(
    plan_path: Path,
    results_root: Path,
    output_dir: Path,
    allow_incomplete: bool,
) -> None:
    common, conditions, raw_plan = load_plan(plan_path)
    expected = task_grid(common, conditions)
    missing = [
        run_id(condition, seed)
        for condition, seed in expected
        if not (results_root / run_id(condition, seed) / "COMPLETED.json").exists()
    ]
    if missing and not allow_incomplete:
        preview = ", ".join(missing[:8])
        raise RuntimeError(f"Missing {len(missing)} completed runs: {preview}")
    rows = [
        read_final_row(
            results_root / run_id(condition, seed), common, condition, seed
        )
        for condition, seed in expected
        if (results_root / run_id(condition, seed) / "COMPLETED.json").exists()
    ]
    present_ids = {row["condition_id"] for row in rows}
    if any(condition.id not in present_ids for condition in conditions):
        raise RuntimeError("At least one condition has no completed run; cannot summarize")
    dgp_hashes = {row["dgp_artifact_sha256"] for row in rows}
    if len(dgp_hashes) != 1:
        raise ValueError(
            f"Expected one shared DGP artifact across all runs, found {sorted(dgp_hashes)}"
        )
    training_code_signatures = {row["training_code_signature"] for row in rows}
    if len(training_code_signatures) != 1:
        raise ValueError(
            "Completed runs were produced by more than one training-code snapshot: "
            f"{sorted(training_code_signatures)}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = condition_summaries(rows, conditions)
    paired, paired_summaries = paired_rows(rows, conditions)
    write_csv(output_dir / "runs.csv", rows)
    write_csv(output_dir / "conditions.csv", summaries)
    write_csv(output_dir / "paired.csv", paired)
    write_csv(output_dir / "paired-summary.csv", paired_summaries)
    make_figure(conditions, summaries, output_dir)
    make_report(
        plan_path, common, conditions, summaries, paired_summaries, output_dir
    )
    artifact_manifest = {
        "format_version": FORMAT_VERSION,
        "plan_path": str(plan_path.resolve()),
        "plan_sha256": file_sha256(plan_path),
        "plan": raw_plan,
        "results_root": str(results_root.resolve()),
        "expected_runs": len(expected),
        "completed_runs": len(rows),
        "missing_runs": missing,
        "dgp_artifact_sha256": next(iter(dgp_hashes)),
        "training_code_signature": next(iter(training_code_signatures)),
        "code_sha256": code_manifest(),
        "outputs_sha256": {
            path.name: file_sha256(path)
            for path in sorted(output_dir.iterdir())
            if path.is_file() and path.name != "manifest.json"
        },
    }
    write_json(output_dir / "manifest.json", artifact_manifest)
    print(f"Aggregated {len(rows)}/{len(expected)} runs into {output_dir.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    dgp_parser = subparsers.add_parser("prepare-dgp", help="Create shared DGP")
    dgp_parser.add_argument("--plan", required=True)
    dgp_parser.add_argument("--artifact-dir", required=True)
    dgp_parser.add_argument("--device", default="auto")

    simulate_parser = subparsers.add_parser(
        "simulate", help="Run all or selected condition/seed pairs"
    )
    simulate_parser.add_argument("--plan", required=True)
    simulate_parser.add_argument("--dgp-path", required=True)
    simulate_parser.add_argument("--output-root", required=True)
    simulate_parser.add_argument("--device", default="auto")
    simulate_parser.add_argument("--resume", action="store_true")
    simulate_parser.add_argument(
        "--conditions",
        nargs="+",
        metavar="ID",
        help="condition IDs to run (default: every condition in the plan)",
    )
    simulate_parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        metavar="SEED",
        help="initialization seeds to run (default: every seed in the plan)",
    )

    aggregate_parser = subparsers.add_parser("aggregate", help="Verify and summarize grid")
    aggregate_parser.add_argument("--plan", required=True)
    aggregate_parser.add_argument("--results-root", required=True)
    aggregate_parser.add_argument("--output-dir", required=True)
    aggregate_parser.add_argument("--allow-incomplete", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    plan_path = Path(args.plan).resolve()
    common, conditions, _ = load_plan(plan_path)
    if args.command == "prepare-dgp":
        prepare_dgp(common, Path(args.artifact_dir), args.device)
        return
    if args.command == "simulate":
        condition_by_id = {condition.id: condition for condition in conditions}
        requested_conditions = args.conditions or list(condition_by_id)
        unknown_conditions = sorted(set(requested_conditions) - set(condition_by_id))
        if unknown_conditions:
            raise ValueError(f"unknown condition IDs: {unknown_conditions}")

        requested_seeds = args.seeds or list(common.seeds)
        unknown_seeds = sorted(set(requested_seeds) - set(common.seeds))
        if unknown_seeds:
            raise ValueError(f"seeds not declared in the plan: {unknown_seeds}")

        selected_grid = [
            (condition_by_id[condition_id], seed)
            for condition_id in requested_conditions
            for seed in requested_seeds
        ]
        for index, (condition, seed) in enumerate(selected_grid, start=1):
            print(
                f"run {index}/{len(selected_grid)}: "
                f"condition={condition.id} seed={seed}"
            )
            run_one(
                common,
                condition,
                seed,
                Path(args.dgp_path),
                Path(args.output_root),
                args.device,
                args.resume,
            )
        return
    if args.command == "aggregate":
        aggregate(
            plan_path,
            Path(args.results_root),
            Path(args.output_dir),
            args.allow_incomplete,
        )
        return
    raise AssertionError(args.command)


if __name__ == "__main__":
    main()
