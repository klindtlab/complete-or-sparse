#!/usr/bin/env python3
"""Reproducible training, reliability analysis, and plotting for Figure 7.

The script has four subcommands:

``plot``
    Rebuild main-text panels A--D from the revised weight-decay-zero runs and
    replace the former example-only panel E with seed-level reliability
    summaries from the paired weight-decay sweep.

``train``
    Run one revised experiment with independent DGP, initialization, training,
    and evaluation seeds, a fixed evaluation set, an exact gradient-step
    budget, append-only metrics, and resumable checkpoints.

``sweep``
    Run a paired sparse-ReLU initialization/weight-decay grid.  All variables
    except initialization and weight decay are held fixed.

``prepare-dgp``
    Materialize one portable DGP artifact for reuse across independent runs.

New outputs are written below ``Data/fig7`` by default. Shared numerical
building blocks live in ``Codes/utils.py``.
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
from typing import Any, Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as F

# Plotting imports are intentionally lazy so simulation-only runs do not pay
# Matplotlib startup cost or create a font cache unnecessarily.
plt: Any = None
Line2D: Any = None


SCRIPT_DIR = Path(__file__).resolve().parent
CODES_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = CODES_DIR.parent
UTILS_PATH = CODES_DIR / "utils.py"
sys.path.insert(0, str(CODES_DIR))

from utils import (  # noqa: E402
    NSphereSpace,
    construct_invertible_mlp,
    disentanglement_score,
    get_mlp,
    linear_disentanglement,
)


CONDITIONS = {
    "hypersphere_linear": (0, 0),
    "hypersphere_relu": (0, 1),
    "sparse_linear": (1, 0),
    "sparse_relu": (1, 1),
}
CONDITION_LABELS = {
    "hypersphere_linear": "Linear",
    "hypersphere_relu": "ReLU",
    "sparse_linear": "Linear",
    "sparse_relu": "ReLU",
}
LINEAR_COLOR = "#4C78A8"
RELU_COLOR = "#F58518"
SEED_TRACE_ALPHA = 0.17


def load_plotting() -> None:
    global plt, Line2D
    if plt is not None:
        return
    local_cache = PROJECT_ROOT / "Data" / ".cache" / "fig7" / "matplotlib"
    local_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(local_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / "Data" / ".cache" / "fig7"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot
    from matplotlib.lines import Line2D as MatplotlibLine2D

    plt = pyplot
    Line2D = MatplotlibLine2D


@dataclass(frozen=True)
class TrainConfig:
    condition: str
    output_root: str
    dgp_seed: int = 20260728
    init_seed: int = 0
    train_seed: int = 20260729
    eval_seed: int = 20260730
    n: int = 5
    nh: int = 5
    nr_clusters: int = 100
    max_steps: int = 1_000_000
    batch_size: int = 1000
    eval_samples: int = 4096
    log_every: int = 1000
    checkpoint_every: int = 10_000
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    threshold: float = 0.90
    n_mixing_layers: int = 4
    mixing_search_steps: int = 25_000
    n_hidden_layers: int = 6
    use_hidden_batch_norm: bool = True
    c_param: float = 10.0
    expon_entropy: float = 1.0
    expon_scale: float = 0.1
    cluster_fit_steps: int = 20_000
    device: str = "auto"
    deterministic: bool = True
    resume: bool = False

    @property
    def expon(self) -> int:
        return CONDITIONS[self.condition][0]

    @property
    def relu_output(self) -> int:
        return CONDITIONS[self.condition][1]


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
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def safe_float(value: float) -> str:
    return format(value, ".8g").replace("-", "m").replace(".", "p")


def run_id(config: TrainConfig) -> str:
    return (
        f"{config.condition}__wd-{safe_float(config.weight_decay)}"
        f"__dgp-{config.dgp_seed}__init-{config.init_seed}"
        f"__train-{config.train_seed}__eval-{config.eval_seed}"
    )


def script_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def code_sha256_manifest() -> dict[str, str]:
    paths = [Path(__file__).resolve(), UTILS_PATH]
    return {
        path.relative_to(PROJECT_ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def scientific_config(config: TrainConfig | dict[str, Any]) -> dict[str, Any]:
    """Return settings that define the numerical experiment."""
    values = asdict(config) if isinstance(config, TrainConfig) else dict(config)
    values.pop("output_root", None)
    values.pop("resume", None)
    return values


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_manifest(device: torch.device) -> dict[str, Any]:
    manifest = {
        "script_sha256": script_sha256(),
        "code_sha256": code_sha256_manifest(),
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
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        manifest["cuda_device_name"] = torch.cuda.get_device_name(device)
    return manifest


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(tmp, path)


def save_torch_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, tmp)
    os.replace(tmp, path)


class BatchNormScaleOnly(torch.nn.BatchNorm1d):
    def __init__(self, num_features: int):
        super().__init__(num_features, affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.batch_norm(
            x,
            self.running_mean * 0,
            self.running_var,
            None,
            None,
            self.training,
            self.momentum,
            self.eps,
        )


class ReLUEncoder(torch.nn.Module):
    def __init__(self, base: torch.nn.Module):
        super().__init__()
        self.base = base
        self.activation = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.base(x))


@dataclass
class GenerativeProcess:
    cluster_vectors: torch.Tensor
    sample_latents: Callable[[int], tuple[torch.Tensor, torch.Tensor]]
    mixing: torch.nn.Module


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dgp_config(config: TrainConfig) -> dict[str, Any]:
    """Settings that uniquely define the generated latent process and mixing."""
    return {
        "latent_family": "sparse" if config.expon else "hypersphere",
        "dgp_seed": config.dgp_seed,
        "n": config.n,
        "nr_clusters": config.nr_clusters,
        "n_mixing_layers": config.n_mixing_layers,
        "mixing_search_steps": config.mixing_search_steps,
        "c_param": config.c_param,
        "expon_entropy": config.expon_entropy,
        "expon_scale": config.expon_scale,
        "cluster_fit_steps": config.cluster_fit_steps,
        "deterministic": config.deterministic,
    }


def fit_sparse_cluster_vectors(config: TrainConfig, device: torch.device) -> torch.Tensor:
    logits = torch.randn(config.nr_clusters, config.n, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=1e-3)
    if config.expon_entropy > 0:
        initial = torch.softmax(logits, dim=1)
        target_log_product = torch.log(initial).sum(dim=1).mean().detach()
    else:
        target_log_product = torch.tensor(config.expon_entropy, device=device)

    for _ in range(config.cluster_fit_steps):
        cluster_vectors = F.softplus(logits)
        log_product = torch.log(cluster_vectors).sum(dim=1)
        loss = ((log_product - target_log_product) ** 2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        cluster_vectors = F.softplus(logits).detach()
    return cluster_vectors


def assemble_generative_process(
    config: TrainConfig,
    device: torch.device,
    cluster_vectors: torch.Tensor,
    mixing: torch.nn.Module,
) -> GenerativeProcess:
    if config.expon:
        def sample_latents(size: int) -> tuple[torch.Tensor, torch.Tensor]:
            labels = torch.randint(0, config.nr_clusters, (size,), device=device)
            rates = config.expon_scale / cluster_vectors[labels]
            samples = torch.distributions.Exponential(rate=rates).sample()
            return samples, labels

    else:
        sphere = NSphereSpace(config.n)
        cluster_vectors = sphere.uniform(config.nr_clusters, device=device)

        def sample_latents(size: int) -> tuple[torch.Tensor, torch.Tensor]:
            labels = torch.randint(0, config.nr_clusters, (size,), device=device)
            means = cluster_vectors[labels]
            samples = sphere.von_mises_fisher(
                means, config.c_param, size=size, device=device
            )
            return samples, labels

    mixing.eval()
    for parameter in mixing.parameters():
        parameter.requires_grad = False
    return GenerativeProcess(cluster_vectors, sample_latents, mixing)


def build_mixing_architecture(config: TrainConfig, device: torch.device) -> torch.nn.Module:
    """Build the fixed LeakyReLU mixing topology without random matrix search."""
    layers: list[torch.nn.Module] = []
    for index in range(config.n_mixing_layers):
        layers.append(torch.nn.Linear(config.n, config.n, bias=False))
        if index < config.n_mixing_layers - 1:
            layers.append(torch.nn.LeakyReLU(negative_slope=0.2))
    return torch.nn.Sequential(*layers).to(device)


def create_generative_process(config: TrainConfig, device: torch.device) -> GenerativeProcess:
    set_seed(config.dgp_seed, config.deterministic)

    if config.expon:
        cluster_vectors = fit_sparse_cluster_vectors(config, device)
    else:
        sphere = NSphereSpace(config.n)
        cluster_vectors = sphere.uniform(config.nr_clusters, device=device)

    mixing = construct_invertible_mlp(
        n=config.n,
        n_layers=config.n_mixing_layers,
        act_fct="leaky_relu",
        cond_thresh_ratio=0.0,
        n_iter_cond_thresh=config.mixing_search_steps,
    ).to(device)
    return assemble_generative_process(config, device, cluster_vectors, mixing)


def prepare_dgp_artifact(config: TrainConfig, artifact_dir: Path) -> Path:
    artifact_dir = artifact_dir.resolve()
    artifact_path = artifact_dir / "dgp.pt"
    manifest_path = artifact_dir / "manifest.json"
    if artifact_path.exists() or manifest_path.exists():
        if not artifact_path.exists() or not manifest_path.exists():
            raise FileExistsError(f"Incomplete DGP artifact directory: {artifact_dir}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("dgp_config") != dgp_config(config):
            raise ValueError(f"Existing DGP artifact has incompatible settings: {artifact_dir}")
        if manifest.get("artifact_sha256") != file_sha256(artifact_path):
            raise ValueError(f"DGP artifact checksum mismatch: {artifact_path}")
        print(f"DGP artifact already exists: {artifact_dir}")
        return artifact_dir

    artifact_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.device)
    dgp = create_generative_process(config, device)
    save_torch_atomic(
        artifact_path,
        {
            "cluster_vectors": dgp.cluster_vectors.detach().cpu(),
            "mixing_state": {
                key: value.detach().cpu() for key, value in dgp.mixing.state_dict().items()
            },
        },
    )
    write_json(
        manifest_path,
        {
            "format_version": 1,
            "dgp_config": dgp_config(config),
            "artifact_sha256": file_sha256(artifact_path),
            "construction_environment": environment_manifest(device),
        },
    )
    print(f"Saved DGP artifact: {artifact_dir}")
    return artifact_dir


def load_dgp_artifact(
    config: TrainConfig, artifact_dir: Path, device: torch.device
) -> GenerativeProcess:
    artifact_dir = artifact_dir.resolve()
    artifact_path = artifact_dir / "dgp.pt"
    manifest_path = artifact_dir / "manifest.json"
    if not artifact_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"Missing DGP artifact under {artifact_dir}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format_version") != 1:
        raise ValueError(f"Unsupported DGP artifact format: {manifest_path}")
    if manifest.get("dgp_config") != dgp_config(config):
        raise ValueError(f"DGP artifact settings do not match this run: {artifact_dir}")
    actual_sha256 = file_sha256(artifact_path)
    if manifest.get("artifact_sha256") != actual_sha256:
        raise ValueError(f"DGP artifact checksum mismatch: {artifact_path}")

    payload = torch.load(artifact_path, map_location="cpu", weights_only=True)
    cluster_vectors = payload["cluster_vectors"].to(device)
    mixing = build_mixing_architecture(config, device)
    mixing.load_state_dict(payload["mixing_state"], strict=True)
    return assemble_generative_process(config, device, cluster_vectors, mixing)


def dgp_artifact_provenance(artifact_dir: Path) -> dict[str, Any]:
    artifact_dir = artifact_dir.resolve()
    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    return {
        "artifact_dir": str(artifact_dir),
        "artifact_sha256": manifest["artifact_sha256"],
        "dgp_config": manifest["dgp_config"],
    }


def create_encoder(config: TrainConfig, device: torch.device) -> torch.nn.Module:
    if config.n_hidden_layers < 2:
        raise ValueError("n_hidden_layers must be at least 2")
    hidden = (
        [config.n * 10]
        + [config.n * 50] * (config.n_hidden_layers - 2)
        + [config.n * 10]
    )
    base = get_mlp(
        n_in=config.n,
        n_out=config.nh,
        layers=hidden,
        layer_normalization="bn" if config.use_hidden_batch_norm else None,
        output_normalization=None,
    )
    model: torch.nn.Module = ReLUEncoder(base) if config.relu_output else base
    return model.to(device)


def correlation_matrix(z: np.ndarray, hz: np.ndarray) -> np.ndarray:
    raw = np.corrcoef(z.T, hz.T)
    corr = raw[: z.shape[1], z.shape[1] :]
    return np.nan_to_num(corr, nan=0.0)


def evaluate(
    config: TrainConfig,
    z_eval: torch.Tensor,
    mixing: torch.nn.Module,
    encoder: torch.nn.Module,
) -> tuple[dict[str, float], np.ndarray, torch.Tensor]:
    encoder.eval()
    with torch.inference_mode():
        hz = encoder(mixing(z_eval))
    z_cpu = z_eval.detach().cpu()
    hz_cpu = hz.detach().cpu()
    # At very early steps a ReLU coordinate can be constant.  Pearson
    # correlation is then mathematically undefined; the legacy evaluator and
    # ``correlation_matrix`` deliberately map that entry to zero.  Suppress the
    # associated NumPy divide warning so large sweeps retain useful logs.
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = correlation_matrix(z_cpu.numpy(), hz_cpu.numpy())
        mcc, _ = disentanglement_score(
            z_cpu, hz_cpu, mode="pearson", reorder=True, nonlinear=False
        )
        (hz_to_z, _), _ = linear_disentanglement(
            z_cpu, hz_cpu, mode="pearson", train_test_split=True
        )
        (z_to_hz, _), _ = linear_disentanglement(
            hz_cpu, z_cpu, mode="pearson", train_test_split=True
        )
    activation_energy = float((hz_cpu**2).sum(dim=1).mean())
    dead_fraction = float((hz_cpu.abs().max(dim=0).values < 1e-12).float().mean())
    metrics = {
        "hz_to_z": float(hz_to_z),
        "z_to_hz": float(z_to_hz),
        "mcc": float(mcc),
        "activation_energy": activation_energy,
        "dead_unit_fraction": dead_fraction,
    }
    return metrics, corr, hz_cpu


def parameter_l2(module: torch.nn.Module) -> float:
    total = 0.0
    with torch.no_grad():
        for parameter in module.parameters():
            total += float((parameter.detach().float() ** 2).sum().cpu())
    return math.sqrt(total)


METRIC_FIELDS = [
    "step",
    "loss",
    "hz_to_z",
    "z_to_hz",
    "mcc",
    "activation_energy",
    "dead_unit_fraction",
    "encoder_parameter_l2",
    "readout_parameter_l2",
    "elapsed_seconds",
]


def append_metric(path: Path, row: dict[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({name: row[name] for name in METRIC_FIELDS})
        stream.flush()
        os.fsync(stream.fileno())


def truncate_metrics_after(path: Path, max_step: int) -> None:
    """Make an append-only metric log consistent with a restored checkpoint."""
    if not path.exists():
        return
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    retained = [row for row in rows if int(float(row["step"])) <= max_step]
    if len(retained) == len(rows):
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(retained)
        stream.flush()
        os.fsync(stream.fileno())
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


def train_one(
    config: TrainConfig,
    prepared_dgp: GenerativeProcess | None = None,
    dgp_artifact_dir: Path | None = None,
) -> Path:
    if config.condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {config.condition}")
    if not (0 < config.threshold <= 1):
        raise ValueError("threshold must lie in (0, 1]")
    if config.log_every <= 0 or config.checkpoint_every <= 0 or config.max_steps <= 0:
        raise ValueError("log_every, checkpoint_every, and max_steps must be positive")

    device = resolve_device(config.device)
    set_seed(config.dgp_seed, config.deterministic)
    output_root = Path(config.output_root).resolve()
    run_dir = output_root / run_id(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    if config_path.exists():
        saved_config = json.loads(config_path.read_text())
        if scientific_config(saved_config) != scientific_config(config):
            raise ValueError(
                f"Existing run has incompatible settings: {run_dir}. "
                "Use a different --output-root."
            )
    completed_path = run_dir / "COMPLETED.json"
    if completed_path.exists():
        if dgp_artifact_dir is not None:
            expected_dgp = dgp_artifact_provenance(dgp_artifact_dir)
            source_path = run_dir / "dgp_source.json"
            if not source_path.exists():
                raise ValueError(f"Completed run lacks DGP provenance: {run_dir}")
            saved_dgp = json.loads(source_path.read_text())
            if (
                saved_dgp.get("artifact_sha256") != expected_dgp["artifact_sha256"]
                or saved_dgp.get("dgp_config") != expected_dgp["dgp_config"]
            ):
                raise ValueError(f"Completed run used a different DGP artifact: {run_dir}")
        print(f"Already complete: {run_dir}")
        return run_dir
    if any(run_dir.iterdir()) and not config.resume:
        raise FileExistsError(
            f"Non-empty incomplete run directory: {run_dir}. Use --resume."
        )

    write_json(config_path, asdict(config))
    write_json(run_dir / "environment.json", environment_manifest(device))

    if prepared_dgp is not None:
        dgp = prepared_dgp
    elif dgp_artifact_dir is not None:
        dgp = load_dgp_artifact(config, dgp_artifact_dir, device)
    else:
        dgp = create_generative_process(config, device)
    save_torch_atomic(run_dir / "cluster_vectors.pt", dgp.cluster_vectors.cpu())
    save_torch_atomic(run_dir / "generator.pt", dgp.mixing.state_dict())
    if dgp_artifact_dir is not None:
        write_json(
            run_dir / "dgp_source.json",
            {"source": "prepared_artifact", **dgp_artifact_provenance(dgp_artifact_dir)},
        )
    else:
        write_json(
            run_dir / "dgp_source.json",
            {"source": "generated_in_process", "dgp_config": dgp_config(config)},
        )

    set_seed(config.eval_seed, config.deterministic)
    z_eval, eval_labels = dgp.sample_latents(config.eval_samples)
    save_torch_atomic(run_dir / "eval_latents.pt", z_eval.detach().cpu())
    save_torch_atomic(run_dir / "eval_labels.pt", eval_labels.detach().cpu())

    set_seed(config.init_seed, config.deterministic)
    encoder = create_encoder(config, device)
    readout = torch.nn.Linear(config.nh, config.nr_clusters, bias=False).to(device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(readout.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    start_step = 0
    first_threshold_step: int | None = None
    best_mcc = -math.inf
    restored_interval_loss = 0.0
    restored_interval_count = 0
    checkpoint_path = run_dir / "checkpoint.pt"
    if config.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        encoder.load_state_dict(checkpoint["encoder"])
        readout.load_state_dict(checkpoint["readout"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        first_threshold_step = checkpoint["first_threshold_step"]
        best_mcc = float(checkpoint["best_mcc"])
        restored_interval_loss = float(checkpoint.get("interval_loss", 0.0))
        restored_interval_count = int(checkpoint.get("interval_count", 0))
        restore_rng_state(checkpoint["rng_state"])
    else:
        set_seed(config.train_seed, config.deterministic)

    metrics_path = run_dir / "metrics.csv"
    if config.resume:
        truncate_metrics_after(metrics_path, start_step)
    started = time.perf_counter()
    interval_loss = torch.tensor(
        restored_interval_loss, device=device, dtype=torch.float64
    )
    interval_count = restored_interval_count
    last_metrics: dict[str, float] | None = None
    last_corr: np.ndarray | None = None
    last_hz: torch.Tensor | None = None

    encoder.train()
    readout.train()
    for step in range(start_step + 1, config.max_steps + 1):
        z_batch, labels = dgp.sample_latents(config.batch_size)
        logits = readout(encoder(dgp.mixing(z_batch)))
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        interval_loss.add_(loss.detach().to(dtype=torch.float64))
        interval_count += 1

        should_log = step % config.log_every == 0 or step == config.max_steps
        if should_log:
            eval_metrics, last_corr, last_hz = evaluate(
                config, z_eval, dgp.mixing, encoder
            )
            last_metrics = eval_metrics
            best_mcc = max(best_mcc, eval_metrics["mcc"])
            if first_threshold_step is None and eval_metrics["mcc"] >= config.threshold:
                first_threshold_step = step
            row = {
                "step": step,
                "loss": float((interval_loss / max(interval_count, 1)).cpu()),
                **eval_metrics,
                "encoder_parameter_l2": parameter_l2(encoder),
                "readout_parameter_l2": parameter_l2(readout),
                "elapsed_seconds": time.perf_counter() - started,
            }
            append_metric(metrics_path, row)
            print(
                f"{run_id(config)} step={step}/{config.max_steps} "
                f"loss={row['loss']:.4f} mcc={row['mcc']:.4f}"
            )
            interval_loss.zero_()
            interval_count = 0
            encoder.train()

        should_checkpoint = step % config.checkpoint_every == 0 or step == config.max_steps
        if should_checkpoint:
            save_torch_atomic(
                checkpoint_path,
                {
                    "step": step,
                    "encoder": encoder.state_dict(),
                    "readout": readout.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "first_threshold_step": first_threshold_step,
                    "best_mcc": best_mcc,
                    "interval_loss": float(interval_loss.cpu()),
                    "interval_count": interval_count,
                    "rng_state": capture_rng_state(),
                },
            )

    if last_metrics is None or last_corr is None or last_hz is None:
        last_metrics, last_corr, last_hz = evaluate(config, z_eval, dgp.mixing, encoder)
        best_mcc = max(best_mcc, last_metrics["mcc"])

    save_torch_atomic(run_dir / "model.pt", encoder.state_dict())
    save_torch_atomic(run_dir / "readout.pt", readout.state_dict())
    save_torch_atomic(run_dir / "final_activations.pt", last_hz)
    np.savetxt(run_dir / "final_correlation.csv", last_corr, delimiter=",")
    summary = {
        "run_id": run_id(config),
        "condition": config.condition,
        "weight_decay": config.weight_decay,
        "dgp_seed": config.dgp_seed,
        "init_seed": config.init_seed,
        "train_seed": config.train_seed,
        "eval_seed": config.eval_seed,
        "threshold": config.threshold,
        "max_steps": config.max_steps,
        "converged": first_threshold_step is not None,
        "first_threshold_step": first_threshold_step,
        "final_mcc": last_metrics["mcc"],
        "best_mcc": best_mcc,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(completed_path, summary)
    return run_dir


def parse_int_spec(spec: str) -> list[int]:
    if ":" in spec:
        start, stop = (int(value) for value in spec.split(":", maxsplit=1))
        return list(range(start, stop))
    return [int(value) for value in spec.split(",") if value]


def parse_float_spec(spec: str) -> list[float]:
    return [float(value) for value in spec.split(",") if value]


def read_revised_run(
    config_path: Path, source: str = "revised_run"
) -> dict[str, Any] | None:
    """Read one completed run written by the ``train`` subcommand."""
    run_dir = config_path.parent
    metrics_path = run_dir / "metrics.csv"
    completed_path = run_dir / "COMPLETED.json"
    if not metrics_path.exists() or not completed_path.exists():
        return None
    config = json.loads(config_path.read_text())
    completed = json.loads(completed_path.read_text())
    environment_path = run_dir / "environment.json"
    environment = (
        json.loads(environment_path.read_text()) if environment_path.exists() else {}
    )
    dgp_source_path = run_dir / "dgp_source.json"
    dgp_source = (
        json.loads(dgp_source_path.read_text()) if dgp_source_path.exists() else {}
    )
    with metrics_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return None
    steps = np.asarray([int(float(row["step"])) for row in rows], dtype=int)
    metrics = {
        key: np.asarray([float(row[key]) for row in rows], dtype=float)
        for key in ("hz_to_z", "z_to_hz", "mcc")
    }
    pairing_files = (
        "cluster_vectors.pt",
        "generator.pt",
        "eval_latents.pt",
        "eval_labels.pt",
    )
    if not all((run_dir / name).exists() for name in pairing_files):
        return None
    pairing_sha256 = {
        name: file_sha256(run_dir / name) for name in pairing_files
    }
    return {
        "source": source,
        "run_dir": str(run_dir),
        "args": config,
        "condition": config["condition"],
        "seed": int(config["init_seed"]),
        "weight_decay": float(config["weight_decay"]),
        "steps": steps,
        "expected_steps": int(config["max_steps"]),
        "completed": completed,
        "environment": environment,
        "dgp_source": dgp_source,
        "pairing_sha256": pairing_sha256,
        **metrics,
    }


def load_revised_root(
    root: Path, source: str = "revised_run"
) -> list[dict[str, Any]]:
    runs = []
    if not root.exists():
        return runs
    for config_path in sorted(root.glob("*/config.json")):
        run = read_revised_run(config_path, source)
        if run is not None:
            runs.append(run)
    return runs


def revised_pairing_signature(run: dict[str, Any]) -> str:
    """Identify settings that must be fixed in a paired sensitivity sweep."""
    config = scientific_config(run["args"])
    for key in ("init_seed", "weight_decay"):
        config.pop(key, None)
    config["paired_artifacts_sha256"] = run["pairing_sha256"]
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def select_balanced_revised_runs(
    runs: list[dict[str, Any]],
    weight_decays: tuple[float, ...] = (0.0, 0.003, 0.01, 0.03),
) -> list[dict[str, Any]]:
    """Return the largest complete, paired initialization grid if available."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        if run["condition"] == "sparse_relu":
            grouped.setdefault(revised_pairing_signature(run), []).append(run)

    best: list[dict[str, Any]] = []
    for group in grouped.values():
        seeds_by_wd = {
            wd: {
                int(run["seed"])
                for run in group
                if math.isclose(float(run["weight_decay"]), wd, abs_tol=1e-12)
            }
            for wd in weight_decays
        }
        common_seeds = set.intersection(*(seeds_by_wd[wd] for wd in weight_decays))
        if not common_seeds:
            continue
        selected = [
            run
            for run in group
            if int(run["seed"]) in common_seeds
            and any(
                math.isclose(float(run["weight_decay"]), wd, abs_tol=1e-12)
                for wd in weight_decays
            )
        ]
        if len(selected) > len(best):
            best = selected
    return sorted(best, key=lambda run: (float(run["weight_decay"]), int(run["seed"])))


def select_revised_baseline_runs(
    baseline_candidates: list[dict[str, Any]],
    reliability_runs: list[dict[str, Any]],
    expected_seed_count: int,
) -> list[dict[str, Any]]:
    """Assemble and validate the four revised weight-decay-zero conditions."""
    expected_seeds = set(range(expected_seed_count))
    allowed_baseline_conditions = set(CONDITIONS) - {"sparse_relu"}
    unexpected = [
        run
        for run in baseline_candidates
        if run["condition"] not in allowed_baseline_conditions
        or not math.isclose(float(run["weight_decay"]), 0.0, abs_tol=1e-12)
    ]
    if unexpected:
        raise ValueError(
            "The revised baseline root must contain only weight-decay-zero "
            "hypersphere_linear, hypersphere_relu, and sparse_linear runs."
        )

    sparse_relu_zero = [
        run
        for run in reliability_runs
        if run["condition"] == "sparse_relu"
        and math.isclose(float(run["weight_decay"]), 0.0, abs_tol=1e-12)
    ]
    assembled = [*baseline_candidates, *sparse_relu_zero]
    for condition in CONDITIONS:
        group = [run for run in assembled if run["condition"] == condition]
        seeds = {int(run["seed"]) for run in group}
        if len(group) != expected_seed_count or seeds != expected_seeds:
            raise ValueError(
                f"Expected {expected_seed_count} revised weight-decay-zero runs "
                f"for {condition} with seeds {sorted(expected_seeds)}, found "
                f"{len(group)} runs with seeds {sorted(seeds)}."
            )
        if any(int(run["steps"][-1]) != int(run["expected_steps"]) for run in group):
            raise ValueError(f"Incomplete optimization budget for {condition}.")

    for family, conditions in (
        ("hypersphere", ("hypersphere_linear", "hypersphere_relu")),
        ("sparse", ("sparse_linear", "sparse_relu")),
    ):
        group = [run for run in assembled if run["condition"] in conditions]
        dgp_hashes = {
            (
                run["pairing_sha256"]["cluster_vectors.pt"],
                run["pairing_sha256"]["generator.pt"],
            )
            for run in group
        }
        if len(dgp_hashes) != 1:
            raise ValueError(
                f"Expected one shared {family} DGP, found "
                f"{len(dgp_hashes)} distinct saved DGPs."
            )

    return sorted(assembled, key=lambda run: (run["condition"], int(run["seed"])))


def first_crossing(run: dict[str, Any], threshold: float) -> int | None:
    matches = np.flatnonzero(run["mcc"] >= threshold)
    return int(run["steps"][matches[0]]) if len(matches) else None


def run_summary_row(run: dict[str, Any], threshold: float) -> dict[str, Any]:
    hit = first_crossing(run, threshold)
    # Some legacy logger versions explicitly evaluated the penultimate update
    # but not the final update. Treat expected_steps - 1 as a completed budget.
    full_budget = int(run["steps"][-1]) >= int(run["expected_steps"]) - 1
    return {
        "source": run["source"],
        "condition": run["condition"],
        "seed": run["seed"],
        "weight_decay": run["weight_decay"],
        "threshold": threshold,
        "expected_steps": run["expected_steps"],
        "last_logged_step": int(run["steps"][-1]),
        "full_budget": full_budget,
        "converged": hit is not None,
        "first_threshold_step": hit,
        "final_hz_to_z": float(run["hz_to_z"][-1]),
        "final_z_to_hz": float(run["z_to_hz"][-1]),
        "final_mcc": float(run["mcc"][-1]),
        "best_mcc": float(np.max(run["mcc"])),
    }


def percentile(values: Iterable[float], q: float) -> float | None:
    values = list(values)
    return float(np.percentile(values, q)) if values else None


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    p = successes / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    spread = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return max(0.0, center - spread), min(1.0, center + spread)


def condition_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["condition"], row["weight_decay"]), []).append(row)
    summaries = []
    for (condition, weight_decay), group in sorted(grouped.items()):
        hits = [
            float(row["first_threshold_step"])
            for row in group
            if row["first_threshold_step"] is not None
        ]
        full_finals = [float(row["final_mcc"]) for row in group if row["full_budget"]]
        successes = len(hits)
        low, high = wilson_interval(successes, len(group))
        summaries.append(
            {
                "condition": condition,
                "weight_decay": weight_decay,
                "n_runs": len(group),
                "n_converged": successes,
                "success_fraction": successes / len(group),
                "success_ci95_low": low,
                "success_ci95_high": high,
                "median_time_to_threshold": percentile(hits, 50),
                "time_to_threshold_q25": percentile(hits, 25),
                "time_to_threshold_q75": percentile(hits, 75),
                "n_full_budget": len(full_finals),
                "final_mcc_median_full_budget": percentile(full_finals, 50),
                "final_mcc_q25_full_budget": percentile(full_finals, 25),
                "final_mcc_q75_full_budget": percentile(full_finals, 75),
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stack_run_metric(
    runs: list[dict[str, Any]], metric: str
) -> tuple[np.ndarray, np.ndarray]:
    if not runs:
        raise ValueError("No runs to stack")
    runs = sorted(runs, key=lambda run: run["seed"])
    common_steps = runs[0]["steps"]
    for run in runs[1:]:
        common_steps = np.intersect1d(common_steps, run["steps"])
    values = []
    for run in runs:
        index = {int(step): idx for idx, step in enumerate(run["steps"])}
        values.append(np.asarray([run[metric][index[int(step)]] for step in common_steps]))
    return common_steps, np.stack(values)


def plot_training_curves(
    ax: plt.Axes,
    runs_by_condition: dict[str, list[dict[str, Any]]],
    conditions: tuple[str, str],
    metric: str,
    title: str,
    ylabel: str | None,
) -> None:
    for condition, color in zip(conditions, (LINEAR_COLOR, RELU_COLOR)):
        steps, values = stack_run_metric(runs_by_condition[condition], metric)
        for trace in values:
            ax.plot(steps, trace, color=color, alpha=SEED_TRACE_ALPHA, linewidth=0.65)
        mean = values.mean(axis=0)
        sem = values.std(axis=0, ddof=0) / math.sqrt(values.shape[0])
        ax.plot(
            steps,
            mean,
            color=color,
            linewidth=2.0,
            label=f"{CONDITION_LABELS[condition]} (n={values.shape[0]})",
        )
        ax.fill_between(steps, mean - sem, mean + sem, color=color, alpha=0.18)
    ax.set_xscale("log")
    ax.set_xlim(left=max(1, min(run["steps"][0] for run in sum(runs_by_condition.values(), []))))
    ax.set_ylim(0, 1.03)
    ax.set_title(title)
    ax.set_xlabel("Training step")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(loc="lower right", frameon=False, fontsize=8)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.19,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
    )


def draw_final_mcc_panel(
    ax: plt.Axes,
    reliability_rows: list[dict[str, Any]],
    weight_decays: list[float],
    threshold: float,
) -> None:
    groups = {
        wd: [
            row
            for row in reliability_rows
            if row["weight_decay"] == wd and row["full_budget"]
        ]
        for wd in weight_decays
    }
    positions = np.arange(len(weight_decays))
    seed_maps = {wd: {int(row["seed"]): row for row in groups[wd]} for wd in weight_decays}
    common_seeds = sorted(set.intersection(*(set(seed_maps[wd]) for wd in weight_decays)))
    for seed in common_seeds:
        values = [seed_maps[wd][seed]["final_mcc"] for wd in weight_decays]
        ax.plot(positions, values, color="0.75", linewidth=0.7, zorder=1)
    for position, wd in zip(positions, weight_decays):
        group = groups[wd]
        values = np.asarray([row["final_mcc"] for row in group], dtype=float)
        ax.scatter(
            np.full(len(values), position),
            values,
            s=24,
            color=RELU_COLOR,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        median = float(np.median(values))
        q25, q75 = np.percentile(values, [25, 75])
        ax.vlines(position, q25, q75, color="black", linewidth=2.5, zorder=4)
        ax.hlines(median, position - 0.15, position + 0.15, color="black", linewidth=2)
        successes = sum(bool(row["converged"]) for row in group)
        ax.text(
            position,
            1.012,
            f"{successes}/{len(group)} reached",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )
    ax.axhline(threshold, color="0.35", linestyle="--", linewidth=0.9)
    ax.text(
        0.02,
        threshold + 0.005,
        "threshold",
        transform=ax.get_yaxis_transform(),
        color="0.35",
        fontsize=7,
        va="bottom",
    )
    ax.set_xticks(positions, [f"{wd:g}" for wd in weight_decays])
    ax.set_xlim(-0.36, len(weight_decays) - 1 + 0.36)
    ax.set_xlabel("Weight decay")
    ax.set_ylabel("Final MCC")
    ax.set_ylim(0.70, 1.035)
    ax.set_title("E  Optimization reliability", loc="left", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)


def draw_time_to_threshold_panel(
    ax: plt.Axes, reliability_rows: list[dict[str, Any]], threshold: float
) -> None:
    weight_decays = [0.0, 0.003, 0.01, 0.03]
    rng = np.random.default_rng(0)
    budgets = sorted({float(row["expected_steps"]) for row in reliability_rows})
    if len(budgets) != 1:
        raise ValueError(f"Expected one shared optimization budget, found {budgets}")
    budget = budgets[0]
    ax.axhline(budget, color="0.55", linestyle=":", linewidth=0.8, zorder=0)
    ax.text(
        0.985,
        budget,
        "budget",
        transform=ax.get_yaxis_transform(),
        color="0.4",
        fontsize=7,
        ha="right",
        va="bottom",
    )
    for position, wd in enumerate(weight_decays):
        group = [row for row in reliability_rows if row["weight_decay"] == wd]
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
                s=22,
                color=RELU_COLOR,
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
            )
            q25, median, q75 = np.percentile(hits, [25, 50, 75])
            ax.vlines(position, q25, q75, color="black", linewidth=2.5, zorder=4)
            ax.hlines(median, position - 0.16, position + 0.16, color="black", linewidth=2)
        if misses:
            miss_budgets = [float(row["expected_steps"]) for row in misses]
            jitter = rng.uniform(-0.10, 0.10, size=len(misses))
            ax.scatter(
                position + jitter,
                miss_budgets,
                marker="x",
                s=28,
                color="#B33A3A",
                linewidth=1.2,
                zorder=4,
            )
        ax.text(
            position,
            1.035,
            f"{len(hits)}/{len(group)}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=7.5,
        )
    ax.set_yscale("log")
    ax.set_ylim(8e3, 1.35e6)
    ax.set_xticks(np.arange(len(weight_decays)), [f"{wd:g}" for wd in weight_decays])
    ax.set_xlim(-0.36, len(weight_decays) - 1 + 0.36)
    ax.set_xlabel("Weight decay")
    ax.set_ylabel(f"Steps to MCC ≥ {threshold:.2f}")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=RELU_COLOR,
                   markeredgecolor="white", label="Converged"),
            Line2D([0], [0], marker="x", color="#B33A3A", linestyle="none",
                   label="Not converged"),
        ],
        loc="lower left",
        frameon=False,
        fontsize=7.5,
    )


def write_caption(
    path: Path,
    summaries: list[dict[str, Any]],
    threshold: float,
    reliability_source: str,
) -> None:
    sparse = {
        float(row["weight_decay"]): row
        for row in summaries
        if row["condition"] == "sparse_relu"
    }
    lines = [
        "**Figure 1. Linear and axis-aligned latent recovery, with optimization reliability.**",
        "Panels A--D use the revised weight-decay-zero experiments for all four full-capacity conditions and show individual seed trajectories in addition to mean ± s.e.m. over ten initialization seeds.",
        f"Panel E quantifies finite-budget optimization reliability for the sparse DGP with a ReLU output using a prespecified MCC threshold of {threshold:.2f}.",
        "The upper panel reports final MCC; gray lines connect the same initialization across weight-decay conditions.",
        "The lower panel reports the first step at which the threshold was reached. Points are individual random seeds; black bars show the median and interquartile range (among successful runs for time-to-threshold). Crosses at the budget line denote runs that did not reach the threshold.",
    ]
    if reliability_source == "revised_balanced":
        lines.append(
            "Panel E uses a paired initialization/weight-decay sweep with the DGP, training batches, evaluation set, and optimization budget held fixed."
        )
    for wd in sorted(sparse):
        row = sparse[wd]
        median = row["median_time_to_threshold"]
        median_text = "undefined" if median is None else f"{median:,.0f} steps"
        lines.append(
            f"At weight decay {wd:g}, {row['n_converged']}/{row['n_runs']} runs reached the threshold; median time among converged runs was {median_text}."
        )
    if reliability_source == "revised_balanced" and 0.0 in sparse and 0.03 in sparse:
        zero_successes = sparse[0.0]["n_converged"]
        regularized_successes = sparse[0.03]["n_converged"]
        if regularized_successes > zero_successes:
            success_verb = "increased"
        elif regularized_successes < zero_successes:
            success_verb = "decreased"
        else:
            success_verb = "was unchanged"
        zero_median = sparse[0.0]["median_time_to_threshold"]
        regularized_median = sparse[0.03]["median_time_to_threshold"]
        zero_median_text = (
            "undefined" if zero_median is None else f"{zero_median:,.0f} steps"
        )
        regularized_median_text = (
            "undefined"
            if regularized_median is None
            else f"{regularized_median:,.0f} steps"
        )
        lines.append(
            f"Thus, with weight decay 0.03 the success rate {success_verb} from "
            f"{zero_successes}/{sparse[0.0]['n_runs']} to "
            f"{regularized_successes}/{sparse[0.03]['n_runs']}; "
            "the median time-to-threshold among successful runs changed from "
            f"{zero_median_text} to {regularized_median_text}."
        )
    path.write_text("\n\n".join(lines) + "\n")


def plot_revised_figure(args: argparse.Namespace) -> None:
    load_plotting()
    baseline_candidates = load_revised_root(
        Path(args.baseline_root), "revised_baseline"
    )
    revised_runs = load_revised_root(
        Path(args.reliability_root), "revised_reliability"
    )
    reliability_runs = select_balanced_revised_runs(revised_runs)
    revised_seed_count = len({int(run["seed"]) for run in reliability_runs})
    expected_reliability_runs = args.minimum_revised_seeds * 4
    if (
        revised_seed_count != args.minimum_revised_seeds
        or len(reliability_runs) != expected_reliability_runs
    ):
        raise ValueError(
            "Expected a complete revised 4-condition weight-decay grid with "
            f"{args.minimum_revised_seeds} paired seeds under "
            f"{Path(args.reliability_root).resolve()}, found "
            f"{len(reliability_runs)} runs and {revised_seed_count} paired seeds."
        )
    reliability_source = "revised_balanced"
    paired_seeds = sorted({int(run["seed"]) for run in reliability_runs})
    baseline = select_revised_baseline_runs(
        baseline_candidates,
        reliability_runs,
        args.minimum_revised_seeds,
    )
    runs_by_condition = {
        condition: sorted(
            [run for run in baseline if run["condition"] == condition],
            key=lambda run: run["seed"],
        )
        for condition in CONDITIONS
    }
    print(
        f"Using revised A--D baseline: {len(baseline)} runs, "
        f"initialization seeds={paired_seeds}, weight_decay=0"
    )
    print(
        f"Using revised paired E sweep: {len(reliability_runs)} runs, "
        f"initialization seeds={paired_seeds}"
    )

    baseline_rows = [run_summary_row(run, args.threshold) for run in baseline]
    reliability_rows = [run_summary_row(run, args.threshold) for run in reliability_runs]
    all_rows = baseline_rows + [
        row
        for row in reliability_rows
        if not (
            row["condition"] == "sparse_relu"
            and math.isclose(float(row["weight_decay"]), 0.0, abs_tol=1e-12)
        )
    ]
    summaries = condition_summary_rows(reliability_rows)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "runs.csv", all_rows)
    write_csv(output_dir / "conditions.csv", summaries)
    all_source_runs = [*baseline_candidates, *reliability_runs]
    source_script_hashes = sorted(
        {
            run["environment"]["script_sha256"]
            for run in all_source_runs
            if run.get("environment", {}).get("script_sha256")
        }
    )
    source_code_manifests = [
        json.loads(serialized)
        for serialized in sorted(
            {
                json.dumps(run["environment"]["code_sha256"], sort_keys=True)
                for run in all_source_runs
                if run.get("environment", {}).get("code_sha256")
            }
        )
    ]
    paired_artifact_hashes = {
        name: sorted(
            {
                run["pairing_sha256"][name]
                for run in reliability_runs
                if run.get("pairing_sha256", {}).get(name)
            }
        )
        for name in (
            "cluster_vectors.pt",
            "generator.pt",
            "eval_latents.pt",
            "eval_labels.pt",
        )
    }
    write_json(
        output_dir / "manifest.json",
        {
            "plot_script_sha256": script_sha256(),
            "plot_code_sha256": code_sha256_manifest(),
            "script_sha256": script_sha256(),
            "code_sha256": code_sha256_manifest(),
            "source_run_script_sha256": source_script_hashes,
            "source_run_code_sha256": source_code_manifests,
            "paired_artifact_sha256": paired_artifact_hashes,
            "threshold": args.threshold,
            "baseline_root": str(Path(args.baseline_root).resolve()),
            "reliability_root": str(Path(args.reliability_root).resolve()),
            "baseline_source": "revised_weight_decay_zero",
            "reliability_source": reliability_source,
            "baseline_runs": len(baseline),
            "baseline_root_runs": len(baseline_candidates),
            "baseline_sparse_relu_runs_from_reliability": len(
                runs_by_condition["sparse_relu"]
            ),
            "reliability_runs": len(reliability_runs),
        },
    )

    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(10.6, 6.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=(1.0, 1.0, 1.06))
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
    )
    plot_training_curves(
        ax_b,
        runs_by_condition,
        ("sparse_linear", "sparse_relu"),
        "z_to_hz",
        "Sparse DGP",
        None,
    )
    plot_training_curves(
        ax_c,
        runs_by_condition,
        ("hypersphere_linear", "hypersphere_relu"),
        "mcc",
        "Hypersphere DGP",
        "Axis alignment (MCC)",
    )
    plot_training_curves(
        ax_d,
        runs_by_condition,
        ("sparse_linear", "sparse_relu"),
        "mcc",
        "Sparse DGP",
        None,
    )
    add_panel_label(ax_a, "A")
    add_panel_label(ax_b, "B")
    add_panel_label(ax_c, "C")
    add_panel_label(ax_d, "D")
    final_weight_decays = [0.0, 0.003, 0.01, 0.03]
    draw_final_mcc_panel(ax_e1, reliability_rows, final_weight_decays, args.threshold)
    draw_time_to_threshold_panel(ax_e2, reliability_rows, args.threshold)

    stem = output_dir / "fig7"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    write_caption(
        output_dir / "caption.md",
        summaries,
        args.threshold,
        reliability_source,
    )
    print(f"Saved {stem.with_suffix('.pdf')}")
    print(f"Saved {stem.with_suffix('.png')}")


def add_train_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_condition: bool = True,
    include_dgp_path: bool = True,
) -> None:
    if include_condition:
        parser.add_argument("--condition", choices=sorted(CONDITIONS), default="sparse_relu")
    else:
        parser.set_defaults(condition="sparse_relu")
    parser.add_argument(
        "--output-root", default=str(PROJECT_ROOT / "Data" / "fig7" / "reliability")
    )
    parser.add_argument("--dgp-seed", type=int, default=20260728)
    parser.add_argument("--init-seed", type=int, default=0)
    parser.add_argument("--train-seed", type=int, default=20260729)
    parser.add_argument("--eval-seed", type=int, default=20260730)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--nh", type=int, default=5)
    parser.add_argument("--nr-clusters", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--eval-samples", type=int, default=4096)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=10_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--n-mixing-layers", type=int, default=4)
    parser.add_argument("--mixing-search-steps", type=int, default=25_000)
    parser.add_argument("--n-hidden-layers", type=int, default=6)
    parser.add_argument("--no-hidden-batch-norm", action="store_true")
    parser.add_argument("--c-param", type=float, default=10.0)
    parser.add_argument("--expon-entropy", type=float, default=1.0)
    parser.add_argument("--expon-scale", type=float, default=0.1)
    parser.add_argument("--cluster-fit-steps", type=int, default=20_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--non-deterministic", action="store_true")
    parser.add_argument("--resume", action="store_true")
    if include_dgp_path:
        parser.add_argument(
            "--dgp-path",
            help="Read-only DGP artifact directory created by prepare-dgp.",
        )


def config_from_args(args: argparse.Namespace, **updates: Any) -> TrainConfig:
    values = {
        "condition": args.condition,
        "output_root": args.output_root,
        "dgp_seed": args.dgp_seed,
        "init_seed": args.init_seed,
        "train_seed": args.train_seed,
        "eval_seed": args.eval_seed,
        "n": args.n,
        "nh": args.nh,
        "nr_clusters": args.nr_clusters,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "eval_samples": args.eval_samples,
        "log_every": args.log_every,
        "checkpoint_every": args.checkpoint_every,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "threshold": args.threshold,
        "n_mixing_layers": args.n_mixing_layers,
        "mixing_search_steps": args.mixing_search_steps,
        "n_hidden_layers": args.n_hidden_layers,
        "use_hidden_batch_norm": not args.no_hidden_batch_norm,
        "c_param": args.c_param,
        "expon_entropy": args.expon_entropy,
        "expon_scale": args.expon_scale,
        "cluster_fit_steps": args.cluster_fit_steps,
        "device": args.device,
        "deterministic": not args.non_deterministic,
        "resume": args.resume,
    }
    values.update(updates)
    return TrainConfig(**values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plot_parser = subparsers.add_parser(
        "plot", help="Create revised Fig. 1 entirely from the new audited runs"
    )
    plot_parser.add_argument(
        "--baseline-root",
        default=str(
            PROJECT_ROOT / "Data" / "fig7" / "baseline"
        ),
        help=(
            "Thirty revised weight-decay-zero runs: ten each for "
            "hypersphere_linear, hypersphere_relu, and sparse_linear."
        ),
    )
    plot_parser.add_argument(
        "--reliability-root",
        default=str(
            PROJECT_ROOT / "Data" / "fig7" / "reliability"
        ),
        help=(
            "Forty revised sparse-ReLU runs spanning ten paired initialization "
            "seeds and four weight-decay values."
        ),
    )
    plot_parser.add_argument(
        "--minimum-revised-seeds",
        type=int,
        default=10,
        help="Minimum number of fully paired initialization seeds required (default: 10).",
    )
    plot_parser.add_argument(
        "--output-dir", default=str(PROJECT_ROOT / "Data" / "fig7" / "report")
    )
    plot_parser.add_argument("--threshold", type=float, default=0.90)

    train_parser = subparsers.add_parser("train", help="Run one revised experiment")
    add_train_arguments(train_parser)

    sweep_parser = subparsers.add_parser("sweep", help="Run paired initialization/WD grid")
    add_train_arguments(sweep_parser, include_condition=False)
    sweep_parser.add_argument("--init-seeds", default="0:10")
    sweep_parser.add_argument("--weight-decays", default="0,0.003,0.01,0.03")

    baseline_parser = subparsers.add_parser(
        "baseline-sweep",
        help="Run the three zero-weight-decay baseline conditions",
    )
    add_train_arguments(baseline_parser, include_condition=False)
    baseline_parser.set_defaults(
        output_root=str(PROJECT_ROOT / "Data" / "fig7" / "baseline")
    )
    baseline_parser.add_argument("--init-seeds", default="0:10")

    dgp_parser = subparsers.add_parser(
        "prepare-dgp", help="Create one reusable DGP artifact"
    )
    add_train_arguments(
        dgp_parser,
        include_condition=True,
        include_dgp_path=False,
    )
    dgp_parser.add_argument("--artifact-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "plot":
        plot_revised_figure(args)
        return
    if args.command == "prepare-dgp":
        prepare_dgp_artifact(config_from_args(args), Path(args.artifact_dir))
        return
    if args.command == "train":
        artifact_dir = Path(args.dgp_path) if args.dgp_path else None
        train_one(config_from_args(args), dgp_artifact_dir=artifact_dir)
        return
    if args.command == "baseline-sweep":
        conditions = ("hypersphere_linear", "hypersphere_relu", "sparse_linear")
        init_seeds = parse_int_spec(args.init_seeds)
        for condition in conditions:
            for init_seed in init_seeds:
                train_one(
                    config_from_args(
                        args,
                        condition=condition,
                        init_seed=init_seed,
                        weight_decay=0.0,
                    )
                )
        return
    if args.command == "sweep":
        init_seeds = parse_int_spec(args.init_seeds)
        weight_decays = parse_float_spec(args.weight_decays)
        print(
            f"Running {len(init_seeds) * len(weight_decays)} paired runs: "
            f"init_seeds={init_seeds}, weight_decays={weight_decays}"
        )
        configs = [
            config_from_args(
                args,
                condition="sparse_relu",
                init_seed=init_seed,
                weight_decay=weight_decay,
            )
            for weight_decay in weight_decays
            for init_seed in init_seeds
        ]
        artifact_dir = Path(args.dgp_path) if args.dgp_path else None
        pending_configs = []
        for config in configs:
            candidate = Path(config.output_root).resolve() / run_id(config)
            if (candidate / "COMPLETED.json").exists():
                train_one(config, dgp_artifact_dir=artifact_dir)
            else:
                pending_configs.append(config)
        if pending_configs:
            device = resolve_device(pending_configs[0].device)
            shared_dgp = (
                load_dgp_artifact(pending_configs[0], artifact_dir, device)
                if artifact_dir is not None
                else create_generative_process(pending_configs[0], device)
            )
            for config in pending_configs:
                train_one(
                    config,
                    prepared_dgp=shared_dgp,
                    dgp_artifact_dir=artifact_dir,
                )
        return
    raise AssertionError(args.command)


if __name__ == "__main__":
    main()
