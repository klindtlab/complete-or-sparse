"""Shared numerical building blocks for the simulation scripts.

The functions here are the small subset of the original ``cl-ica`` helpers
that is actually used by Figures 1, 7, and 8. Keeping them in one module makes
the public reproduction code self-contained without duplicate ``ref`` trees.
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
import torch
from scipy import stats
from scipy.optimize import linear_sum_assignment
from scipy.stats import ortho_group
from sklearn import linear_model, metrics
from torch import nn


Metric = Literal["r2", "adjusted_r2", "pearson", "spearman", "mae"]


class RescaleLayer(nn.Module):
    """Project each row onto a sphere with a fixed or learned radius."""

    def __init__(self, init_r: float = 1.0, fixed_r: bool = False, mode: str = "eq"):
        super().__init__()
        if mode not in {"eq", "leq"}:
            raise ValueError(f"unknown sphere projection mode: {mode}")
        self.mode = mode
        radius = torch.ones(1) * init_r
        self.r = radius if fixed_r else nn.Parameter(radius)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        radius = self.r.to(values.device)
        norms = torch.linalg.norm(values, dim=-1, keepdim=True)
        if self.mode == "eq":
            return values / norms * radius
        scale = torch.maximum(torch.ones_like(norms), norms / radius)
        return values / scale


class SoftclipLayer(nn.Module):
    """Map unconstrained values into a fixed or learned positive box."""

    def __init__(self, n: int, init_abs_bound: float = 1.0, fixed_abs_bound: bool = True):
        super().__init__()
        bound = torch.ones(n) * init_abs_bound
        self.max_abs_bound = bound if fixed_abs_bound else nn.Parameter(bound)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(values) * self.max_abs_bound.to(values.device).unsqueeze(0)


def get_mlp(
    n_in: int,
    n_out: int,
    layers: Sequence[int],
    layer_normalization: str | None = None,
    output_normalization: str | None = None,
    output_normalization_kwargs: dict | None = None,
) -> nn.Sequential:
    """Construct the encoder MLP used by the simulation experiments."""
    widths = [*layers, n_out]
    if not widths:
        if n_in != n_out:
            raise ValueError("an empty MLP requires n_in == n_out")
        return nn.Sequential(nn.Identity())

    modules: list[nn.Module] = []
    previous = n_in
    for index, width in enumerate(widths):
        modules.append(nn.Linear(previous, width))
        if index < len(widths) - 1:
            if layer_normalization == "bn":
                modules.append(nn.BatchNorm1d(width))
            elif layer_normalization == "gn":
                modules.append(nn.GroupNorm(1, width))
            elif layer_normalization is not None:
                raise ValueError(f"unknown layer normalization: {layer_normalization}")
            modules.append(nn.LeakyReLU())
        previous = width

    kwargs = output_normalization_kwargs or {}
    if output_normalization == "fixed_sphere":
        modules.append(RescaleLayer(fixed_r=True, **kwargs))
    elif output_normalization == "learnable_sphere":
        modules.append(RescaleLayer(init_r=1.0, fixed_r=False, **kwargs))
    elif output_normalization == "fixed_box":
        modules.append(SoftclipLayer(n_out, fixed_abs_bound=True, **kwargs))
    elif output_normalization == "learnable_box":
        modules.append(SoftclipLayer(n_out, fixed_abs_bound=False, **kwargs))
    elif output_normalization is not None:
        raise ValueError(f"unknown output normalization: {output_normalization}")
    return nn.Sequential(*modules)


def construct_invertible_mlp(
    n: int = 20,
    n_layers: int = 2,
    n_iter_cond_thresh: int = 10_000,
    cond_thresh_ratio: float = 0.25,
    weight_matrix_init: str = "pcl",
    act_fct: str = "leaky_relu",
) -> nn.Sequential:
    """Create the fixed approximately invertible nonlinear mixing network."""

    class SmoothLeakyReLU(nn.Module):
        def __init__(self, alpha: float = 0.2):
            super().__init__()
            self.alpha = alpha

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return self.alpha * values + (1 - self.alpha) * torch.log1p(torch.exp(values))

    activations: dict[str, tuple[type[nn.Module], dict[str, float]]] = {
        "relu": (nn.ReLU, {}),
        "leaky_relu": (nn.LeakyReLU, {"negative_slope": 0.2}),
        "elu": (nn.ELU, {"alpha": 1.0}),
        "smooth_leaky_relu": (SmoothLeakyReLU, {"alpha": 0.2}),
        "softplus": (nn.Softplus, {"beta": 1}),
    }
    if act_fct not in activations:
        raise ValueError(f"unknown activation: {act_fct}")
    activation, activation_kwargs = activations[act_fct]

    def normalize_columns(matrix: np.ndarray) -> np.ndarray:
        return matrix / np.sqrt(np.sum(matrix * matrix, axis=0))

    if weight_matrix_init == "pcl":
        condition_numbers = np.zeros(n_iter_cond_thresh)
        for index in range(n_iter_cond_thresh):
            candidate = normalize_columns(np.random.uniform(-1, 1, (n, n)))
            condition_numbers[index] = np.linalg.cond(candidate)
        condition_numbers.sort()
        threshold = condition_numbers[int(n_iter_cond_thresh * cond_thresh_ratio)]
        print(f"condition number threshold: {threshold:f}")

    modules: list[nn.Module] = []
    for index in range(n_layers):
        linear = nn.Linear(n, n, bias=False)
        if weight_matrix_init == "pcl":
            condition = threshold + 1
            while condition > threshold:
                matrix = normalize_columns(np.random.uniform(-1, 1, (n, n)))
                condition = np.linalg.cond(matrix)
            print(f"layer {index + 1}/{n_layers},  condition number: {condition}")
        elif weight_matrix_init == "rvs":
            matrix = ortho_group.rvs(n)
        else:
            raise ValueError(f"unknown weight initialization: {weight_matrix_init}")
        linear.weight.data = torch.tensor(matrix, dtype=torch.float32)
        modules.append(linear)
        if index < n_layers - 1:
            modules.append(activation(**activation_kwargs))

    network = nn.Sequential(*modules)
    for parameter in network.parameters():
        parameter.requires_grad = False
    return network


def _sample_vmf_weight(kappa: float, dim: int, count: int) -> np.ndarray:
    dim = dim - 1
    b = dim / (np.sqrt(4.0 * kappa**2 + dim**2) + 2 * kappa)
    x = (1.0 - b) / (1.0 + b)
    c = kappa * x + dim * np.log(1 - x**2)
    accepted: list[np.ndarray] = []
    total = 0
    while total < count:
        z = np.random.beta(dim / 2.0, dim / 2.0, size=count)
        weights = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
        uniform = np.random.uniform(size=count)
        mask = kappa * weights + dim * np.log(1.0 - x * weights) - c >= np.log(uniform)
        accepted.append(weights[mask])
        total += int(mask.sum())
    return np.concatenate(accepted)[:count]


def sample_vmf(mean: np.ndarray, kappa: float, count: int) -> np.ndarray:
    """Sample from a von Mises--Fisher distribution around row-wise means."""
    if mean.ndim == 1:
        mean = np.repeat(mean[None, :], count, axis=0)
    weights = _sample_vmf_weight(kappa, mean.shape[1], count)
    random_vectors = np.random.randn(*mean.shape)
    projection = mean * np.einsum("ij,ij->i", mean, random_vectors)[:, None]
    projection /= np.linalg.norm(mean, axis=-1, keepdims=True)
    orthogonal = random_vectors - projection
    orthogonal /= np.linalg.norm(orthogonal, axis=-1, keepdims=True)
    return orthogonal * np.sqrt(1.0 - weights**2)[:, None] + weights[:, None] * mean


def sample_generalized_normal(
    mean: torch.Tensor, scale: float, power: int, shape: tuple[int, ...]
) -> torch.Tensor:
    inverse_power = 1.0 / power
    gamma = torch.distributions.Gamma(inverse_power, 1.0).rsample(shape)
    signs = torch.randint(0, 2, shape, dtype=mean.dtype) * 2 - 1
    return mean + scale * signs.to(mean.device) * torch.abs(gamma).pow(inverse_power).to(mean.device)


class NSphereSpace:
    """Sampling operations on the unit sphere in ``R^n``."""

    def __init__(self, n: int, radius: float = 1.0):
        self.n = n
        self.radius = radius

    def uniform(self, size: int, device: str | torch.device = "cpu") -> torch.Tensor:
        values = torch.randn((size, self.n), device=device)
        return values / torch.linalg.norm(values, dim=-1, keepdim=True)

    def normal(
        self, mean: torch.Tensor, std: float | torch.Tensor, size: int,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        mean = mean.to(device)
        values = torch.randn((size, self.n), device=device) * std + mean
        return values / torch.linalg.norm(values, dim=-1, keepdim=True)

    def laplace(
        self, mean: torch.Tensor, scale: float, size: int,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        mean = mean.to(device)
        values = torch.distributions.Laplace(torch.zeros(self.n), scale).rsample((size,)).to(device) + mean
        return values / torch.linalg.norm(values, dim=-1, keepdim=True)

    def generalized_normal(
        self, mean: torch.Tensor, scale: float, power: int, size: int,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        values = sample_generalized_normal(mean.to(device), scale, power, (size, self.n))
        return values / torch.linalg.norm(values, dim=-1, keepdim=True)

    def von_mises_fisher(
        self, mean: torch.Tensor, kappa: float, size: int,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        mean_np = mean.detach().cpu().numpy()
        if mean_np.ndim == 1:
            mean_np = np.repeat(mean_np[None, :], size, axis=0)
        if not np.allclose(np.linalg.norm(mean_np, axis=-1), self.radius):
            raise ValueError("von Mises--Fisher means must lie on the sphere")
        return torch.tensor(sample_vmf(mean_np, kappa, size), dtype=torch.float32, device=device)


def normalize_columns(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array).copy()
    values -= values.mean(axis=0, keepdims=True)
    values /= np.linalg.norm(values, axis=0, keepdims=True)
    return values


def disentanglement_score(
    truth: np.ndarray | torch.Tensor,
    estimate: np.ndarray | torch.Tensor,
    mode: Metric = "r2",
    reorder: bool = False,
    nonlinear: bool = False,
) -> tuple[float, float | None]:
    """Score direct or permutation-and-scaling recovery between two codes."""
    if torch.is_tensor(truth):
        truth = truth.detach().cpu().numpy()
    if torch.is_tensor(estimate):
        estimate = estimate.detach().cpu().numpy()
    truth = np.asarray(truth)
    estimate = np.asarray(estimate)

    if mode == "r2":
        return float(metrics.r2_score(truth, estimate)), None
    if mode == "adjusted_r2":
        r2 = metrics.r2_score(truth, estimate)
        n, p = truth.shape
        return float(1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)), None
    if mode == "mae":
        return float(metrics.mean_absolute_error(truth, estimate)), None
    if mode not in {"pearson", "spearman"}:
        raise ValueError(f"unknown metric: {mode}")

    dim = truth.shape[-1]
    raw = stats.spearmanr(truth, estimate).statistic if mode == "spearman" else np.corrcoef(truth.T, estimate.T)
    correlation = np.asarray(raw)[:dim, dim:]
    correlation[np.isnan(correlation)] = 0
    if not reorder:
        score = float(np.diag(np.abs(correlation)).mean())
        return score, score

    rows, columns = linear_sum_assignment(-np.abs(correlation))
    score = float(np.mean(np.abs(correlation[rows, columns])))
    nonlinear_score = score
    if nonlinear and estimate.min() > 0:
        reordered = np.zeros_like(estimate)
        for row, column in zip(rows, columns):
            reordered[:, row] = estimate[:, column]
        reordered = reordered.clip(min=0)
        truth_normalized = normalize_columns(truth)
        nonlinear_score = float(
            max(
                np.mean(np.abs(np.sum(truth_normalized * normalize_columns(reordered**power), axis=0)))
                for power in np.linspace(0.1, 4, 100)
            )
        )
    return score, nonlinear_score


def linear_disentanglement(
    truth: np.ndarray | torch.Tensor,
    estimate: np.ndarray | torch.Tensor,
    mode: Metric = "r2",
    train_test_split: bool = False,
) -> tuple[tuple[float, float | None], tuple[np.ndarray, np.ndarray]]:
    """Fit an unrestricted linear readout and score held-out reconstruction."""
    if torch.is_tensor(truth):
        truth = truth.detach().cpu().numpy()
    if torch.is_tensor(estimate):
        estimate = estimate.detach().cpu().numpy()
    truth = np.asarray(truth)
    estimate = np.asarray(estimate)
    split = len(truth) // 2 if train_test_split else len(truth)
    train_truth = truth[:split]
    train_estimate = estimate[:split]
    test_truth = truth[split:] if train_test_split else truth
    test_estimate = estimate[split:] if train_test_split else estimate
    model = linear_model.LinearRegression().fit(train_estimate, train_truth)
    prediction = model.predict(test_estimate)
    return disentanglement_score(test_truth, prediction, mode=mode), (test_truth, prediction)
