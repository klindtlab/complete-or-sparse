"""Minimal Machine Interpretability Score implementation for Figure 4.

This file contains only the MIS computation and precomputed-embedding metric
used by ``Codes/fig4-cross_system_comparison/vision.ipynb``. It is adapted from
https://github.com/klindtlab/MachInterp under the MIT License; see
``Codes/MACHINTERP_LICENSE``.
"""

from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


class PrecomputedEmbeddingMetric:
    """Cosine similarity over a fixed matrix of precomputed embeddings."""

    precomputed = True
    num_scores = 1

    def __init__(self, embeddings):
        self.embeddings = torch.as_tensor(embeddings).detach().cpu()
        if self.embeddings.ndim != 2:
            raise ValueError(
                "embeddings must have shape (samples, features), got "
                f"{tuple(self.embeddings.shape)}"
            )

    def precomputed_similarity(self, ind_batch_a, ind_batch_b):
        ind_batch_a = torch.as_tensor(ind_batch_a, dtype=torch.int64)
        ind_batch_b = torch.as_tensor(ind_batch_b, dtype=torch.int64)
        embed_a = self.embeddings[ind_batch_a]
        embed_b = self.embeddings[ind_batch_b]
        return F.cosine_similarity(
            embed_a[:, None], embed_b[None, :], dim=-1
        ).numpy()


def randomized_argsort(array):
    """Sort an array, breaking equal-value ties randomly."""
    array = np.asarray(array)
    return np.lexsort((np.random.rand(array.shape[0]), array))


def get_extreme_ind(activations, quantile):
    """Select values outside the central quantile interval."""
    return np.logical_or(
        activations <= np.quantile(activations, quantile),
        activations >= np.quantile(activations, 1 - quantile),
    )


def extract_logits(similarities, zscore, pool_fun):
    num_references = similarities.shape[0] // 2
    if zscore:
        similarities = similarities.copy()
        similarities -= np.mean(similarities)
        std = np.std(similarities)
        if np.isnan(std) or std == 0.0:
            std = 1.0
        similarities /= std

    logits = np.zeros((2, 2))
    logits[0, 0] = pool_fun(similarities[:num_references, 0])
    logits[0, 1] = pool_fun(similarities[num_references:, 1])
    logits[1, 0] = pool_fun(similarities[:num_references, 1])
    logits[1, 1] = pool_fun(similarities[num_references:, 0])
    return logits


def compute_accuracy(logits):
    actual_logits = logits.sum(-1)
    return np.mean(actual_logits[:, :, 0] - actual_logits[:, :, 1] > 0, axis=1)


def run_single_unit_psychophysics(
    inputs,
    activations,
    metrics,
    quantiles: Optional[List[float]] = None,
    num_trials=20,
    num_references=9,
    pool_fun=np.mean,
    zscore=True,
    seed=42,
):
    """Run the MIS psychophysics task for one activation unit."""
    if quantiles is None:
        quantiles = [0.0, 0.25]
    if activations.ndim != 1:
        raise ValueError(
            f"Activations must be a vector, but have shape {list(activations.shape)}."
        )

    np.random.seed(seed)
    output = {
        f"logits_{key}": np.zeros((len(quantiles), num_trials, 2, 2))
        for key in metrics
    }

    for quantile_index, quantile in enumerate(quantiles):
        extreme = get_extreme_ind(activations, quantile)
        ranked_values = activations.copy()
        ranked_values[extreme] = np.median(ranked_values)
        ind_sort_bottom = randomized_argsort(ranked_values)
        ind_sort_top = ind_sort_bottom[::-1]
        ind_query_top = np.random.permutation(ind_sort_top[:num_trials])
        ind_query_bottom = np.random.permutation(ind_sort_bottom[:num_trials])
        reference_offsets = num_trials + np.arange(num_references) * num_trials

        for trial_index in range(num_trials):
            ind_reference = np.concatenate(
                [
                    ind_sort_top[reference_offsets + trial_index],
                    ind_sort_bottom[reference_offsets + trial_index],
                ]
            )
            ind_query = np.array(
                [ind_query_top[trial_index], ind_query_bottom[trial_index]]
            )

            for key, metric in metrics.items():
                similarities = metric.precomputed_similarity(
                    ind_reference, ind_query
                )
                output[f"logits_{key}"][
                    quantile_index, trial_index
                ] = extract_logits(similarities, zscore, pool_fun)

    for key in list(output):
        output["accuracy" + key[6:]] = compute_accuracy(output[key])
    return output


def compute_score(
    inputs,
    activations,
    metrics,
    quantiles: Optional[List[float]] = None,
    num_trials=50,
    num_references=9,
    pool_fun=np.mean,
    zscore=True,
    seed=42,
):
    """Run the Machine Interpretability Score over every activation unit."""
    if quantiles is None:
        quantiles = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    if activations.ndim != 2:
        raise ValueError(
            f"Activations must be a matrix, but have shape {list(activations.shape)}."
        )

    num_data, num_units = activations.shape
    if inputs.shape[0] != num_data:
        raise ValueError("Input and activations must have the same first dimension.")
    if not all(q1 <= q2 for q1, q2 in zip(quantiles, quantiles[1:])):
        raise ValueError("Quantiles must be in ascending order.")
    if quantiles[0] < 0.0 or quantiles[-1] > 0.5:
        raise ValueError("Quantiles must lie between 0.0 and 0.5.")

    required = num_trials * (num_references + 1) * 2
    if num_data < required:
        max_trials = num_data // ((num_references + 1) * 2)
        raise ValueError(
            "Not enough data for the specified number of trials and references. "
            f"At most num_trials={max_trials}."
        )

    result = {"quantiles": np.asarray(quantiles)}
    for unit_index in tqdm(range(num_units)):
        output = run_single_unit_psychophysics(
            inputs=inputs,
            activations=activations[:, unit_index],
            metrics=metrics,
            quantiles=quantiles,
            num_trials=num_trials,
            num_references=num_references,
            pool_fun=pool_fun,
            zscore=zscore,
            seed=seed,
        )
        for key, value in output.items():
            result.setdefault(key, []).append(value)

    for key in list(result):
        if key != "quantiles":
            result[key] = np.stack(result[key], axis=0)
    return result
