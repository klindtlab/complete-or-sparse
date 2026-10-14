#!/usr/bin/env python3
"""Prepare the public Majaj2015 IT data and LPIPS similarities for Figure 4E.

The repository does not redistribute the stimuli or neural recordings.  Point
``--responses`` and ``--animals`` to the two public IT arrays and
``--images-dir`` to the 3,200 stimulus images in matching order.  The script
writes only generated files below ``Data/fig4/macaque``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "Data" / "fig4" / "macaque"


def load_array(path: Path, preferred: str) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    payload = loadmat(path)
    if preferred in payload:
        return np.asarray(payload[preferred])
    arrays = [value for key, value in payload.items() if not key.startswith("__")]
    if len(arrays) != 1:
        raise ValueError(f"Could not identify {preferred!r} in {path}")
    return np.asarray(arrays[0])


def image_paths(directory: Path) -> list[Path]:
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    paths = sorted(path for path in directory.rglob("*") if path.suffix.lower() in suffixes)
    if len(paths) != 3200:
        raise ValueError(f"Expected 3,200 stimuli, found {len(paths)} in {directory}")
    return paths


def lpips_similarity(
    paths: list[Path],
    batch_size: int,
    pair_batch_size: int,
    device: str,
    output: Path,
) -> None:
    """Compute the full LPIPS matrix through cached weighted feature vectors.

    LPIPS is a channel-weighted squared distance at each normalized AlexNet
    feature layer. Caching those features turns millions of pairwise network
    evaluations into five block matrix multiplications.
    """
    import lpips as lpips_package
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image
    from torchvision.transforms import functional as tvf

    class Images(Dataset):
        def __len__(self) -> int:
            return len(paths)

        def __getitem__(self, index: int):
            image = Image.open(paths[index]).convert("RGB")
            return tvf.to_tensor(image).mul(2).sub(1)

    model = lpips_package.LPIPS(net="alex").to(device).eval()
    loader = DataLoader(Images(), batch_size=batch_size, shuffle=False, num_workers=2)
    feature_files: list[Path] = []
    feature_arrays: list[np.memmap] = []
    layer_weights: list[np.ndarray] = []
    offset = 0
    with torch.no_grad():
        for images in loader:
            raw_features = model.net.forward(model.scaling_layer(images.to(device)))
            weighted = []
            for layer, raw in enumerate(raw_features):
                normalized = lpips_package.normalize_tensor(raw)
                linear = next(
                    module
                    for module in model.lins[layer].modules()
                    if isinstance(module, nn.Conv2d)
                )
                vector = normalized.flatten(1).cpu().numpy().astype(np.float32)
                weighted.append(vector)
                if not feature_arrays:
                    spatial_size = raw.shape[-2] * raw.shape[-1]
                    weights = linear.weight.detach().cpu().numpy().reshape(-1)
                    layer_weights.append(
                        np.repeat(weights / spatial_size, spatial_size).astype(np.float32)
                    )
            if not feature_arrays:
                for layer, vector in enumerate(weighted):
                    path = output / f".lpips-layer-{layer}.npy"
                    feature_files.append(path)
                    feature_arrays.append(
                        np.lib.format.open_memmap(
                            path,
                            mode="w+",
                            dtype=np.float32,
                            shape=(len(paths), vector.shape[1]),
                        )
                    )
            stop = offset + len(images)
            for array, vector in zip(feature_arrays, weighted):
                array[offset:stop] = vector
            offset = stop
    for array in feature_arrays:
        array.flush()
    del feature_arrays

    count = len(paths)
    similarity = np.lib.format.open_memmap(
        output / "lpips-similarity.npy",
        mode="w+",
        dtype=np.float32,
        shape=(count, count),
    )
    similarity[:] = 0
    for path, weights_array in zip(feature_files, layer_weights):
        features = np.load(path, mmap_mode="r")
        weights = torch.from_numpy(weights_array).to(device)
        for row_start in range(0, count, pair_batch_size):
            row_stop = min(count, row_start + pair_batch_size)
            left = torch.from_numpy(np.asarray(features[row_start:row_stop], dtype=np.float32)).to(device)
            weighted_left = left * weights
            left_norm = (weighted_left * left).sum(dim=1, keepdim=True)
            for column_start in range(row_start, count, pair_batch_size):
                column_stop = min(count, column_start + pair_batch_size)
                right = torch.from_numpy(np.asarray(features[column_start:column_stop], dtype=np.float32)).to(device)
                right_norm = ((right * weights) * right).sum(dim=1)[None, :]
                distance = left_norm + right_norm - 2 * (weighted_left @ right.T)
                values = -distance.cpu().numpy().astype(np.float32)
                similarity[row_start:row_stop, column_start:column_stop] += values
                if column_start != row_start:
                    similarity[column_start:column_stop, row_start:row_stop] += values.T
        del features
        path.unlink()
    np.fill_diagonal(similarity, 0)
    similarity.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--animals", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--pair-batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    responses = load_array(args.responses, "neural_responses").astype(np.float32)
    animals = np.char.strip(load_array(args.animals, "animals").astype(str).reshape(-1))
    if responses.shape != (3200, 168):
        raise ValueError(f"Expected IT response shape (3200, 168), got {responses.shape}")
    counts = {animal: int(np.sum(animals == animal)) for animal in np.unique(animals)}
    if counts != {"Chabo": 58, "Tito": 110}:
        raise ValueError(f"Unexpected animal labels: {counts}")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "majaj2015-it.npz", responses=responses, animals=animals)
    lpips_similarity(
        image_paths(args.images_dir),
        args.batch_size,
        args.pair_batch_size,
        args.device,
        output,
    )
    print(f"Saved prepared Figure 4E inputs to {output}")


if __name__ == "__main__":
    main()
