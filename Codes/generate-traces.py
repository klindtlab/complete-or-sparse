#!/usr/bin/env python3
"""Generate the simulation traces used by Figures 1, 5, and 6.

The three figures share the same data-generating process and training code.
This wrapper records the exact grids used in the paper and sends every output
to ``Data/``.  Model checkpoints are omitted because none of the plotters read
them; pass ``--save-checkpoints`` only for additional debugging.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "Codes" / "fig1&5&6-axis_aligned_neurons" / "train.py"


def parse_seeds(value: str) -> list[int]:
    if ":" in value:
        start, stop = (int(part) for part in value.split(":", 1))
        return list(range(start, stop))
    return [int(part) for part in value.split(",") if part]


def configurations(figure: str) -> tuple[Path, int, list[dict[str, object]]]:
    if figure == "fig1":
        grid = [
            {"expon": expon, "relu_enc": activation}
            for expon in (0, 1)
            for activation in (0, 1)
        ]
        return ROOT / "Data" / "fig1" / "runs", 100_000, grid
    if figure == "fig5":
        grid = [
            {"expon": expon, "expon_enc": activation}
            for expon in (0, 1)
            for activation in (0, 1)
        ]
        return ROOT / "Data" / "fig5" / "runs", 50_000, grid
    if figure == "fig6":
        return (
            ROOT / "Data" / "fig6" / "weight-decay",
            100_000,
            [{"expon": 1, "relu_enc": 1, "weight_decay": 0.03}],
        )
    raise ValueError(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figure", choices=("fig1", "fig5", "fig6"))
    parser.add_argument("--seeds", default="0:10", help="Half-open range or CSV list.")
    parser.add_argument("--n-epochs", type=int, help="Override the paper training budget.")
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output, paper_epochs, grid = configurations(args.figure)
    epochs = args.n_epochs or paper_epochs
    for seed in parse_seeds(args.seeds):
        for setting in grid:
            command = [
                sys.executable,
                str(TRAIN),
                "--seed",
                str(seed),
                "--n-epochs",
                str(epochs),
                "--save_folder",
                str(output),
            ]
            for name, value in setting.items():
                command.extend([f"--{name}", str(value)])
            if args.save_checkpoints:
                command.append("--save-checkpoints")
            print(" ".join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
