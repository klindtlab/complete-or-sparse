#!/usr/bin/env python3
"""Recover the exact plotted Fig. 4A-D points from the archived vector PDFs.

The original activation matrices are several gigabytes. Matplotlib stored every
scatter point as a vector marker in the panel PDFs, so this one-time curation
step recovers the plot-level MIS/MMU table without retaining those large,
licensed-data-derived intermediates.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from pypdf import PdfReader
from pypdf.generic import ContentStream


ROOT = Path(__file__).resolve().parents[2]


def multiply(left: tuple[float, ...], right: tuple[float, ...]) -> tuple[float, ...]:
    a, b, c, d, e, f = left
    g, h, i, j, k, l = right
    return (
        a * g + c * h,
        b * g + d * h,
        a * i + c * j,
        b * i + d * j,
        a * k + c * l + e,
        b * k + d * l + f,
    )


def marker_positions(path: Path) -> dict[int, list[tuple[float, float]]]:
    reader = PdfReader(path)
    if len(reader.pages) != 1:
        raise ValueError(f"Expected one-page panel PDF: {path}")
    stream = ContentStream(reader.pages[0]["/Contents"].get_object(), reader)
    identity = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    stack = [identity]
    positions: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for operands, operator in stream.operations:
        if operator == b"q":
            stack.append(stack[-1])
        elif operator == b"Q":
            if len(stack) == 1:
                raise ValueError(f"Unbalanced graphics state in {path}")
            stack.pop()
        elif operator == b"cm":
            matrix = tuple(float(value) for value in operands)
            stack[-1] = multiply(stack[-1], matrix)
        elif operator == b"Do":
            name = str(operands[0])
            if name.startswith("/M") and name[2:].isdigit():
                positions[int(name[2:])].append((stack[-1][4], stack[-1][5]))
    return dict(positions)


def pearson(xs: list[float], ys: list[float]) -> float:
    mean_x = math.fsum(xs) / len(xs)
    mean_y = math.fsum(ys) / len(ys)
    dx = [value - mean_x for value in xs]
    dy = [value - mean_y for value in ys]
    numerator = math.fsum(x * y for x, y in zip(dx, dy))
    denominator = math.sqrt(
        math.fsum(value * value for value in dx)
        * math.fsum(value * value for value in dy)
    )
    return numerator / denominator


def linear_map(
    point: tuple[float, float],
    x_tick_0: tuple[float, float],
    x_tick_1: tuple[float, float],
    y_tick_0: tuple[float, float],
    y_tick_1: tuple[float, float],
) -> tuple[float, float]:
    page_x, page_y = point
    x_page_0, x_data_0 = x_tick_0
    x_page_1, x_data_1 = x_tick_1
    y_page_0, y_data_0 = y_tick_0
    y_page_1, y_data_1 = y_tick_1
    x = x_data_0 + (page_x - x_page_0) * (x_data_1 - x_data_0) / (
        x_page_1 - x_page_0
    )
    y = y_data_0 + (page_y - y_page_0) * (y_data_1 - y_data_0) / (
        y_page_1 - y_page_0
    )
    return x, y


def extract_vision(path: Path) -> list[dict[str, object]]:
    models = ["vgg19", "resnet50", "vit-b16", "dinov2-s14"]
    axes_x0 = [57.72, 267.5, 477.28, 687.06]
    positions = marker_positions(path)
    rows: list[dict[str, object]] = []
    marker = 0
    for source_index, source in enumerate(models):
        targets = [model for index, model in enumerate(models) if index != source_index]
        for target in targets:
            for unit, point in enumerate(positions[marker]):
                mis, mmu = linear_map(
                    point,
                    (axes_x0[source_index], 0.4),
                    (axes_x0[source_index] + 39.555, 0.5),
                    (51.96, 0.0),
                    (94.163077, 0.2),
                )
                rows.append(
                    {"source": source, "target": target, "unit": unit, "mis": mis, "mmu": mmu}
                )
            marker += 1
    return rows


def extract_language(path: Path) -> list[dict[str, object]]:
    models = ["gpt2", "pythia-70m", "tinyllama"]
    transforms = [
        ((90.000781, 0.0), (136.952126, 0.02), (71.673675, 0.1), (111.124690, 0.2)),
        ((290.299296, 0.0), (348.034898, 0.02), (65.927269, 0.1), (111.524944, 0.2)),
        ((503.451375, 0.0), (564.634514, 0.01), (71.227082, 0.1), (113.370718, 0.2)),
    ]
    positions = marker_positions(path)
    rows: list[dict[str, object]] = []
    marker = 0
    for source_index, source in enumerate(models):
        targets = [model for index, model in enumerate(models) if index != source_index]
        for target in targets:
            for unit, point in enumerate(positions[marker]):
                mis, mmu = linear_map(point, *transforms[source_index])
                rows.append(
                    {"source": source, "target": target, "unit": unit, "mis": mis, "mmu": mmu}
                )
            marker += 1
    return rows


def write_outputs(rows: list[dict[str, object]], table: Path, correlations: Path) -> None:
    table.parent.mkdir(parents=True, exist_ok=True)
    with table.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["source", "target", "unit", "mis", "mmu"])
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["source"]), str(row["target"]))].append(row)
    with correlations.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["source", "target", "n", "r"])
        writer.writeheader()
        for (source, target), group in grouped.items():
            xs = [float(row["mis"]) for row in group]
            ys = [float(row["mmu"]) for row in group]
            writer.writerow({"source": source, "target": target, "n": len(group), "r": pearson(xs, ys)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vision-pdf", type=Path, required=True)
    parser.add_argument("--language-pdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "Data" / "fig4")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    write_outputs(
        extract_vision(args.vision_pdf.resolve()),
        output / "vision-points.csv",
        output / "vision-correlations.csv",
    )
    write_outputs(
        extract_language(args.language_pdf.resolve()),
        output / "language-points.csv",
        output / "language-correlations.csv",
    )
    print(f"Wrote compact Figure 4A-D data to {output}")


if __name__ == "__main__":
    main()

