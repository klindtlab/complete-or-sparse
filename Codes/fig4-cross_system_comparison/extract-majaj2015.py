#!/usr/bin/env python3
"""Extract the manually downloaded public MajajHong2015 Brain-Score files.

The raw assembly contains repeated presentations and both V4 and IT units.
This script selects IT, averages repeated presentations for each of the 3,200
stimuli, records the 58 Chabo and 110 Tito units, and copies the corresponding
stimuli into ``Data/fig4/raw/majaj2015`` for LPIPS preprocessing.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import zipfile
from pathlib import Path

import h5py
import numpy as np
from scipy.io import savemat


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "Data" / "fig4" / "raw" / "majaj2015"
ASSEMBLY_KEY = "dicarlo.MajajHong2015.public"


def decode_strings(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values).reshape(-1)
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]
    )


def read_stimulus_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"image_id", "filename"}
    missing = required - set(rows[0] if rows else ())
    if missing:
        raise ValueError(f"Stimulus CSV is missing columns: {sorted(missing)}")
    if len(rows) != 3200:
        raise ValueError(f"Expected 3,200 stimulus rows, found {len(rows)}")
    return rows


def extract_responses(
    assembly_path: Path, stimulus_rows: list[dict[str, str]]
) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(assembly_path, "r") as assembly:
        if ASSEMBLY_KEY not in assembly:
            raise KeyError(f"Missing {ASSEMBLY_KEY!r} in {assembly_path}")
        regions = decode_strings(assembly["region"][:])
        keep = np.flatnonzero(np.char.upper(regions) == "IT")
        if len(keep) != 168:
            raise ValueError(f"Expected 168 IT units, found {len(keep)}")
        animals = decode_strings(assembly["animal"][:])[keep]
        presentation_ids = decode_strings(assembly["image_id"][:])
        values = np.asarray(assembly[ASSEMBLY_KEY][keep, :, :], dtype=np.float32)

    values = values.mean(axis=2).T
    stimulus_ids = [row["image_id"] for row in stimulus_rows]
    index_by_id = {stimulus_id: index for index, stimulus_id in enumerate(stimulus_ids)}
    unknown = sorted(set(presentation_ids) - set(index_by_id))
    if unknown:
        raise ValueError(f"Assembly contains {len(unknown)} image IDs absent from the CSV")
    group_indices = np.asarray([index_by_id[value] for value in presentation_ids])
    order = np.argsort(group_indices, kind="stable")
    sorted_groups = group_indices[order]
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_groups)) + 1]
    stops = np.r_[starts[1:], len(sorted_groups)]

    responses = np.full((len(stimulus_ids), len(keep)), np.nan, dtype=np.float32)
    for start, stop in zip(starts, stops):
        with np.errstate(invalid="ignore"):
            responses[sorted_groups[start]] = np.nanmean(sorted_values[start:stop], axis=0)
    if responses.shape != (3200, 168) or not np.isfinite(responses).all():
        incomplete = int(np.sum(~np.isfinite(responses).all(axis=1)))
        raise ValueError(
            f"Expected complete IT responses with shape (3200, 168); "
            f"got {responses.shape} with {incomplete} incomplete stimuli"
        )
    counts = {animal: int(np.sum(animals == animal)) for animal in np.unique(animals)}
    if counts != {"Chabo": 58, "Tito": 110}:
        raise ValueError(f"Unexpected IT unit ownership: {counts}")
    return responses, animals


def extract_stimuli(
    archive_path: Path, stimulus_rows: list[dict[str, str]], output: Path
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        by_basename: dict[str, str] = {}
        for name in archive.namelist():
            if name.endswith("/"):
                continue
            basename = Path(name).name
            if basename in by_basename:
                raise ValueError(f"Duplicate stimulus basename in ZIP: {basename}")
            by_basename[basename] = name
        for index, row in enumerate(stimulus_rows):
            filename = row["filename"]
            if filename not in by_basename:
                raise FileNotFoundError(f"Stimulus ZIP does not contain {filename}")
            destination = output / f"{index:04d}{Path(filename).suffix.lower()}"
            with archive.open(by_basename[filename]) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly", type=Path, required=True)
    parser.add_argument("--stimulus-csv", type=Path, required=True)
    parser.add_argument("--stimulus-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    stimulus_rows = read_stimulus_rows(args.stimulus_csv.resolve())
    responses, animals = extract_responses(args.assembly.resolve(), stimulus_rows)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    savemat(output / "responses.mat", {"neural_responses": responses})
    savemat(output / "animals.mat", {"animals": animals.astype(object)})
    extract_stimuli(args.stimulus_zip.resolve(), stimulus_rows, output / "stimuli")
    print(f"Saved averaged IT responses and 3,200 stimuli to {output}")


if __name__ == "__main__":
    main()
