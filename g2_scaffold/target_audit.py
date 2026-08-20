"""Chunked eligibility audit for the fixed-cardinality G2 targets.

The audit produces one compact index array per split.  It never rewrites the
source HDF5 files and never truncates a target mask.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, Mapping, Tuple

import h5py
import numpy as np


MASK_KEYS = ("masks_tops", "masks_Ws", "valid_tops", "valid_Ws")


def _chunks(size: int, chunk_size: int) -> Iterator[Tuple[int, int]]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    for start in range(0, size, chunk_size):
        yield start, min(start + chunk_size, size)


def _attr_value(value):
    """Convert common HDF5 attribute values to JSON-safe values."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def audit_file(path: str | Path, output_dir: str | Path, split: str,
               chunk_size: int = 8192) -> Dict[str, object]:
    """Audit one processed HDF5 file and write ``<split>_eligible_indices.npy``.

    An event is eligible iff every valid W has two particles and every valid
    full-top (valid top and valid W) contains its W pair plus one particle in
    ``top - W``. Reasons are counted per event; an event can contribute to more
    than one reason.
    """
    path = Path(path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eligible_chunks = []
    counts = {
        "invalid_mask_values": 0,
        "invalid_validity_values": 0,
        "w_cardinality": 0,
        "full_top_w_containment": 0,
        "full_top_b_cardinality": 0,
    }

    with h5py.File(path, "r") as handle:
        missing = [key for key in MASK_KEYS if key not in handle]
        if missing:
            raise KeyError(f"{path} is missing required datasets: {missing}")

        top_ds, w_ds = handle["masks_tops"], handle["masks_Ws"]
        top_valid_ds, w_valid_ds = handle["valid_tops"], handle["valid_Ws"]
        if top_ds.shape != w_ds.shape or top_ds.ndim != 3:
            raise ValueError("top and W masks must have matching shape [N, objects, particles]")
        if top_valid_ds.shape != w_valid_ds.shape or top_valid_ds.ndim != 2:
            raise ValueError("top and W validity arrays must have matching shape [N, objects]")
        if top_ds.shape[:2] != top_valid_ds.shape:
            raise ValueError("mask and validity object dimensions do not match")
        if top_ds.shape[0] != top_valid_ds.shape[0]:
            raise ValueError("mask and validity event dimensions do not match")

        total = int(top_ds.shape[0])
        for start, stop in _chunks(total, chunk_size):
            # Keep these reads explicitly sliced: the source arrays can contain
            # tens of millions of events.
            tops_raw = np.asarray(top_ds[start:stop])
            ws_raw = np.asarray(w_ds[start:stop])
            tops = tops_raw > 0.5
            ws = ws_raw > 0.5
            valid_tops_raw = np.asarray(top_valid_ds[start:stop])
            valid_ws_raw = np.asarray(w_valid_ds[start:stop])
            valid_tops = valid_tops_raw.astype(bool)
            valid_ws = valid_ws_raw.astype(bool)

            invalid_mask = (
                ~np.isfinite(tops_raw)
                | ~np.isfinite(ws_raw)
                | ((tops_raw != 0) & (tops_raw != 1))
                | ((ws_raw != 0) & (ws_raw != 1))
            ).any(axis=(1, 2))
            invalid_validity = (
                ~np.isin(valid_tops_raw, (0, 1))
                | ~np.isin(valid_ws_raw, (0, 1))
            ).any(axis=1)

            w_bad = (valid_ws & (ws.sum(axis=-1) != 2)).any(axis=1)
            full_top = valid_tops & valid_ws
            full_top_w_bad = (full_top & (ws & ~tops).any(axis=-1)).any(axis=1)
            b_counts = (tops & ~ws).sum(axis=-1)
            full_top_b_bad = (full_top & (b_counts != 1)).any(axis=1)

            bad = (
                invalid_mask | invalid_validity | w_bad
                | full_top_w_bad | full_top_b_bad
            )
            eligible = ~bad
            eligible_chunks.append(
                np.flatnonzero(eligible).astype(np.int64, copy=False) + start
            )
            counts["invalid_mask_values"] += int(invalid_mask.sum())
            counts["invalid_validity_values"] += int(invalid_validity.sum())
            counts["w_cardinality"] += int(w_bad.sum())
            counts["full_top_w_containment"] += int(full_top_w_bad.sum())
            counts["full_top_b_cardinality"] += int(full_top_b_bad.sum())

        eligible_indices = (
            np.concatenate(eligible_chunks)
            if eligible_chunks else np.empty(0, dtype=np.int64)
        )
        index_path = output_dir / f"{split}_eligible_indices.npy"
        np.save(index_path, eligible_indices)

        provenance = {
            key: _attr_value(handle.attrs[key])
            for key in ("source_hash", "contract_hash", "schema_version")
            if key in handle.attrs
        }

    excluded = total - int(eligible_indices.size)
    manifest = {
        "schema_version": "g2-target-audit-v1",
        "split": split,
        "source_file": str(path),
        # Store paths relative to the manifest directory so the manifest can
        # be moved as one self-contained artifact.
        "eligible_indices": index_path.name,
        "chunk_size": chunk_size,
        "counts": {
            "total": total,
            "eligible": int(eligible_indices.size),
            "excluded": excluded,
            "eligible_fraction": (int(eligible_indices.size) / total if total else 0.0),
        },
        "exclusion_reasons": counts,
        "provenance": provenance,
        "eligibility": {
            "valid_w_particles": 2,
            "valid_full_top_b_extension": 1,
            "policy": "exclude-invalid-or-ambiguous; never-truncate",
        },
    }
    return manifest


def audit_splits(files: Mapping[str, str | Path], output_dir: str | Path,
                 chunk_size: int = 8192) -> Dict[str, object]:
    """Audit named split files and write one aggregate JSON manifest."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = {
        split: audit_file(path, output_dir, split, chunk_size)
        for split, path in files.items()
    }
    manifest = {
        "schema_version": "g2-target-audit-v1",
        "chunk_size": chunk_size,
        "splits": splits,
    }
    manifest_path = output_dir / "target_audit_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _default_files(processed_dir: Path, stress_file: Path | None,
                   selected: Iterable[str]) -> Dict[str, Path]:
    files = {
        split: processed_dir / f"ttbar_contract_processed_{split}.h5"
        for split in selected
    }
    if stress_file is not None:
        files["stress"] = stress_file
    return files


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir", type=Path,
        default=Path("data/topquarkreconstruction/contract_v3/processed"),
        help="directory containing processed train/val/calibration/test files",
    )
    parser.add_argument("--stress-file", type=Path,
                        help="explicit processed stress HDF5 file")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument(
        "--split", action="append", choices=("train", "val", "calibration", "test"),
        dest="splits", help="split to audit; repeat to select a subset (default: all)",
    )
    args = parser.parse_args(argv)
    selected = args.splits or ("train", "val", "calibration", "test")
    files = _default_files(args.processed_dir, args.stress_file, selected)
    missing = {name: str(path) for name, path in files.items() if not path.is_file()}
    if missing:
        parser.error("input files do not exist: " + json.dumps(missing, sort_keys=True))
    manifest = audit_splits(files, args.output_dir, args.chunk_size)
    for split, result in manifest["splits"].items():
        counts = result["counts"]
        print(f"{split}: {counts['eligible']}/{counts['total']} eligible")
    print(f"manifest: {args.output_dir / 'target_audit_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
