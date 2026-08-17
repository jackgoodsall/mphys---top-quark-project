"""Split a contract HDF5 by stable generator groups.

Rows are read sequentially, but every `(source_file_id, source_group)` is
assigned as a unit.  A separate calibration split is mandatory.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import os
from pathlib import Path
import shutil
import tempfile

import h5py
import numpy as np

try:
    from src.data.data_contract import content_hash, load_contract, split_names, write_manifest
except ImportError:  # direct `python src/data/split_raw_h5.py`
    from data_contract import content_hash, load_contract, split_names, write_manifest


IDENTITY_KEYS = ("event_id", "source_file_id", "source_entry", "generator_group")


def _create_outputs(source: h5py.File, output_dir: Path, prefix: str, split_order):
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for split in split_order:
        path = output_dir / f"{prefix}{split}.h5"
        handle = h5py.File(path, "w")
        for key, value in source.attrs.items():
            handle.attrs[key] = value
        handle.attrs["split"] = split
        for key, dataset in source.items():
            kwargs = {}
            if dataset.compression is not None:
                kwargs["compression"] = dataset.compression
                if dataset.compression_opts is not None:
                    kwargs["compression_opts"] = dataset.compression_opts
            handle.create_dataset(
                key,
                shape=(0,) + dataset.shape[1:],
                maxshape=(None,) + dataset.shape[1:],
                dtype=dataset.dtype,
                **kwargs,
            )
        outputs[split] = handle
    return outputs


def _validate_contract_attrs(source: h5py.File, contract):
    if contract is None:
        return
    expected = {
        "schema_version": contract["schema_version"],
        "contract_hash": content_hash(contract),
        "selection_hash": content_hash(contract["selection"]),
        "matcher_hash": content_hash(contract["matching"]),
    }
    mismatches = {
        key: (str(source.attrs.get(key, "missing")), str(value))
        for key, value in expected.items()
        if str(source.attrs.get(key, "missing")) != str(value)
    }
    if mismatches:
        raise ValueError(f"input HDF5 contract attributes do not match config: {mismatches}")
    source_hash = str(source.attrs.get("source_hash", "missing"))
    if source_hash in {"missing", "not-computed-audit-only"}:
        raise ValueError("input HDF5 has no production source hash")
    if not bool(source.attrs.get("complete_source", False)):
        raise ValueError("production splitting rejects audit subsets/incomplete source conversion")


def split_file(input_path: Path, output_dir: Path, prefix: str, split_config,
               chunk_size=500_000, force=False, contract=None):
    if force:
        raise ValueError("transactional contract splits do not support --force; use a new versioned output directory")
    split_order = tuple(split_config["fractions"])
    counts = Counter()
    groups_by_split = defaultdict(set)
    seen_groups = {}
    final_paths = {split: output_dir / f"{prefix}{split}.h5" for split in split_order}
    if output_dir.exists():
        raise FileExistsError(f"refusing to replace existing split directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="contract-splits-", dir=output_dir.parent))

    try:
        with h5py.File(input_path, "r") as source:
            _validate_contract_attrs(source, contract)
            source_lineage = {
                key: (source.attrs.get(key).item() if isinstance(source.attrs.get(key), np.generic)
                      else source.attrs.get(key))
                for key in (
                    "source_hash", "source_entry_start", "source_entry_stop",
                    "source_total_entries", "complete_source",
                )
            }
            missing = [key for key in IDENTITY_KEYS if key not in source]
            if missing:
                raise ValueError(f"contract input is missing identity keys: {missing}")
            keys = tuple(source.keys())
            outputs = _create_outputs(source, temp_dir, prefix, split_order)
            last_event_id = None
            try:
                length = source["event_id"].shape[0]
                for start in range(0, length, chunk_size):
                    stop = min(start + chunk_size, length)
                    event_ids = source["event_id"][start:stop]
                    if event_ids.size and (
                        np.any(event_ids[1:] <= event_ids[:-1])
                        or (last_event_id is not None and event_ids[0] <= last_event_id)
                    ):
                        raise ValueError("event_id must be globally unique and strictly increasing")
                    if event_ids.size:
                        last_event_id = event_ids[-1]
                    source_ids = source["source_file_id"][start:stop]
                    entries = source["source_entry"][start:stop]
                    groups = source["generator_group"][start:stop]
                    expected_groups = entries // np.uint64(split_config["generator_group_size"])
                    if not np.array_equal(groups, expected_groups):
                        raise ValueError("stored generator_group disagrees with the split contract")
                    assignments = split_names(source_ids, entries, split_config)

                    for source_id, group, split in zip(source_ids, groups, assignments):
                        group_key = (int(source_id), int(group))
                        old = seen_groups.setdefault(group_key, str(split))
                        if old != split:
                            raise AssertionError(f"generator group {group_key} crossed {old}/{split}")
                        groups_by_split[str(split)].add(group_key)

                    chunk = {key: source[key][start:stop] for key in keys}
                    for split in split_order:
                        mask = assignments == split
                        n_keep = int(mask.sum())
                        if not n_keep:
                            continue
                        out = outputs[split]
                        old_len = out["event_id"].shape[0]
                        new_len = old_len + n_keep
                        for key in keys:
                            out[key].resize((new_len,) + out[key].shape[1:])
                            selected_values = chunk[key][mask]
                            if h5py.check_dtype(vlen=out[key].dtype) is not None:
                                for offset, value in enumerate(selected_values):
                                    out[key][old_len + offset] = value
                            else:
                                out[key][old_len:new_len] = selected_values
                        counts[split] += n_keep
            finally:
                for handle in outputs.values():
                    handle.close()

        if sum(counts.values()) != length:
            raise AssertionError("split event counts do not sum to the input length")
        group_sets = list(groups_by_split.values())
        for i, left in enumerate(group_sets):
            for right in group_sets[i + 1:]:
                if left & right:
                    raise AssertionError("generator group appears in more than one split")
        os.replace(temp_dir, output_dir)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return {
        "input": str(input_path),
        "event_counts": dict(counts),
        "group_counts": {name: len(groups_by_split[name]) for name in split_order},
        "split_config": dict(split_config),
        "source_lineage": source_lineage,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/top_reconstruction_data_contract.yaml")
    parser.add_argument("--input")
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    contract = load_contract(args.config)
    output = contract["output"]
    report = split_file(
        Path(args.input or output["raw_path"]),
        Path(output["split_dir"]),
        output["split_prefix"],
        contract["split"],
        args.chunk_size or int(contract["conversion"]["chunk_size"]),
        force=args.force,
        contract=contract,
    )
    report["contract_hash"] = content_hash(contract)
    write_manifest(Path(output["manifest_path"]).with_suffix(".splits.json"), report)
    for name, count in report["event_counts"].items():
        print(f"{name}: {count:,}")


if __name__ == "__main__":
    main()
