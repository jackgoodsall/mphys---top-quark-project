#!/usr/bin/env python
"""Fail closed when a requested job does not satisfy the v3 blocking gates."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.data_contract import content_hash, load_contract, split_names  # noqa: E402
from src.utils.utils import load_any_config  # noqa: E402


def audit_gate(contract_path):
    contract = load_contract(contract_path)
    if contract["source"].get("mode") == "fixed_sources":
        missing = [item["path"] for item in contract["source"]["sources"] if not Path(item["path"]).exists()]
        if missing:
            raise SystemExit(f"BLOCKED: fixed source files do not exist: {missing}")
        if contract["matching"]["version"] != "upstream-jet-truthmatch-v1":
            raise SystemExit("BLOCKED: unreviewed fixed-source truth-match declaration")
    else:
        source = Path(contract["source"]["path"])
        if not source.exists():
            raise SystemExit(f"BLOCKED: source ROOT file does not exist: {source}")
        if contract["matching"]["version"] != "delta-r-exclusive-v1":
            raise SystemExit("BLOCKED: unreviewed truth matcher version")
    print(f"audit contract: {content_hash(contract)}")


def _verified_manifest(path):
    if not path.is_file():
        raise SystemExit(f"BLOCKED: missing manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    saved_hash = payload.pop("manifest_hash", None)
    if saved_hash != content_hash(payload):
        raise SystemExit(f"BLOCKED: invalid manifest self-hash: {path}")
    return payload


def _sha256_file(path, block_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_processed_split(processed_path, raw_path, split, contract, scaler_hash,
                            declared, expected_source_hash):
    required = {
        "jet", "src_mask", "event_id", "source_file_id", "source_entry",
        "valid_tops", "valid_Ws", "masks_tops", "masks_Ws",
    }
    expected_attrs = {
        "schema_version": contract["schema_version"],
        "contract_hash": content_hash(contract),
        "selection_hash": content_hash(contract["selection"]),
        "matcher_hash": content_hash(contract["matching"]),
        "scaler_hash": scaler_hash,
        "split": split,
    }
    with h5py.File(raw_path, "r") as raw, h5py.File(processed_path, "r") as processed:
        if str(raw.attrs.get("source_hash", "missing")) != str(expected_source_hash):
            raise SystemExit(f"BLOCKED: raw split source lineage mismatch in {raw_path}")
        missing = required - set(processed)
        if missing:
            raise SystemExit(f"BLOCKED: {processed_path} missing datasets: {sorted(missing)}")
        mismatches = {
            key: (str(processed.attrs.get(key, "missing")), str(value))
            for key, value in expected_attrs.items()
            if str(processed.attrs.get(key, "missing")) != str(value)
        }
        if mismatches:
            raise SystemExit(f"BLOCKED: processed attribute mismatch in {processed_path}: {mismatches}")
        if str(processed.attrs.get("source_hash", "missing")) != str(raw.attrs.get("source_hash", "missing")):
            raise SystemExit(f"BLOCKED: source lineage mismatch in {processed_path}")
        rows = int(processed["event_id"].shape[0])
        if rows <= 0 or rows != int(declared.get("rows", -1)) or rows != raw["event_id"].shape[0]:
            raise SystemExit(f"BLOCKED: row-count mismatch/empty split in {processed_path}")
        last = None
        for start in range(0, rows, 500_000):
            stop = min(start + 500_000, rows)
            ids = processed["event_id"][start:stop]
            if (ids.size and (np.any(ids[1:] <= ids[:-1]) or (last is not None and ids[0] <= last))):
                raise SystemExit(f"BLOCKED: event IDs are not unique/increasing in {processed_path}")
            if not np.array_equal(ids, raw["event_id"][start:stop]):
                raise SystemExit(f"BLOCKED: raw/processed event identity mismatch in {processed_path}")
            if contract["split"].get("mode") != "fixed_source_files":
                expected_split = split_names(
                    processed["source_file_id"][start:stop],
                    processed["source_entry"][start:stop], contract["split"],
                )
                if np.any(expected_split != split):
                    raise SystemExit(f"BLOCKED: deterministic split assignment mismatch in {processed_path}")
            if ids.size:
                last = ids[-1]


def training_gate(contract_path, config_path):
    contract = load_contract(contract_path)
    if contract["source"]["btag_provenance"] == "unknown":
        raise SystemExit("BLOCKED: b-tag provenance is still unknown")
    if contract["split"].get("group_provenance") == "unverified_entry_block":
        raise SystemExit("BLOCKED: generator-group provenance is not verified")
    manifest_path = Path(contract["output"]["manifest_path"])
    manifest = _verified_manifest(manifest_path)
    if manifest.get("contract_hash") != content_hash(contract):
        raise SystemExit("BLOCKED: manifest/config hash mismatch")
    if not manifest.get("complete_source", False):
        raise SystemExit("BLOCKED: raw manifest describes an audit subset")
    split_manifest = _verified_manifest(manifest_path.with_suffix(".splits.json"))
    if split_manifest.get("contract_hash") != content_hash(contract):
        raise SystemExit("BLOCKED: split manifest/config hash mismatch")
    source_lineage = split_manifest.get("source_lineage", {})
    for key in (
        "source_hash", "source_entry_start", "source_entry_stop",
        "source_total_entries", "complete_source",
    ):
        # fixed-source manifests record source_hash at top level, not in source_lineage
        declared = source_lineage.get(key, split_manifest.get(key, "missing"))
        if str(declared) != str(manifest.get(key, "missing")):
            raise SystemExit(f"BLOCKED: raw/split source lineage mismatch for {key}")

    cfg = load_any_config(config_path)
    tasks = cfg.get("tasks", {})
    if "invariant_mass" in tasks:
        raise SystemExit("BLOCKED: invariant-mass training loss is prohibited")
    if not tasks.get("chain_state", {}).get("enabled", False):
        raise SystemExit("BLOCKED: three-state chain detection is not enabled")
    # The null penalty trains unmatched queries toward empty masks. By default this
    # is refused because censored (unknowable) components would be supervised.
    # A config may consciously opt in via gate_overrides — the 21 Aug diagnosis
    # found its removal contributed to query collapse (see G_TRACK_HANDOFF.md).
    allow_null_penalty = bool(
        cfg.get("gate_overrides", {}).get("allow_null_penalty_on_censored", False)
    )
    for name in ("mask", "mask_W"):
        if float(tasks.get(name, {}).get("null_mask_penalty", 0)) != 0 and not allow_null_penalty:
            raise SystemExit(f"BLOCKED: {name} null penalty would supervise censored components")
    data = cfg["data_modules"]
    prefix = Path(data["input_path"]) / data["input_prefix"]
    splits = ("train", "val", "calibration", "test")
    missing = [name for name in splits if not Path(f"{prefix}{name}.h5").exists()]
    if missing:
        raise SystemExit(f"BLOCKED: missing processed splits: {missing}")
    processed_dir = Path(data["input_path"])
    preprocessing_manifest = _verified_manifest(processed_dir / "preprocessing_manifest.json")
    scaler_path = processed_dir / "target_transforms.joblib"
    if not scaler_path.is_file():
        raise SystemExit(f"BLOCKED: missing scaler artifact: {scaler_path}")
    scaler_hash = _sha256_file(scaler_path)
    if preprocessing_manifest.get("scaler_hash") != scaler_hash:
        raise SystemExit("BLOCKED: scaler artifact hash mismatch")
    outputs = preprocessing_manifest.get("outputs", {})
    raw_dir = Path(contract["output"]["split_dir"])
    raw_prefix = contract["output"]["split_prefix"]
    for split in splits:
        processed_path = Path(f"{prefix}{split}.h5")
        raw_path = raw_dir / f"{raw_prefix}{split}.h5"
        declared = outputs.get(processed_path.name)
        if declared is None:
            raise SystemExit(f"BLOCKED: processed output absent from manifest: {processed_path.name}")
        _verify_processed_split(
            processed_path, raw_path, split, contract, scaler_hash, declared,
            manifest["source_hash"],
        )
    if set(outputs) != {Path(f"{prefix}{split}.h5").name for split in splits}:
        raise SystemExit("BLOCKED: preprocessing manifest contains unexpected/missing outputs")
    shared_weight = tasks.get("exclusivity", {}).get("loss_weight")
    if shared_weight is None:
        raise SystemExit("BLOCKED: G1 requires one shared exclusivity weight")
    if any("loss_weight" in tasks.get(name, {}) for name in ("exclusive_ce", "exclusive_ce_W")):
        raise SystemExit("BLOCKED: per-head exclusivity weights may silently diverge")
    print("training gate: OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("audit", "training"))
    parser.add_argument("--contract", default="config/top_reconstruction_data_contract.yaml")
    parser.add_argument("--config")
    args = parser.parse_args()
    if args.stage == "audit":
        audit_gate(args.contract)
    elif not args.config:
        parser.error("training stage requires --config")
    else:
        training_gate(args.contract, args.config)


if __name__ == "__main__":
    main()
