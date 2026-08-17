"""Small, deterministic helpers for the top-reconstruction data contract."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np
import yaml


REQUIRED_SPLITS = ("train", "val", "calibration", "test")


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_hash(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def load_contract(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("data-contract config must be a mapping")
    validate_contract(config)
    return config


def validate_contract(config: Mapping) -> None:
    for section in (
        "schema_version", "source", "output", "population", "selection",
        "matching", "truncation", "split", "conversion", "weights",
    ):
        if section not in config:
            raise ValueError(f"missing data-contract section: {section}")

    fractions = config["split"]["fractions"]
    if tuple(fractions) != REQUIRED_SPLITS:
        raise ValueError(f"split fractions must be ordered as {REQUIRED_SPLITS}")
    values = [float(fractions[name]) for name in REQUIRED_SPLITS]
    if any(value < 0 for value in values):
        raise ValueError("split fractions cannot be negative")
    if not np.isclose(sum(values), 1.0):
        raise ValueError("split fractions must sum to one")
    required_truth = {
        "top_id", "w_id", "b_id", "b_eta", "b_phi",
        "w_decay_eta", "w_decay_phi", "w_decay_id",
    }
    if set(config["source"].get("truth_branches", {})) != required_truth:
        raise ValueError(f"source.truth_branches must declare exactly {sorted(required_truth)}")
    if float(config["matching"]["delta_r_max"]) <= 0:
        raise ValueError("matching.delta_r_max must be positive")
    supported = {
        "schema_version": (config["schema_version"], "top-reconstruction-v3"),
        "population.truth_decay": (config["population"].get("truth_decay"), "all_hadronic"),
        "matching.version": (config["matching"].get("version"), "delta-r-exclusive-v1"),
        "matching.objective": (
            config["matching"].get("objective"), "max-cardinality-then-min-total-delta-r"
        ),
        "matching.tie_break": (config["matching"].get("tie_break"), "lowest-jet-index"),
        "truncation.order": (config["truncation"].get("order"), "source_order"),
    }
    invalid = {name: value for name, (value, expected) in supported.items() if value != expected}
    if invalid:
        raise ValueError(f"unsupported contract declarations: {invalid}")
    if config["matching"].get("exact_cardinality") is not True:
        raise ValueError("matching.exact_cardinality must be true")
    if not isinstance(config["population"].get("reco_lepton_veto"), bool):
        raise ValueError("population.reco_lepton_veto must be boolean")
    source_file_id = int(config["source"].get("source_file_id", -1))
    if not 0 <= source_file_id < 2**16:
        raise ValueError("source_file_id must fit the stable event-ID layout")
    selection = config["selection"]
    if float(selection["jet_pt_min_gev"]) < 0 or float(selection["jet_abs_eta_max"]) <= 0:
        raise ValueError("selection pT/eta thresholds are invalid")
    if int(selection["min_jets"]) < 0 or int(selection["min_btags"]) < 0:
        raise ValueError("selection multiplicities cannot be negative")
    if int(config["truncation"]["max_particles"]) < 6:
        raise ValueError("truncation.max_particles must permit a six-jet event")
    if config["truncation"]["policy"] not in {"retain_and_report", "exclude"}:
        raise ValueError("unsupported truncation policy")
    if config["weights"]["missing_policy"] != "unit":
        raise ValueError("only explicit unit fallback weights are supported")
    if int(config["conversion"]["chunk_size"]) <= 0:
        raise ValueError("conversion.chunk_size must be positive")
    if config["conversion"].get("compression") not in {None, "gzip"}:
        raise ValueError("only gzip or uncompressed HDF5 output is supported")
    if config["split"].get("group_provenance") not in {
        "upstream_generator_group", "unverified_entry_block"
    }:
        raise ValueError("split.group_provenance must declare verified or audit-only grouping")


def stable_event_id(source_file_id, source_entry):
    """Pack `(source_file_id, source_entry)` into a reproducible uint64."""
    source = np.asarray(source_file_id, dtype=np.uint64)
    entry = np.asarray(source_entry, dtype=np.uint64)
    if np.any(source >= 2**16) or np.any(entry >= 2**48):
        raise ValueError("event identity exceeds the 16-bit source / 48-bit entry layout")
    return (source << np.uint64(48)) | entry


def split_names(source_file_id, source_entry, split_config: Mapping) -> np.ndarray:
    """Assign complete generator groups, never individual rows, to data splits."""
    source = np.asarray(source_file_id, dtype=np.uint64)
    entry = np.asarray(source_entry, dtype=np.uint64)
    source, entry = np.broadcast_arrays(source, entry)
    group_size = int(split_config["generator_group_size"])
    if group_size <= 0:
        raise ValueError("generator_group_size must be positive")
    groups = entry // np.uint64(group_size)
    seed = int(split_config["seed"])

    # SplitMix64 gives a stable, well-mixed mapping without Python's salted hash.
    x = groups ^ (source << np.uint64(32)) ^ np.uint64(seed)
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    x = x ^ (x >> np.uint64(31))
    u = x.astype(np.float64) / np.float64(np.iinfo(np.uint64).max)

    fractions = split_config["fractions"]
    edges = np.cumsum([float(fractions[name]) for name in REQUIRED_SPLITS])
    indices = np.searchsorted(edges, u, side="right").clip(max=len(REQUIRED_SPLITS) - 1)
    return np.asarray(REQUIRED_SPLITS, dtype="U11")[indices]


def write_manifest(path: str | Path, manifest: Mapping) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(manifest)
    payload["manifest_hash"] = content_hash(payload)
    temp = output.with_name(output.name + f".tmp-{os.getpid()}")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, output)
