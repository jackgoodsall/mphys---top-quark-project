"""Convert fixed HYPER and inclusive sources into the v3 raw HDF5 contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import uproot

try:
    from src.data.data_contract import content_hash, load_contract, stable_event_id, write_manifest
    from src.data.root_to_h5 import _append, _create_datasets, _sha256_file
except ImportError:  # direct invocation from src/data
    from data_contract import content_hash, load_contract, stable_event_id, write_manifest
    from root_to_h5 import _append, _create_datasets, _sha256_file


PARTON_LABELS = np.arange(1, 7, dtype=np.int8)


def _as_1d(value, dtype=np.float32):
    return np.asarray(value, dtype=dtype).reshape(-1)


def _record(jet_pt, jet_eta, jet_phi, jet_energy, jet_mass, jet_btag, jet_tag,
            source_file_id, source_entry, contract, include=True):
    pt = _as_1d(jet_pt)
    eta = _as_1d(jet_eta)
    phi = _as_1d(jet_phi)
    energy = _as_1d(jet_energy)
    mass = _as_1d(jet_mass)
    btag = _as_1d(jet_btag)
    raw_tag = _as_1d(jet_tag)
    tag = np.where(np.isfinite(raw_tag), np.rint(raw_tag), 0).astype(np.int16)
    if not (pt.shape == eta.shape == phi.shape == energy.shape == mass.shape == btag.shape == tag.shape):
        raise ValueError("fixed-source jet fields must have equal shapes")
    if np.any((tag < 0) | (tag > 6)):
        raise ValueError("truth tags must be integers in [0, 6]")

    finite = np.isfinite(pt) & np.isfinite(eta) & np.isfinite(phi)
    raw_indices = np.flatnonzero(finite)
    max_particles = int(contract["truncation"]["max_particles"])
    selected = raw_indices[:max_particles]
    truncated = len(raw_indices) > max_particles
    if len(selected):
        missing_energy = ~np.isfinite(energy[selected])
        if np.any(missing_energy):
            energy[selected[missing_energy]] = np.sqrt(
                (pt[selected[missing_energy]] * np.cosh(eta[selected[missing_energy]])) ** 2
                + mass[selected[missing_energy]] ** 2
            )

    selection = contract["selection"]
    pt_acceptance = finite & (pt >= float(selection["jet_pt_min_gev"]))
    eta_acceptance = finite & (np.abs(eta) <= float(selection["jet_abs_eta_max"]))
    accepted = pt_acceptance & eta_acceptance
    n_btags = int(np.count_nonzero(btag[accepted] > float(selection["btag_threshold"])))
    bits = {
        "selection_truth_decay": True,
        "selection_reco_lepton_veto": True,
        "selection_pt": int(np.count_nonzero(pt_acceptance) >= int(selection["min_jets"])),
        "selection_eta": int(np.count_nonzero(accepted) >= int(selection["min_jets"])),
        "selection_min_jets": int(np.count_nonzero(accepted) >= int(selection["min_jets"])),
        "selection_min_btags": int(n_btags >= int(selection["min_btags"])),
        "selection_truncation": int(not truncated),
    }

    raw_to_stored = {int(raw): i for i, raw in enumerate(selected)}
    selected_parton_jet = np.full(6, -1, dtype=np.int16)
    second_closest = np.full(6, -1, dtype=np.int32)
    incidence = np.zeros((6, len(tag)), dtype=np.uint8)
    ambiguous = np.zeros(6, dtype=np.uint8)
    for parton, label in enumerate(PARTON_LABELS):
        matches = np.flatnonzero(tag == label)
        incidence[parton, matches] = 1
        ambiguous[parton] = np.uint8(len(matches) > 1)
        if len(matches):
            selected_parton_jet[parton] = raw_to_stored.get(int(matches[0]), -1)
        if len(matches) > 1:
            second_closest[parton] = int(matches[1])

    stored_tag = np.zeros(max_particles, dtype=np.float32)
    stored_jet = np.full((max_particles, 7), np.nan, dtype=np.float32)
    if len(selected):
        count = len(selected)
        stored_tag[:count] = tag[selected]
        stored_jet[:count] = np.stack(
            (pt[selected], eta[selected], phi[selected], energy[selected], mass[selected],
             btag[selected], stored_tag[:count]), axis=-1,
        )

    source_id = np.uint16(source_file_id)
    entry = np.uint64(source_entry)
    group_size = max(1, int(contract["split"]["generator_group_size"]))
    record = {
        "jet": stored_jet,
        "event": np.array([len(selected), n_btags, np.all(selected_parton_jet >= 0)], dtype=np.float32),
        "event_id": stable_event_id(source_id, entry),
        "source_file_id": source_id,
        "source_entry": entry,
        "source_event_number": np.int64(source_entry),
        "generator_group": np.uint64(source_entry // group_size),
        "generator_weight": np.float64(1.0),
        "weight_sign": np.int8(1),
        "truth_decay_channel": np.uint8(0),
        "raw_njets": np.uint16(len(raw_indices)),
        "pt_accepted_njets": np.uint16(np.count_nonzero(pt_acceptance)),
        "accepted_njets": np.uint16(np.count_nonzero(accepted)),
        "selected_njets": np.uint16(len(selected)),
        "accepted_nbtags": np.uint16(n_btags),
        "truncated_at_max_particles": np.uint8(truncated),
        "match_delta_r_raw": np.full(6 * len(tag), np.nan, dtype=np.float32),
        "match_incidence_raw": incidence.ravel(),
        "selected_parton_jet": selected_parton_jet,
        "selected_parton_distance": np.full(6, np.nan, dtype=np.float32),
        "second_closest_local_raw_jet": second_closest,
        "second_closest_local_distance": np.full(6, np.nan, dtype=np.float32),
        "pre_acceptance_matchable": (incidence.any(axis=1)).astype(np.uint8),
        "post_acceptance_matchable": np.array(
            [np.any((tag == label) & accepted) for label in PARTON_LABELS], dtype=np.uint8
        ),
        "post_truncation_matchable": (selected_parton_jet >= 0).astype(np.uint8),
        "parton_ambiguous": ambiguous,
        "parton_collision": np.zeros(6, dtype=np.uint8),
    }
    record.update({key: np.uint8(value) for key, value in bits.items()})
    # Fixed source partitions are intentionally preserved, including events that
    # fail the historical >=6-jet selection.  The selection bits remain audit
    # diagnostics; selection_pass means included by the declared source policy.
    record["selection_pass"] = np.uint8(bool(include))
    return record, bits


def _source_entries(source, max_events=None, chunk_size=100_000):
    path = Path(source["path"])
    fmt = source["format"]
    limit = max_events
    if fmt == "legacy_raw_h5":
        with h5py.File(path, "r") as handle:
            total = handle["jet"].shape[0]
            stop = min(total, limit) if limit is not None else total
            for start in range(0, stop, chunk_size):
                end = min(start + chunk_size, stop)
                jet = handle["jet"][start:end]
                for offset, row in enumerate(jet):
                    yield start + offset, (row[:, 0], row[:, 1], row[:, 2], row[:, 3],
                                           row[:, 4], row[:, 5], row[:, 6])
        return
    if fmt != "hyper_delphes":
        raise ValueError(f"unsupported fixed source format: {fmt}")
    with uproot.open(path) as root:
        tree = root[source.get("tree", "Delphes")]
        stop = min(tree.num_entries, limit) if limit is not None else tree.num_entries
        branches = ["jet_pt", "jet_eta", "jet_phi", "jet_e", "jet_m", "jet_bTag", "jet_truthmatch"]
        offset = 0
        for arrays in tree.iterate(branches, entry_start=0, entry_stop=stop,
                                   step_size=chunk_size, library="ak"):
            rows = len(arrays["jet_pt"])
            for local in range(rows):
                yield offset + local, tuple(arrays[key][local] for key in branches)
            offset += rows


def _source_hashes(sources, skip):
    if skip:
        return "not-computed-audit-only", {}
    hashes = {str(source["path"]): _sha256_file(source["path"]) for source in sources}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return digest, hashes


def convert(contract, max_events_per_source=None, skip_source_hash=False, output_dir=None):
    sources = contract["source"]["sources"]
    output = contract["output"]
    split_dir = Path(output_dir or output["split_dir"])
    stress_dir = split_dir.parent / "stress"
    split_dir.mkdir(parents=True, exist_ok=True)
    stress_dir.mkdir(parents=True, exist_ok=True)
    source_hash, individual_hashes = _source_hashes(sources, skip_source_hash)
    contract_hash = content_hash(contract)
    outputs = {}
    handles = {}
    counts = Counter()
    try:
        for split in ("train", "val", "calibration", "test", "stress"):
            path = (stress_dir if split == "stress" else split_dir) / f"{output['split_prefix']}{split}.h5"
            temp = path.with_name(path.name + f".tmp-{os.getpid()}")
            if temp.exists() or path.exists():
                raise FileExistsError(f"output already exists: {path}")
            handle = h5py.File(temp, "w")
            _create_datasets(handle, int(contract["truncation"]["max_particles"]),
                             contract["conversion"].get("compression"),
                             int(contract["conversion"].get("compression_level", 4)))
            handles[split] = handle
            outputs[split] = (path, temp)

        source_manifest = []
        for source in sources:
            source_path = Path(source["path"])
            split = source["split"]
            emitted = 0
            source_entries = 0
            records = []
            for entry, values in _source_entries(
                source, max_events=max_events_per_source,
                chunk_size=int(contract["conversion"]["chunk_size"]),
            ):
                record, _ = _record(*values, source["source_file_id"], entry, contract)
                records.append(record)
                if len(records) == 10_000:
                    _append(handles[split], records)
                    records.clear()
                emitted += 1
                source_entries = max(source_entries, entry + 1)
                counts[f"{split}_generated"] += 1
                counts[f"{split}_selected"] += 1
            _append(handles[split], records)
            source_manifest.append({
                "path": str(source_path), "format": source["format"],
                "source_file_id": int(source["source_file_id"]), "split": split,
                "source_entries": source_entries, "emitted_rows": emitted,
                "sha256": individual_hashes.get(str(source_path), "not-computed-audit-only"),
            })
        total_rows = sum(item["emitted_rows"] for item in source_manifest)
        complete = max_events_per_source is None
        for split, handle in handles.items():
            handle.attrs.update({
                "schema_version": contract["schema_version"], "contract_hash": contract_hash,
                "source_hash": source_hash, "selection_hash": content_hash(contract["selection"]),
                "matcher_hash": content_hash(contract["matching"]), "scaler_hash": "unfitted",
                "source_entry_start": 0, "source_entry_stop": total_rows,
                "source_total_entries": total_rows, "complete_source": complete,
                "split": split,
            })
            handle.close()
            if complete:
                os.replace(outputs[split][1], outputs[split][0])
            else:
                outputs[split][1].unlink()
                outputs[split][0].unlink(missing_ok=True)
    except BaseException:
        for handle in handles.values():
            if not handle.id.valid:
                continue
            handle.close()
        for _, temp in outputs.values():
            temp.unlink(missing_ok=True)
        raise

    manifest = {
        "schema_version": contract["schema_version"], "contract": contract,
        "contract_hash": contract_hash, "source_hash": source_hash,
        "source_entry_start": 0, "source_entry_stop": sum(x["source_entries"] for x in source_manifest),
        "source_total_entries": sum(x["source_entries"] for x in source_manifest),
        "complete_source": complete, "source_files": source_manifest,
        "outputs": {split: str(path) for split, (path, _) in outputs.items()},
        "cutflow": dict(counts), "btag_provenance": contract["source"]["btag_provenance"],
        "truthmatch_provenance": contract["source"].get("truthmatch_provenance"),
    }
    write_manifest(output["manifest_path"], manifest)
    write_manifest(Path(output["manifest_path"]).with_suffix(".splits.json"), {
        "contract_hash": contract_hash, "source_hash": source_hash,
        "source_lineage": {key: manifest[key] for key in (
            "source_entry_start", "source_entry_stop", "source_total_entries", "complete_source"
        )}, "event_counts": {split: counts[f"{split}_selected"] for split in outputs},
        "fixed_source_partitions": True,
    })
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/top_reconstruction_data_contract.yaml")
    parser.add_argument("--max-events-per-source", type=int)
    parser.add_argument("--skip-source-hash", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--manifest")
    args = parser.parse_args(argv)
    contract = load_contract(args.config)
    if args.manifest:
        contract["output"]["manifest_path"] = args.manifest
    manifest = convert(contract, args.max_events_per_source, args.skip_source_hash, args.output_dir)
    print(json.dumps(manifest["cutflow"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
