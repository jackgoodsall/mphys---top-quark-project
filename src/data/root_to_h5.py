"""Manifest-backed inclusive ROOT -> HDF5 converter for the v3 data contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import awkward as ak
import h5py
import numpy as np
import uproot

try:
    from src.data.data_contract import content_hash, load_contract, stable_event_id, write_manifest
    from src.data.truth_matching import match_truth_to_jets
except ImportError:  # direct `python src/data/root_to_h5.py`
    from data_contract import content_hash, load_contract, stable_event_id, write_manifest
    from truth_matching import match_truth_to_jets


PARTON_TAGS = np.arange(1, 7, dtype=np.int8)
TRUTH_HADRONIC, TRUTH_SEMILEPTONIC, TRUTH_DILEPTONIC = 0, 1, 2
MATCH_ARRAY_KEYS = (
    "pre_acceptance_matchable", "post_acceptance_matchable",
    "post_truncation_matchable", "parton_ambiguous", "parton_collision",
)


def compute_energy(pt, eta, mass):
    return np.sqrt((pt * np.cosh(eta)) ** 2 + mass ** 2)


def truth_decay_channel(w_decay_ids) -> int:
    ids = np.abs(np.asarray(w_decay_ids, dtype=np.int64))
    if ids.shape != (4,):
        raise ValueError(f"expected four W decay products, got shape {ids.shape}")
    leptonic_w = sum(bool(np.any((side >= 11) & (side <= 16))) for side in (ids[:2], ids[2:]))
    return (TRUTH_HADRONIC, TRUTH_SEMILEPTONIC, TRUTH_DILEPTONIC)[leptonic_w]


def _partons(truth):
    top_id = np.asarray(truth["top_id"], dtype=np.int64)
    w_id = np.asarray(truth["w_id"], dtype=np.int64)
    b_id = np.asarray(truth["b_id"], dtype=np.int64)
    b_eta = np.asarray(truth["b_eta"], dtype=np.float64)
    b_phi = np.asarray(truth["b_phi"], dtype=np.float64)
    w_eta = np.asarray(truth["w_decay_eta"], dtype=np.float64)
    w_phi = np.asarray(truth["w_decay_phi"], dtype=np.float64)
    if any(value.shape != (2,) for value in (top_id, w_id, b_id, b_eta, b_phi)) \
            or w_eta.shape != (4,) or w_phi.shape != (4,):
        raise ValueError("truth record must contain two b and four W-daughter coordinates")
    if set(top_id.tolist()) != {6, -6} or set(w_id.tolist()) != {24, -24} \
            or set(b_id.tolist()) != {5, -5}:
        raise ValueError("truth top/W/b IDs do not form a ttbar decay")
    if not np.array_equal(np.sign(top_id), np.sign(w_id)) \
            or not np.array_equal(np.sign(top_id), np.sign(b_id)):
        raise ValueError("truth b/W array ordering is not aligned with top IDs")
    decay_ids = np.asarray(truth["w_decay_id"], dtype=np.int64).reshape(2, 2)
    charge3_abs = {1: -1, 2: 2, 3: -1, 4: 2, 5: -1, 6: 2,
                   11: -3, 12: 0, 13: -3, 14: 0, 15: -3, 16: 0}
    for chain in range(2):
        try:
            charge3 = sum(
                charge3_abs[abs(int(pid))] * (1 if pid > 0 else -1)
                for pid in decay_ids[chain]
            )
        except KeyError as exc:
            raise ValueError(f"unsupported W daughter PDG ID: {exc.args[0]}") from exc
        if charge3 != 3 * int(np.sign(w_id[chain])):
            raise ValueError("W daughter ordering/charge is inconsistent with the parent W")
    # Canonical chain order is top then antitop, independent of source row order.
    order = np.argsort(-top_id)
    w_eta_by_chain = w_eta.reshape(2, 2)[order]
    w_phi_by_chain = w_phi.reshape(2, 2)[order]
    return (
        np.concatenate([[b_eta[order[0]]], w_eta_by_chain[0], [b_eta[order[1]]], w_eta_by_chain[1]]),
        np.concatenate([[b_phi[order[0]]], w_phi_by_chain[0], [b_phi[order[1]]], w_phi_by_chain[1]]),
    )


def process_event(reco, truth, source_file_id, source_entry, contract):
    """Convert one event to a selected record plus independent selection bits."""
    pt = np.asarray(reco["jet_pt"], dtype=np.float32)
    eta = np.asarray(reco["jet_eta"], dtype=np.float32)
    phi = np.asarray(reco["jet_phi"], dtype=np.float32)
    mass = np.asarray(reco["jet_mass"], dtype=np.float32)
    btag = np.asarray(reco["jet_btag"], dtype=np.float32)
    if not (pt.shape == eta.shape == phi.shape == mass.shape == btag.shape):
        raise ValueError("reconstructed jet fields must have equal shapes")

    selection = contract["selection"]
    pt_acceptance = pt >= float(selection["jet_pt_min_gev"])
    eta_acceptance = np.abs(eta) <= float(selection["jet_abs_eta_max"])
    acceptance = pt_acceptance & eta_acceptance
    accepted = np.flatnonzero(acceptance)
    n_btags = int(np.count_nonzero(btag[accepted] > float(selection["btag_threshold"])))
    channel = truth_decay_channel(truth["w_decay_id"])
    reco_veto = len(reco["el_pt"]) == 0 and len(reco["mu_pt"]) == 0
    truth_ok = channel == TRUTH_HADRONIC if contract["population"]["truth_decay"] == "all_hadronic" else True
    bits = {
        "selection_truth_decay": truth_ok,
        "selection_reco_lepton_veto": reco_veto if contract["population"]["reco_lepton_veto"] else True,
        "selection_pt": np.count_nonzero(pt_acceptance) >= int(selection["min_jets"]),
        "selection_eta": len(accepted) >= int(selection["min_jets"]),
        "selection_min_jets": len(accepted) >= int(selection["min_jets"]),
        "selection_min_btags": n_btags >= int(selection["min_btags"]),
    }

    max_particles = int(contract["truncation"]["max_particles"])
    truncated = len(accepted) > max_particles
    bits["selection_truncation"] = not truncated if contract["truncation"]["policy"] == "exclude" else True
    selected = accepted[:max_particles]
    passed = all(bits.values())
    if not passed:
        return None, bits

    parton_eta, parton_phi = _partons(truth)
    radius = float(contract["matching"]["delta_r_max"])
    pre = match_truth_to_jets(parton_eta, parton_phi, eta, phi, radius)
    post_acceptance = match_truth_to_jets(parton_eta, parton_phi, eta, phi, radius, acceptance)
    truncation_mask = np.zeros(len(pt), dtype=bool)
    truncation_mask[selected] = True
    post_truncation = match_truth_to_jets(parton_eta, parton_phi, eta, phi, radius, truncation_mask)

    raw_to_stored = {int(raw): i for i, raw in enumerate(selected)}
    selected_parton_jet = np.array(
        [raw_to_stored.get(int(raw), -1) for raw in post_truncation.selected_jet], dtype=np.int16
    )
    tags = np.zeros(max_particles, dtype=np.float32)
    for parton, jet_index in enumerate(selected_parton_jet):
        if jet_index >= 0:
            tags[jet_index] = PARTON_TAGS[parton]

    jet = np.full((max_particles, 7), np.nan, dtype=np.float32)
    n_selected = len(selected)
    if n_selected:
        jet[:n_selected] = np.stack(
            (pt[selected], eta[selected], phi[selected], compute_energy(pt[selected], eta[selected], mass[selected]),
             mass[selected], btag[selected], tags[:n_selected]),
            axis=-1,
        )

    source_id = np.uint16(source_file_id)
    entry = np.uint64(source_entry)
    event_number = int(reco.get("event_number", source_entry))
    weight = float(reco.get("generator_weight", 1.0))
    group_size = int(contract["split"]["generator_group_size"])
    record = {
        "jet": jet,
        "event": np.array([n_selected, n_btags, np.all(selected_parton_jet >= 0)], dtype=np.float32),
        "event_id": stable_event_id(source_id, entry),
        "source_file_id": source_id,
        "source_entry": entry,
        "source_event_number": np.int64(event_number),
        "generator_group": np.uint64(source_entry // group_size),
        "generator_weight": np.float64(weight),
        "weight_sign": np.int8(np.sign(weight)),
        "truth_decay_channel": np.uint8(channel),
        "raw_njets": np.uint16(len(pt)),
        "pt_accepted_njets": np.uint16(np.count_nonzero(pt_acceptance)),
        "accepted_njets": np.uint16(len(accepted)),
        "selected_njets": np.uint16(n_selected),
        "accepted_nbtags": np.uint16(n_btags),
        "truncated_at_max_particles": np.uint8(truncated),
        "match_delta_r_raw": pre.distances.astype(np.float32).ravel(),
        "match_incidence_raw": pre.incidence.astype(np.uint8).ravel(),
        "selected_parton_jet": selected_parton_jet,
        "selected_parton_distance": post_truncation.selected_distance.astype(np.float32),
        "second_closest_local_raw_jet": post_truncation.second_closest_local_jet.astype(np.int32),
        "second_closest_local_distance": post_truncation.second_closest_local_distance.astype(np.float32),
        "pre_acceptance_matchable": (pre.selected_jet >= 0).astype(np.uint8),
        "post_acceptance_matchable": (post_acceptance.selected_jet >= 0).astype(np.uint8),
        "post_truncation_matchable": (post_truncation.selected_jet >= 0).astype(np.uint8),
        "parton_ambiguous": pre.parton_ambiguous.astype(np.uint8),
        "parton_collision": pre.parton_collision.astype(np.uint8),
    }
    record.update({key: np.uint8(value) for key, value in bits.items()})
    record["selection_pass"] = np.uint8(True)
    return record, bits


def _dataset_specs(max_particles):
    scalar = {
        "event_id": "u8", "source_file_id": "u2", "source_entry": "u8",
        "source_event_number": "i8", "generator_group": "u8", "generator_weight": "f8",
        "weight_sign": "i1", "truth_decay_channel": "u1", "raw_njets": "u2",
        "pt_accepted_njets": "u2",
        "accepted_njets": "u2", "selected_njets": "u2", "accepted_nbtags": "u2",
        "truncated_at_max_particles": "u1", "selection_truth_decay": "u1",
        "selection_reco_lepton_veto": "u1", "selection_pt": "u1", "selection_eta": "u1",
        "selection_min_jets": "u1",
        "selection_min_btags": "u1", "selection_truncation": "u1", "selection_pass": "u1",
    }
    specs = {key: ((), dtype) for key, dtype in scalar.items()}
    specs.update({
        "jet": ((max_particles, 7), "f4"), "event": ((3,), "f4"),
        "selected_parton_jet": ((6,), "i2"), "selected_parton_distance": ((6,), "f4"),
        "second_closest_local_raw_jet": ((6,), "i4"),
        "second_closest_local_distance": ((6,), "f4"),
    })
    specs.update({key: ((6,), "u1") for key in MATCH_ARRAY_KEYS})
    specs["match_delta_r_raw"] = ((), h5py.vlen_dtype(np.dtype("f4")))
    specs["match_incidence_raw"] = ((), h5py.vlen_dtype(np.dtype("u1")))
    return specs


def _create_datasets(handle, max_particles, compression, level):
    for key, (shape, dtype) in _dataset_specs(max_particles).items():
        kwargs = {}
        # HDF5 supports compression for variable-length arrays, too.
        if compression:
            kwargs["compression"] = compression
            if compression == "gzip":
                kwargs["compression_opts"] = level
        handle.create_dataset(key, shape=(0,) + shape, maxshape=(None,) + shape, dtype=dtype, **kwargs)


def _append(handle, records):
    if not records:
        return
    start = handle["event_id"].shape[0]
    stop = start + len(records)
    for key, dataset in handle.items():
        dataset.resize((stop,) + dataset.shape[1:])
        if key in {"match_delta_r_raw", "match_incidence_raw"}:
            for i, record in enumerate(records):
                # h5py interprets a one-element object array containing an
                # ndarray as a rectangular (1, N) value. Element-wise writes
                # preserve the intended VLEN dataset semantics.
                dataset[start + i] = record[key]
            continue
        else:
            values = np.asarray([record[key] for record in records], dtype=dataset.dtype)
        dataset[start:stop] = values


def _sha256_file(path, block_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def convert(contract, max_events=None, force=False, hash_source=True):
    source_cfg, output_cfg = contract["source"], contract["output"]
    source_path = Path(source_cfg["path"])
    output_path = Path(output_cfg["raw_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {output_path}; pass --force explicitly")
    temp_path = output_path.with_name(output_path.name + f".tmp-{os.getpid()}")
    if temp_path.exists():
        raise FileExistsError(f"temporary output already exists: {temp_path}")

    cutflow, cardinalities = Counter(), Counter()
    contract_digest = content_hash(contract)
    source_digest = _sha256_file(source_path) if hash_source else "not-computed-audit-only"
    conversion = contract["conversion"]
    try:
        with uproot.open(source_path) as root, h5py.File(temp_path, "w") as output:
            reco_tree = root[source_cfg["reco_tree"]]
            truth_tree = root[source_cfg["truth_tree"]]
            total = min(reco_tree.num_entries, max_events) if max_events else reco_tree.num_entries
            _create_datasets(
                output, int(contract["truncation"]["max_particles"]),
                conversion.get("compression"), int(conversion.get("compression_level", 4)),
            )
            output.attrs.update({
                "schema_version": contract["schema_version"], "contract_hash": contract_digest,
                "source_hash": source_digest, "selection_hash": content_hash(contract["selection"]),
                "matcher_hash": content_hash(contract["matching"]), "scaler_hash": "unfitted",
                "source_entry_start": 0, "source_entry_stop": int(total),
                "source_total_entries": int(reco_tree.num_entries),
                "complete_source": bool(total == reco_tree.num_entries),
            })
            reco_branches = ["jet_pt", "jet_eta", "jet_phi", "jet_mass", "jet_btag", "el_pt", "mu_pt"]
            if source_cfg.get("event_number_branch"):
                reco_branches.append(source_cfg["event_number_branch"])
            weight_branch = source_cfg.get("generator_weight_branch")
            if weight_branch:
                reco_branches.append(weight_branch)
            truth_map = source_cfg["truth_branches"]
            truth_branches = list(truth_map.values())
            chunk_size = int(conversion["chunk_size"])
            for start in range(0, total, chunk_size):
                stop = min(start + chunk_size, total)
                reco_arrays = reco_tree.arrays(reco_branches, entry_start=start, entry_stop=stop)
                truth_arrays = truth_tree.arrays(truth_branches, entry_start=start, entry_stop=stop)
                records = []
                for local in range(stop - start):
                    cutflow["generated"] += 1
                    reco = {key: ak.to_numpy(reco_arrays[key][local]) for key in reco_branches}
                    reco["event_number"] = np.asarray(
                        reco.pop(source_cfg["event_number_branch"])
                    ).item()
                    if weight_branch:
                        reco["generator_weight"] = float(reco.pop(weight_branch))
                    truth = {name: ak.to_numpy(truth_arrays[branch][local]) for name, branch in truth_map.items()}
                    record, bits = process_event(
                        reco, truth, int(source_cfg["source_file_id"]), start + local, contract
                    )
                    cumulative = True
                    for key in (
                        "selection_truth_decay", "selection_reco_lepton_veto",
                        "selection_pt", "selection_eta", "selection_min_jets",
                        "selection_min_btags", "selection_truncation",
                    ):
                        value = bool(bits[key])
                        cutflow[f"independent_{key}"] += int(value)
                        cumulative = cumulative and value
                        cutflow[f"cumulative_{key}"] += int(cumulative)
                    if record is None:
                        continue
                    cutflow["selected"] += 1
                    tags = record["jet"][:, 6]
                    cardinalities[f"top1_{np.isin(tags, [1, 2, 3]).sum()}"] += 1
                    cardinalities[f"top2_{np.isin(tags, [4, 5, 6]).sum()}"] += 1
                    cardinalities[f"W1_{np.isin(tags, [2, 3]).sum()}"] += 1
                    cardinalities[f"W2_{np.isin(tags, [5, 6]).sum()}"] += 1
                    records.append(record)
                _append(output, records)
                print(f"processed {stop:,}/{total:,}; selected {cutflow['selected']:,}", flush=True)
            output.flush()
        os.replace(temp_path, output_path)
    except BaseException:
        if temp_path.exists():
            temp_path.unlink()
        raise

    manifest = {
        "schema_version": contract["schema_version"], "contract": contract,
        "contract_hash": contract_digest, "source_hash": source_digest,
        "source_entry_start": 0, "source_entry_stop": int(total),
        "source_total_entries": int(reco_tree.num_entries),
        "complete_source": bool(total == reco_tree.num_entries),
        "raw_output": str(output_path), "cutflow": dict(cutflow),
        "label_cardinalities": dict(cardinalities),
        "generator_weight_available": bool(source_cfg.get("generator_weight_branch")),
        "btag_provenance": source_cfg["btag_provenance"],
    }
    write_manifest(output_cfg["manifest_path"], manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/top_reconstruction_data_contract.yaml")
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--output")
    parser.add_argument("--manifest")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-source-hash", action="store_true", help="audit-only; blocks production provenance")
    args = parser.parse_args(argv)
    contract = load_contract(args.config)
    if args.output:
        contract["output"]["raw_path"] = args.output
    if args.manifest:
        contract["output"]["manifest_path"] = args.manifest
    manifest = convert(
        contract, max_events=args.max_events, force=args.force,
        hash_source=not args.skip_source_hash,
    )
    print(json.dumps(manifest["cutflow"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
