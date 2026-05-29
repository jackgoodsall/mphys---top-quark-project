"""
Fix many-to-one jet-to-parton matching in the inclusive ttbar ROOT file.

For each parton branch in the matching tree, if >1 jet is matched to the same
parton in an event, only the jet with the smallest delta R to the truth-level
parton is kept; all others are removed from that branch's index list.

All three trees (reco, matching, truth) are written to the output file unchanged
except for the disambiguated matching branches.

truth_decay ordering (verified from PDG IDs in truth tree):
    index 0 → b_from_Wplus   (b quark, id=5)
    index 1 → b_from_Wminus  (anti-b quark, id=-5)
    index 2 → Wplus_decay1
    index 3 → Wplus_decay2
    index 4 → Wminus_decay1
    index 5 → Wminus_decay2

Usage:
    uv run src/data/fix_matching_unique_tags.py
"""

import math
import numpy as np
import uproot
import awkward as ak
from pathlib import Path


def cast_int64(chunk):
    """Cast any var * int64 fields to var * int32 (uproot can't write int64 jagged branches).
    Returns a plain dict so uproot sees it as {branch_name: array} rather than a record type."""
    return {key: ak.values_astype(chunk[key], np.int32)
            if "int64" in str(ak.type(chunk[key])) else chunk[key]
            for key in chunk.fields}

# ── Config ──────────────────────────────────────────────────────────────────
ROOT_IN  = Path("data/topquarkreconstruction/root_data/ttbarLO_inclusive_mostof20M.root")
ROOT_OUT = Path("data/topquarkreconstruction/root_data/ttbarLO_inclusive_unique_tags.root")
CHUNK_SIZE = 200_000

# matching branch → index into truth_decay arrays
BRANCH_TRUTH_IDX = {
    "b_from_Wplus_jet_indices":  0,
    "Wplus_decay1_jet_indices":  2,
    "Wplus_decay2_jet_indices":  3,
    "b_from_Wminus_jet_indices": 1,
    "Wminus_decay1_jet_indices": 4,
    "Wminus_decay2_jet_indices": 5,
}


def fix_branch(indices_ak, jet_eta_ak, jet_phi_ak, parton_eta_ak, parton_phi_ak):
    """
    For each event where a parton branch lists >1 jet, keep only the jet
    with the smallest delta R to the truth parton.

    Returns an ak.Array of the same type but with at most 1 entry per event.
    """
    indices_list   = indices_ak.tolist()
    jet_eta_list   = jet_eta_ak.tolist()
    jet_phi_list   = jet_phi_ak.tolist()
    parton_etas    = parton_eta_ak.tolist()
    parton_phis    = parton_phi_ak.tolist()

    for i, idxs in enumerate(indices_list):
        if len(idxs) <= 1:
            continue

        p_eta = parton_etas[i]
        p_phi = parton_phis[i]

        best_idx = None
        best_dr  = math.inf
        for ji in idxs:
            if ji < 0 or ji >= len(jet_eta_list[i]):
                continue
            d_eta = jet_eta_list[i][ji] - p_eta
            d_phi = jet_phi_list[i][ji] - p_phi
            d_phi = (d_phi + math.pi) % (2.0 * math.pi) - math.pi
            dr = math.sqrt(d_eta * d_eta + d_phi * d_phi)
            if dr < best_dr:
                best_dr  = dr
                best_idx = ji

        indices_list[i] = [best_idx] if best_idx is not None else []

    return ak.Array(indices_list)


def process_matching_chunk(matching_chunk, reco_chunk, truth_chunk):
    """Return a new matching dict with disambiguated jet indices."""
    jet_eta = reco_chunk["jet_eta"]
    jet_phi = reco_chunk["jet_phi"]
    truth_decay_eta = truth_chunk["truth_decay_eta"]
    truth_decay_phi = truth_chunk["truth_decay_phi"]

    fixed = {}
    multi_total = 0
    for branch, tidx in BRANCH_TRUTH_IDX.items():
        indices = matching_chunk[branch]

        n_multi = int(ak.sum(ak.num(indices) > 1))
        multi_total += n_multi

        parton_eta = truth_decay_eta[:, tidx]
        parton_phi = truth_decay_phi[:, tidx]

        fixed[branch] = fix_branch(indices, jet_eta, jet_phi, parton_eta, parton_phi)

    # Copy remaining branches (electron / muon indices) unchanged
    for key in matching_chunk.fields:
        if key not in BRANCH_TRUTH_IDX:
            fixed[key] = matching_chunk[key]

    return fixed, multi_total


def main():
    ROOT_OUT.parent.mkdir(parents=True, exist_ok=True)

    f_in = uproot.open(ROOT_IN)
    reco_tree    = f_in["reco;1"]
    matching_tree = f_in["matching;1"]
    truth_tree   = f_in["truth;1"]

    total_entries = reco_tree.num_entries
    print(f"Input events : {total_entries:,}")
    print(f"Output       : {ROOT_OUT}")

    reco_branches     = reco_tree.keys()
    matching_branches = matching_tree.keys()
    truth_branches    = ["truth_decay_eta", "truth_decay_phi"]
    truth_all_branches = truth_tree.keys()

    grand_multi = 0
    processed   = 0
    first_chunk = True

    with uproot.recreate(ROOT_OUT) as f_out:
        reco_iter    = reco_tree.iterate(reco_branches,
                                         step_size=CHUNK_SIZE, library="ak")
        match_iter   = matching_tree.iterate(matching_branches,
                                             step_size=CHUNK_SIZE, library="ak")
        truth_iter   = truth_tree.iterate(truth_all_branches,
                                          step_size=CHUNK_SIZE, library="ak")

        for reco_chunk, match_chunk, truth_chunk in zip(reco_iter, match_iter, truth_iter):
            n = len(reco_chunk["jet_pt"])

            fixed_matching, n_multi = process_matching_chunk(
                match_chunk, reco_chunk, truth_chunk
            )
            grand_multi += n_multi
            processed   += n

            reco_write    = cast_int64(reco_chunk)
            truth_write   = cast_int64(truth_chunk)
            match_write   = {k: ak.values_astype(v, np.int32)
                             if "int64" in str(ak.type(v)) else v
                             for k, v in fixed_matching.items()}

            if first_chunk:
                f_out["reco"]     = reco_write
                f_out["truth"]    = truth_write
                f_out["matching"] = match_write
                first_chunk = False
            else:
                f_out["reco"].extend(reco_write)
                f_out["truth"].extend(truth_write)
                f_out["matching"].extend(match_write)

            print(f"  {processed:>10,} / {total_entries:,} events  "
                  f"(multi-matches fixed this chunk: {n_multi})")

    print(f"\nDone. Total multi-match assignments fixed: {grand_multi:,}")
    print(f"Output: {ROOT_OUT}")


if __name__ == "__main__":
    main()
