"""
Write back a ROOT file with disambiguated matching indices.

All events (hadronic, semi-leptonic, fully leptonic) are preserved — no events
are filtered or dropped. The six matching index branches are corrected so that
each parton tag appears at most once across all jets in an event.

When multiple jets share the same parton tag (many-to-one collision), only the
jet with the smallest delta R to the truth-level parton is kept; the others are
reset to unmatched (tag 0). This can reduce the number of matched jets in an
event: a branch that previously held multiple jet indices will end up with at
most one. Leptonic decay branches are already empty so disambiguation is a
no-op for them.

All three trees (reco, matching, truth) are written back in full; only the
six matching index branches are modified.

truth_decay ordering (verified from PDG IDs):
    index 0 → tag 1  (b from top containing W+)
    index 1 → tag 4  (b from top containing W-)
    index 2 → tag 2  (W+ decay product 1)
    index 3 → tag 3  (W+ decay product 2)
    index 4 → tag 5  (W- decay product 1)
    index 5 → tag 6  (W- decay product 2)

Usage:
    uv run src/data/root_unique_tags.py
"""

import numpy as np
import uproot
import awkward as ak
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
ROOT_PATH      = Path("data/topquarkreconstruction/root_data/ttbarLO_inclusive_mostof20M.root")
ROOT_SAVE_PATH = Path("data/topquarkreconstruction/root_data/ttbarLO_inclusive_unique_tags.root")
MAX_PARTICLES = 20
CHUNK_SIZE = 500_000

MATCHING_TAG_MAP = {
    "b_from_Wplus_jet_indices": 1,
    "Wplus_decay1_jet_indices": 2,
    "Wplus_decay2_jet_indices": 3,
    "b_from_Wminus_jet_indices": 4,
    "Wminus_decay1_jet_indices": 5,
    "Wminus_decay2_jet_indices": 6,
}

# truth_decay array index → parton tag
TRUTH_IDX_TO_TAG = {0: 1, 1: 4, 2: 2, 3: 3, 4: 5, 5: 6}


def disambiguate_tags(tags_np, eta_np, phi_np, truth_eta, truth_phi):
    """
    For each parton, if more than one jet shares the same tag, keep only the
    jet with the minimum delta R to the truth-level parton; reset the rest to 0.

    Args:
        tags_np:    (N, P) int32
        eta_np:     (N, P) float32  — NaN for padding slots
        phi_np:     (N, P) float32  — NaN for padding slots
        truth_eta:  (N, 6) float32
        truth_phi:  (N, 6) float32
    """
    for truth_idx, tag in TRUTH_IDX_TO_TAG.items():
        tag_mask = tags_np == tag          # (N, P) bool
        n_tagged = tag_mask.sum(axis=1)    # (N,)
        multi_events = np.where(n_tagged > 1)[0]

        if len(multi_events) == 0:
            continue

        for ev in multi_events:
            jet_cols = np.where(tag_mask[ev])[0]

            p_eta = truth_eta[ev, truth_idx]
            p_phi = truth_phi[ev, truth_idx]

            d_eta = eta_np[ev, jet_cols] - p_eta
            d_phi = phi_np[ev, jet_cols] - p_phi
            d_phi = (d_phi + np.pi) % (2.0 * np.pi) - np.pi   # wrap to [-π, π]

            delta_r = np.sqrt(d_eta**2 + d_phi**2)
            best = int(np.nanargmin(delta_r))

            for i, ji in enumerate(jet_cols):
                if i != best:
                    tags_np[ev, ji] = 0

    return tags_np


def tags_to_matching_arrays(tags_np):
    """
    Reconstruct per-branch jagged index arrays from the disambiguated tags array.

    For each parton branch, each event gets 0 or 1 jet index (the unique best match).
    Returns a dict {branch_name: list-of-lists-of-int}.
    """
    corrected = {}
    for branch_name, tag_value in MATCHING_TAG_MAP.items():
        indices = []
        for ev_tags in tags_np:
            jet_cols = np.where(ev_tags == tag_value)[0]
            indices.append(jet_cols.tolist())
        corrected[branch_name] = indices
    return corrected


def process_chunk(reco_arrays, matching_arrays, truth_arrays,
                  max_particles=MAX_PARTICLES):
    r = {k: reco_arrays[k] for k in reco_arrays.fields}
    m = {k: matching_arrays[k] for k in matching_arrays.fields}
    t = {k: truth_arrays[k] for k in truth_arrays.fields}

    n_events = len(r["jet_eta"])

    # ── 2. Pad eta/phi for delta-R disambiguation ───────────────────────────
    def pad_to_numpy(arr):
        padded = ak.pad_none(arr, max_particles, clip=True, axis=1)
        padded = ak.fill_none(padded, np.nan)
        return ak.to_numpy(padded).astype(np.float32)

    eta_np = pad_to_numpy(r["jet_eta"])
    phi_np = pad_to_numpy(r["jet_phi"])

    # ── 3. Build tags (all matched jets initially get the parton tag) ───────
    tags_np = np.zeros((n_events, max_particles), dtype=np.int32)

    for branch_name, tag_value in MATCHING_TAG_MAP.items():
        indices = m[branch_name]
        flat_jet_indices = ak.flatten(indices)
        if len(flat_jet_indices) == 0:
            continue

        event_indices = np.repeat(
            np.arange(n_events),
            ak.to_numpy(ak.num(indices))
        )
        flat_jet_np = ak.to_numpy(flat_jet_indices).astype(np.int64)

        valid = (flat_jet_np >= 0) & (flat_jet_np < max_particles)
        tags_np[event_indices[valid], flat_jet_np[valid]] = tag_value

    # ── 4. Disambiguate: keep only the min-delta-R jet per parton ──────────
    # truth_decay shape (N, 6), always exactly 6 entries for hadronic events
    truth_eta = ak.to_numpy(t["truth_decay_eta"]).astype(np.float32)
    truth_phi = ak.to_numpy(t["truth_decay_phi"]).astype(np.float32)

    tags_np = disambiguate_tags(tags_np, eta_np, phi_np, truth_eta, truth_phi)

    # ── 5. Rebuild corrected matching index arrays ──────────────────────────
    corrected_matching = tags_to_matching_arrays(tags_np)

    return corrected_matching, n_events


def main():
    ROOT_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    f = uproot.open(ROOT_PATH)
    reco_tree     = f["reco;1"]
    matching_tree = f["matching;1"]
    truth_tree    = f["truth;1"]

    total_entries = reco_tree.num_entries
    print(f"Total events in ROOT file: {total_entries:,}")
    print(f"ROOT output: {ROOT_SAVE_PATH}")

    # Read all branches from each tree
    reco_all_branches     = reco_tree.keys()
    matching_all_branches = matching_tree.keys()
    truth_all_branches    = truth_tree.keys()

    total_kept = 0
    trees_created = False

    with uproot.recreate(ROOT_SAVE_PATH) as root_out:
        for start in range(0, total_entries, CHUNK_SIZE):
            stop = min(start + CHUNK_SIZE, total_entries)
            print(f"\nProcessing {start:,} – {stop:,} / {total_entries:,} ...")

            # Read everything
            reco_arrays     = reco_tree.arrays(entry_start=start, entry_stop=stop)
            matching_arrays = matching_tree.arrays(entry_start=start, entry_stop=stop)
            truth_arrays    = truth_tree.arrays(entry_start=start, entry_stop=stop)

            corrected_matching, n_kept = process_chunk(
                reco_arrays, matching_arrays, truth_arrays
            )

            print(f"  Processed {n_kept:,} events")

            # Replace the index branches in matching with corrected versions
            matching_out = {b: matching_arrays[b] for b in matching_all_branches
                            if b not in MATCHING_TAG_MAP}
            for b in MATCHING_TAG_MAP:
                matching_out[b] = ak.Array(corrected_matching[b])

            # Create trees on first chunk
            if not trees_created:
                root_out.mktree("reco",     {b: reco_arrays[b].type    for b in reco_all_branches})
                root_out.mktree("matching", {b: matching_out[b].type   for b in matching_all_branches})
                root_out.mktree("truth",    {b: truth_arrays[b].type   for b in truth_all_branches})
                trees_created = True

            root_out["reco"].extend(    {b: reco_arrays[b]    for b in reco_all_branches})
            root_out["matching"].extend(matching_out)
            root_out["truth"].extend(   {b: truth_arrays[b]   for b in truth_all_branches})

            total_kept += n_kept
            print(f"  Running total: {total_kept:,}")

    print(f"\nDone. {total_kept:,} events → {ROOT_SAVE_PATH}")


if __name__ == "__main__":
    main()
