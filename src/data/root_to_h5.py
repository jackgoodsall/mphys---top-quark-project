"""
Convert ttbarLO_inclusive ROOT file to raw HDF5 format for preprocessing.py.

Skims for all-hadronic events (0 leptons at reco level) and constructs
truthmatch tags from the matching tree indices.

Truthmatch tag convention (matching existing data):
    1 = b from top containing W+
    2 = W+ decay product 1 (jet)
    3 = W+ decay product 2 (jet)
    4 = b from top containing W-
    5 = W- decay product 1 (jet)
    6 = W- decay product 2 (jet)
    0 = unmatched jet

Usage:
    uv run src/data/root_to_h5.py
"""

import numpy as np
import h5py
import uproot
import awkward as ak
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
ROOT_PATH = Path("data/topquarkreconstruction/root_data/ttbarLO_inclusive_mostof20M.root")
SAVE_DIR = Path("data/topquarkreconstruction/h5py_data")
SAVE_PREFIX = "ttbar_h5py_raw_"
MAX_PARTICLES = 20
CHUNK_SIZE = 500_000  # events per read chunk

# Matching branch → truthmatch tag mapping
MATCHING_TAG_MAP = {
    "b_from_Wplus_jet_indices": 1,     # b from top1
    "Wplus_decay1_jet_indices": 2,     # W+ decay1
    "Wplus_decay2_jet_indices": 3,     # W+ decay2
    "b_from_Wminus_jet_indices": 4,    # b from top2
    "Wminus_decay1_jet_indices": 5,    # W- decay1
    "Wminus_decay2_jet_indices": 6,    # W- decay2
}


def compute_energy(pt, eta, mass):
    """E = sqrt((pt * cosh(eta))^2 + m^2)"""
    p = pt * np.cosh(eta)
    return np.sqrt(p**2 + mass**2)


def process_chunk(reco_arrays, matching_arrays, max_particles=MAX_PARTICLES):
    """
    Process a chunk of events: filter all-hadronic, build jet features + tags.

    Returns:
        jet_array: (N_had, max_particles, 7) with NaN padding
        event_array: (N_had, 3) [njet, nbtag, all_matched]
        n_kept: number of all-hadronic events
    """
    # ── 1. All-hadronic filter: 0 electrons AND 0 muons ──
    n_el = ak.num(reco_arrays["el_pt"])
    n_mu = ak.num(reco_arrays["mu_pt"])
    had_mask = (n_el == 0) & (n_mu == 0)

    if ak.sum(had_mask) == 0:
        return None, None, 0

    # Apply filter
    r = {k: reco_arrays[k][had_mask] for k in reco_arrays.fields}
    m = {k: matching_arrays[k][had_mask] for k in matching_arrays.fields}

    n_events = int(ak.sum(had_mask))

    # ── 2. Build jet features ──
    n_jets_per_event = ak.num(r["jet_pt"])

    # Pad and convert to numpy
    def pad_to_numpy(arr):
        padded = ak.pad_none(arr, max_particles, clip=True, axis=1)
        padded = ak.fill_none(padded, np.nan)
        return ak.to_numpy(padded).astype(np.float32)

    pt_np = pad_to_numpy(r["jet_pt"])
    eta_np = pad_to_numpy(r["jet_eta"])
    phi_np = pad_to_numpy(r["jet_phi"])
    mass_np = pad_to_numpy(r["jet_mass"])
    btag_np = pad_to_numpy(r["jet_btag"])

    # Compute energy: E = sqrt((pt * cosh(eta))^2 + m^2)
    # NaN propagates naturally through arithmetic
    jet_e = compute_energy(pt_np, eta_np, mass_np)

    # ── 3. Construct truthmatch tags from matching indices (vectorized) ──
    # Plain integer encoding (0-6) matching preprocessing.py convention.
    # If a jet is matched to multiple partons (rare), the last tag wins.
    tags_np = np.zeros((n_events, max_particles), dtype=np.int32)

    for branch_name, tag_value in MATCHING_TAG_MAP.items():
        indices = m[branch_name]  # awkward array of variable-length lists

        # Build (event_idx, jet_idx) pairs from the variable-length lists
        flat_jet_indices = ak.flatten(indices)
        if len(flat_jet_indices) == 0:
            continue

        # Event index for each flattened entry
        event_indices = np.repeat(
            np.arange(n_events),
            ak.to_numpy(ak.num(indices))
        )
        flat_jet_np = ak.to_numpy(flat_jet_indices).astype(np.int64)

        # Filter to valid range
        valid = (flat_jet_np >= 0) & (flat_jet_np < max_particles)
        tags_np[event_indices[valid], flat_jet_np[valid]] = tag_value

    # ── 4. Stack into jet array: [pt, eta, phi, E, m, btag, truthmatch] ──
    jet_array = np.stack([pt_np, eta_np, phi_np, jet_e, mass_np, btag_np, tags_np], axis=-1)

    # ── 5. Build event-level features: [njet, nbtag, all_matched] ──
    n_jets_np = ak.to_numpy(n_jets_per_event).astype(np.float32)

    # Count b-tagged jets per event
    btag_counts = np.nansum(btag_np > 0, axis=1).astype(np.float32)

    # all_matched: 1 if all 6 tags present, 0 otherwise
    all_matched = np.ones(n_events, dtype=np.float32)
    for tag_val in range(1, 7):
        has_tag = np.any(tags_np == tag_val, axis=1)
        all_matched *= has_tag.astype(np.float32)

    event_array = np.stack([n_jets_np, btag_counts, all_matched], axis=-1)

    return jet_array, event_array, n_events


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    f = uproot.open(ROOT_PATH)
    reco_tree = f["reco;1"]
    matching_tree = f["matching;1"]

    total_entries = reco_tree.num_entries
    print(f"Total events in ROOT file: {total_entries}")

    # Branches to read
    reco_branches = ["jet_pt", "jet_eta", "jet_phi", "jet_mass", "jet_btag",
                     "el_pt", "mu_pt"]
    matching_branches = list(MATCHING_TAG_MAP.keys())

    save_path = SAVE_DIR / f"{SAVE_PREFIX}train.h5"
    print(f"Output: {save_path}")

    total_kept = 0
    datasets_created = False

    with h5py.File(save_path, "w") as hf:
        for start in range(0, total_entries, CHUNK_SIZE):
            stop = min(start + CHUNK_SIZE, total_entries)
            print(f"\nProcessing events {start:,} - {stop:,} / {total_entries:,} ...")

            reco_arrays = reco_tree.arrays(reco_branches, entry_start=start, entry_stop=stop)
            matching_arrays = matching_tree.arrays(matching_branches, entry_start=start, entry_stop=stop)

            jet_chunk, event_chunk, n_kept = process_chunk(reco_arrays, matching_arrays)

            if n_kept == 0:
                print(f"  No all-hadronic events in this chunk, skipping")
                continue

            print(f"  Kept {n_kept:,} all-hadronic events")

            if not datasets_created:
                _, P, F = jet_chunk.shape
                _, E = event_chunk.shape
                hf.create_dataset("jet", shape=(0, P, F), maxshape=(None, P, F),
                                  compression="gzip", compression_opts=4, dtype="float32")
                hf.create_dataset("event", shape=(0, E), maxshape=(None, E),
                                  compression="gzip", compression_opts=4, dtype="float32")
                datasets_created = True

            cur = hf["jet"].shape[0]
            new_len = cur + n_kept
            hf["jet"].resize((new_len,) + hf["jet"].shape[1:])
            hf["event"].resize((new_len,) + hf["event"].shape[1:])
            hf["jet"][cur:new_len] = jet_chunk
            hf["event"][cur:new_len] = event_chunk

            total_kept += n_kept
            print(f"  Running total: {total_kept:,} events")

    print(f"\nDone! Saved {total_kept:,} all-hadronic events to {save_path}")
    print(f"  jet shape: ({total_kept}, {MAX_PARTICLES}, 7)")
    print(f"  event shape: ({total_kept}, 3)")


if __name__ == "__main__":
    main()
