"""
Convert ttbar semi-leptonic ROOT file to raw HDF5 format for preprocessing.py.

Skims for semi-leptonic events (exactly 1 lepton at reco level) and constructs:
  - Unified jet array:  [pt, eta, phi, E, m, btag, charge=0, lep_type=0, truthtag]
  - Lepton array:       [pt, eta, phi, E, m=0, btag=0, charge, lep_type, truthtag=7]
  - Unified particle array written as 'jet' with particle_type sidecar for the preprocessor
  - MET:               [MET_pt, MET_phi]
  - neutrino_pz_truth: [pz]           (from parton-level truth)
  - event:             [n_jets, n_bjets, n_leptons, MET_pt, all_matched]

Unified truthmatch tag convention:
    1 = b from hadronic top
    2 = hadronic W decay product 1 (jet)
    3 = hadronic W decay product 2 (jet)
    4 = b from leptonic top
    7 = lepton (from leptonic W)    — NEW vs hadronic convention
    0 = unmatched jet / no tag

Usage:
    uv run src/data/root_to_h5_leptonic.py

CONFIGURE: Review the branch name maps below before first use.
The exact names depend on your analysis framework / ntuple production.
"""

import numpy as np
import h5py
import uproot
import awkward as ak
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
ROOT_PATH = Path("data/topquarkreconstruction/root_data/ttbar_semileptonic.root")
SAVE_DIR  = Path("data/topquarkreconstruction/h5py_data/semileptonic")
SAVE_PREFIX = "ttbar_semilepraw_"
MAX_JETS    = 20   # max padded jets (NOT counting the lepton)
CHUNK_SIZE  = 500_000

# ── CONFIGURE: reco branches ────────────────────────────────────────────────
RECO_BRANCHES = [
    "jet_pt", "jet_eta", "jet_phi", "jet_mass", "jet_btag",
    "el_pt",  "el_eta",  "el_phi",  "el_mass",  "el_charge",
    "mu_pt",  "mu_eta",  "mu_phi",  "mu_mass",  "mu_charge",
    "met_pt", "met_phi",
]

# ── CONFIGURE: matching branches → truthtag value ────────────────────────────
# "had" = hadronic top/W, "lep" = leptonic top
# Adjust branch names to match your ntuple schema.
HAD_MATCHING_TAG_MAP = {
    "had_b_jet_indices":      1,   # b from hadronic top
    "had_W_decay1_jet_indices": 2, # hadronic W daughter 1
    "had_W_decay2_jet_indices": 3, # hadronic W daughter 2
    "lep_b_jet_indices":      4,   # b from leptonic top
}
# Truth neutrino pz branch (parton level)
NU_PZ_BRANCH = "truth_nu_pz"   # scalar per event; set to None if unavailable


def compute_energy(pt, eta, mass):
    p = pt * np.cosh(eta)
    return np.sqrt(p ** 2 + mass ** 2)


def pad_to_numpy(arr, max_len, nan_fill=np.nan):
    """Pad awkward array to fixed length, NaN-fill."""
    padded = ak.pad_none(arr, max_len, clip=True, axis=1)
    padded = ak.fill_none(padded, nan_fill)
    return ak.to_numpy(padded).astype(np.float32)


def process_chunk(reco_arrays, matching_arrays, nu_pz_array=None,
                  max_jets=MAX_JETS):
    """
    Process one chunk of events: filter semi-leptonic, build arrays.

    Returns:
        jet_unified:    [N, max_jets+1, 9]   jets then lepton, 9 features
        particle_type:  [N, max_jets+1]       0=jet, 1=lepton
        MET:            [N, 2]                [MET_pt, MET_phi]
        neutrino_pz:    [N, 1]                truth nu pz (or zeros)
        event_array:    [N, 5]
    """
    r = reco_arrays
    m = matching_arrays

    # ── 1. Semi-leptonic filter: exactly 1 lepton ──
    n_el = ak.num(r["el_pt"])
    n_mu = ak.num(r["mu_pt"])
    slep_mask = ((n_el == 1) & (n_mu == 0)) | ((n_el == 0) & (n_mu == 1))

    if ak.sum(slep_mask) == 0:
        return None, None, None, None, None, 0

    r = {k: r[k][slep_mask] for k in r.fields}
    m = {k: m[k][slep_mask] for k in m.fields}
    if nu_pz_array is not None:
        nu_pz_array = nu_pz_array[slep_mask]

    n_events = int(ak.sum(slep_mask))

    # ── 2. Jet features ──
    pt_np   = pad_to_numpy(r["jet_pt"],   max_jets)
    eta_np  = pad_to_numpy(r["jet_eta"],  max_jets)
    phi_np  = pad_to_numpy(r["jet_phi"],  max_jets)
    mass_np = pad_to_numpy(r["jet_mass"], max_jets)
    btag_np = pad_to_numpy(r["jet_btag"], max_jets)
    jet_e   = compute_energy(pt_np, eta_np, mass_np)

    # ── 3. Jet truthmatch tags ──
    jet_tags = np.zeros((n_events, max_jets), dtype=np.int32)
    event_indices_base = np.repeat(np.arange(n_events), ak.to_numpy(ak.num(r["jet_pt"])).clip(0, max_jets))
    # Limit to max_jets entries per event
    for branch_name, tag_value in HAD_MATCHING_TAG_MAP.items():
        if branch_name not in m.fields:
            continue
        indices = m[branch_name]
        flat_jet_idx = ak.to_numpy(ak.flatten(indices)).astype(np.int64)
        ev_idx = np.repeat(np.arange(n_events), ak.to_numpy(ak.num(indices)))
        valid = (flat_jet_idx >= 0) & (flat_jet_idx < max_jets)
        jet_tags[ev_idx[valid], flat_jet_idx[valid]] = tag_value

    # Jet unified feature vector: [pt, eta, phi, E, m, btag, charge=0, lep_type=0, tag]
    jet_arr = np.stack([
        pt_np, eta_np, phi_np, jet_e, mass_np, btag_np,
        np.zeros((n_events, max_jets), dtype=np.float32),  # charge placeholder
        np.zeros((n_events, max_jets), dtype=np.float32),  # lepton_type placeholder
        jet_tags.astype(np.float32),
    ], axis=-1)  # [N, max_jets, 9]

    # ── 4. Lepton features ──
    # Pick whichever lepton type is present
    use_electron = (ak.num(r["el_pt"]) == 1)
    el_present = ak.to_numpy(use_electron)   # [N] bool

    lep_pt   = np.where(el_present, ak.to_numpy(ak.pad_none(r["el_pt"],  1, clip=True, axis=1)[:, 0]).filled(np.nan),
                                    ak.to_numpy(ak.pad_none(r["mu_pt"],  1, clip=True, axis=1)[:, 0]).filled(np.nan)).astype(np.float32)
    lep_eta  = np.where(el_present, ak.to_numpy(ak.pad_none(r["el_eta"], 1, clip=True, axis=1)[:, 0]).filled(np.nan),
                                    ak.to_numpy(ak.pad_none(r["mu_eta"], 1, clip=True, axis=1)[:, 0]).filled(np.nan)).astype(np.float32)
    lep_phi  = np.where(el_present, ak.to_numpy(ak.pad_none(r["el_phi"], 1, clip=True, axis=1)[:, 0]).filled(np.nan),
                                    ak.to_numpy(ak.pad_none(r["mu_phi"], 1, clip=True, axis=1)[:, 0]).filled(np.nan)).astype(np.float32)
    lep_mass = np.where(el_present, ak.to_numpy(ak.pad_none(r["el_mass"],1, clip=True, axis=1)[:, 0]).filled(0.000511),
                                    ak.to_numpy(ak.pad_none(r["mu_mass"],1, clip=True, axis=1)[:, 0]).filled(0.10566)  ).astype(np.float32)
    lep_charge = np.where(el_present,
                          ak.to_numpy(ak.pad_none(r["el_charge"],1,clip=True,axis=1)[:,0]).filled(0),
                          ak.to_numpy(ak.pad_none(r["mu_charge"],1,clip=True,axis=1)[:,0]).filled(0)).astype(np.float32)
    lep_type = np.where(el_present, 1.0, 2.0).astype(np.float32)  # 1=electron, 2=muon
    lep_e    = compute_energy(lep_pt, lep_eta, lep_mass)

    # Lepton unified feature vector: [pt, eta, phi, E, m≈0, btag=0, charge, lep_type, tag=7]
    lep_arr = np.stack([
        lep_pt, lep_eta, lep_phi, lep_e, lep_mass,
        np.zeros(n_events, dtype=np.float32),          # btag = 0
        lep_charge,
        lep_type,
        np.full(n_events, 7.0, dtype=np.float32),      # truthtag = 7
    ], axis=-1)[:, None, :]  # [N, 1, 9]

    # ── 5. Unified array: jets then lepton ──
    jet_unified   = np.concatenate([jet_arr, lep_arr], axis=1)   # [N, max_jets+1, 9]
    particle_type = np.concatenate([
        np.zeros((n_events, max_jets), dtype=np.uint8),
        np.ones( (n_events, 1),        dtype=np.uint8),
    ], axis=1)  # [N, max_jets+1]

    # ── 6. MET ──
    met_pt  = ak.to_numpy(r["met_pt"]).astype(np.float32)
    met_phi = ak.to_numpy(r["met_phi"]).astype(np.float32)
    met_arr = np.stack([met_pt, met_phi], axis=-1)  # [N, 2]

    # ── 7. Neutrino truth pz ──
    if nu_pz_array is not None:
        nu_pz = ak.to_numpy(nu_pz_array[slep_mask if isinstance(slep_mask, type(nu_pz_array)) else slep_mask]
                            ).astype(np.float32).reshape(-1, 1)
    else:
        nu_pz = np.zeros((n_events, 1), dtype=np.float32)

    # ── 8. Event array ──
    n_jets_np  = ak.to_numpy(ak.num(r["jet_pt"])).astype(np.float32).clip(0, max_jets)
    btag_count = np.nansum(btag_np > 0, axis=1).astype(np.float32)
    n_lep_np   = np.ones(n_events, dtype=np.float32)  # always 1 by construction

    # all_matched: 1 if all 4 jet tags present (had_b, had_W1, had_W2, lep_b)
    all_matched = np.ones(n_events, dtype=np.float32)
    for tag_val in [1, 2, 3, 4]:
        has_tag = np.any(jet_tags == tag_val, axis=1)
        all_matched *= has_tag.astype(np.float32)

    event_array = np.stack([n_jets_np, btag_count, n_lep_np, met_pt, all_matched], axis=-1)

    return jet_unified, particle_type, met_arr, nu_pz, event_array, n_events


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Opening ROOT file: {ROOT_PATH}")
    f = uproot.open(ROOT_PATH)
    reco_tree    = f["reco;1"]
    matching_tree = f["matching;1"]

    total_entries = reco_tree.num_entries
    print(f"Total events: {total_entries:,}")

    matching_branches = [b for b in HAD_MATCHING_TAG_MAP.keys() if b in matching_tree.keys()]
    if NU_PZ_BRANCH and NU_PZ_BRANCH in reco_tree.keys():
        reco_branches = RECO_BRANCHES + [NU_PZ_BRANCH]
        has_nu_pz = True
    else:
        reco_branches = RECO_BRANCHES
        has_nu_pz = False
        print(f"[WARN] {NU_PZ_BRANCH} not found — neutrino_pz will be zeros")

    save_path = SAVE_DIR / f"{SAVE_PREFIX}train.h5"
    print(f"Output: {save_path}")

    total_kept = 0
    datasets_created = False

    with h5py.File(save_path, "w") as hf:
        for start in range(0, total_entries, CHUNK_SIZE):
            stop = min(start + CHUNK_SIZE, total_entries)
            print(f"\nProcessing {start:,} - {stop:,} / {total_entries:,} ...")

            reco_arrays    = reco_tree.arrays(reco_branches, entry_start=start, entry_stop=stop)
            matching_arrays = matching_tree.arrays(matching_branches, entry_start=start, entry_stop=stop)
            nu_pz_array    = reco_arrays[NU_PZ_BRANCH] if has_nu_pz else None

            result = process_chunk(reco_arrays, matching_arrays, nu_pz_array)
            jet_unified, particle_type, met_arr, nu_pz, event_array, n_kept = result

            if n_kept == 0:
                print("  No semi-leptonic events in chunk, skipping")
                continue

            print(f"  Kept {n_kept:,} semi-leptonic events")

            if not datasets_created:
                _, P, F = jet_unified.shape
                _, E    = event_array.shape
                hf.create_dataset("jet",           shape=(0, P, F),  maxshape=(None, P, F),  compression="gzip", compression_opts=4, dtype="float32")
                hf.create_dataset("particle_type", shape=(0, P),     maxshape=(None, P),     compression="gzip", compression_opts=4, dtype="uint8")
                hf.create_dataset("MET",           shape=(0, 2),     maxshape=(None, 2),     compression="gzip", compression_opts=4, dtype="float32")
                hf.create_dataset("neutrino_pz_truth", shape=(0, 1), maxshape=(None, 1),     compression="gzip", compression_opts=4, dtype="float32")
                hf.create_dataset("event",         shape=(0, E),     maxshape=(None, E),     compression="gzip", compression_opts=4, dtype="float32")
                datasets_created = True

            cur = hf["jet"].shape[0]
            new_len = cur + n_kept
            for key, arr in [("jet", jet_unified), ("particle_type", particle_type),
                             ("MET", met_arr), ("neutrino_pz_truth", nu_pz),
                             ("event", event_array)]:
                hf[key].resize((new_len,) + hf[key].shape[1:])
                hf[key][cur:new_len] = arr

            total_kept += n_kept
            print(f"  Running total: {total_kept:,} events")

    print(f"\nDone! {total_kept:,} semi-leptonic events → {save_path}")
    print(f"  jet (unified) shape: ({total_kept}, {MAX_JETS+1}, 9)")
    print(f"  particle_type shape: ({total_kept}, {MAX_JETS+1})")


if __name__ == "__main__":
    main()
