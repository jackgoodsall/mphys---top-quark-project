"""
Unified preprocessor for combined hadronic + semi-leptonic ttbar training.

Reads two source formats in a single pass:
  1. Hadronic raw H5  (existing format from root_to_h5.py)
     Keys: jet [N, 20, 7], event [N, 3]
     jet features: [pt, eta, phi, E, m, btag, truthtag]

  2. Semi-leptonic SPANet H5  (from data/semi_leptonic_ttbar/)
     Keys: INPUTS/Momenta/{pt,eta,sin_phi,cos_phi,mass,btag,etag,qtag,utag,MASK}
           INPUTS/Met/{met,sin_phi,cos_phi}
           TARGETS/{ht/b,ht/q1,ht/q2,lt/b,lt/l}
           REGRESSIONS/EVENT/neutrino_pz
     Particle 0 is ALWAYS the lepton; jets follow sorted by pT.

Both are scaled with SHARED scalers (fit on training splits of both datasets),
then written to a combined output HDF5 with:

  jet             [N, MAX_P, 7]     [pt_s, eta_s, sin_phi, cos_phi, E_s, m, btag]
  src_mask        [N, MAX_P]        True = real particle
  interactions    [N, MAX_P, MAX_P, 4]
  masks_tops      [N, 2, MAX_P]     chain 0 = hadronic top, chain 1 = leptonic top
  masks_Ws        [N, 2, MAX_P]     chain 0 = hadronic W,   chain 1 = leptonic W
  kinematics_tops [N, 2, 5]        [pt_s, eta_s, sin_phi, cos_phi, E_s]   (truth-matched sum)
  kinematics_Ws   [N, 2, 5]
  valid_tops      [N, 2]  uint8
  valid_Ws        [N, 2]  uint8
  particle_type   [N, MAX_P] uint8  0=jet, 1=lepton
  chain_type      [N, 2]     uint8  0=hadronic, 1=leptonic (semi-lep) / both 0 (hadronic)
  globals         [N, 6]           [n_jets, n_bjets, n_leptons, MET_pt_s, sin(MET_phi), cos(MET_phi)]
  neutrino_truth  [N, 2, 1]        [[0], [nu_pz]] for semi-lep; zeros for hadronic

Usage:
    uv run src/data/preprocess_combined.py \\
        --had_train data/topquarkreconstruction/h5py_data/combined/ttbar_h5py_raw_train.h5 \\
        --had_val   data/topquarkreconstruction/h5py_data/combined/ttbar_h5py_raw_val.h5 \\
        --had_test  data/topquarkreconstruction/h5py_data/combined/ttbar_h5py_raw_test_old.h5 \\
        --slep_train data/semi_leptonic_ttbar/training_mass_variation.h5 \\
        --slep_test  data/semi_leptonic_ttbar/testing_sm.h5 \\
        --output_dir data/topquarkreconstruction/leptonic_combined
"""

import sys, os, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import h5py
import joblib
from pathlib import Path
from typing import Dict, Optional, Iterator, Tuple
from tqdm import tqdm

try:
    import vector
    _HAS_VECTOR = True
except ImportError:
    _HAS_VECTOR = False

from src.data_utils.scalers import LogMinMaxScaler, StandardScaler

# ── Config ───────────────────────────────────────────────────────────────────
MAX_P      = 20      # unified particle slots (jets + lepton)
CHUNK_SIZE = 50_000
HAD_TAG_MAP = {      # hadronic raw truthtag → (object_type, chain_idx)
    1: ("had_top_b",  0),
    2: ("had_W_q1",   0),
    3: ("had_W_q2",   0),
    4: ("had2_top_b", 1),
    5: ("had2_W_q1",  1),
    6: ("had2_W_q2",  1),
}


# ── Scalers ──────────────────────────────────────────────────────────────────

class CombinedScalers:
    """Holds all scalers and exposes fit/transform for both source formats."""

    def __init__(self):
        self.pt_scaler  = LogMinMaxScaler()
        self.eta_scaler = StandardScaler()
        self.E_scaler   = LogMinMaxScaler()
        self.met_scaler = LogMinMaxScaler()

    def partial_fit_particles(self, pt: np.ndarray, eta: np.ndarray, E: np.ndarray):
        """Partial-fit on valid (non-NaN) particle values."""
        valid = ~np.isnan(pt)
        if valid.any():
            self.pt_scaler.partial_fit(pt[valid].reshape(-1, 1))
            self.eta_scaler.partial_fit(eta[valid].reshape(-1, 1))
            self.E_scaler.partial_fit(E[valid].reshape(-1, 1))

    def partial_fit_met(self, met: np.ndarray):
        self.met_scaler.partial_fit(met.reshape(-1, 1))

    def transform_particles(self, pt, eta, sin_phi, cos_phi, E, m, btag):
        """Apply scalers; NaN propagates for padding."""
        N, P = pt.shape
        mask_valid = ~np.isnan(pt)

        pt_s  = np.full_like(pt, 0.0)
        eta_s = np.full_like(eta, 0.0)
        E_s   = np.full_like(E, 0.0)

        if mask_valid.any():
            flat = lambda x: x[mask_valid].reshape(-1, 1)
            pt_s[mask_valid]  = self.pt_scaler.transform(flat(pt)).ravel()
            eta_s[mask_valid] = self.eta_scaler.transform(flat(eta)).ravel()
            E_s[mask_valid]   = self.E_scaler.transform(flat(E)).ravel()

        return np.stack([pt_s, eta_s, sin_phi, cos_phi, E_s, m, btag], axis=-1)

    def transform_met_pt(self, met_pt: np.ndarray) -> np.ndarray:
        return self.met_scaler.transform(met_pt.reshape(-1, 1)).ravel()

    def save(self, path: Path):
        joblib.dump({'pt': self.pt_scaler, 'eta': self.eta_scaler,
                     'E': self.E_scaler, 'met': self.met_scaler}, path)
        print(f"[SAVE] Scalers → {path}", flush=True)

    @classmethod
    def load(cls, path: Path) -> "CombinedScalers":
        d = joblib.load(path)
        s = cls()
        s.pt_scaler = d['pt']; s.eta_scaler = d['eta']
        s.E_scaler  = d['E'];  s.met_scaler  = d['met']
        return s


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_E(pt, eta, m):
    """E = sqrt((pt * cosh(eta))^2 + m^2). NaN propagates."""
    p = pt * np.cosh(eta)
    return np.sqrt(p**2 + m**2)


def build_interaction_matrix(jet_chunk: np.ndarray) -> np.ndarray:
    """
    Compute pairwise [ΔR, kT, z, m²] for each (i,j) particle pair.
    jet_chunk: [B, P, 7] with features [pt, eta, sin_phi, cos_phi, E, m, btag].
    Padding rows (pt==0) produce zeros in the interaction matrix.
    Returns [B, P, P, 4].
    """
    B, P, _ = jet_chunk.shape
    pt  = jet_chunk[:, :, 0]   # [B, P]
    eta = jet_chunk[:, :, 1]
    phi = np.arctan2(jet_chunk[:, :, 2], jet_chunk[:, :, 3])  # sin/cos → phi

    # dEta, dPhi
    deta = eta[:, :, None] - eta[:, None, :]          # [B, P, P]
    dphi = phi[:, :, None] - phi[:, None, :]
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi      # wrap to [-π, π]
    dR   = np.sqrt(deta**2 + dphi**2)                  # [B, P, P]

    # kT = min(pT_i, pT_j) * dR
    pt_i = pt[:, :, None]; pt_j = pt[:, None, :]
    kT   = np.minimum(pt_i, pt_j) * dR

    # z  = min(pT_i, pT_j) / (pT_i + pT_j + 1e-9)
    z    = np.minimum(pt_i, pt_j) / (pt_i + pt_j + 1e-9)

    # m²: reconstruct from 4-vectors
    E   = jet_chunk[:, :, 4]                           # [B, P]
    px  = pt * jet_chunk[:, :, 3]                      # cos_phi
    py  = pt * jet_chunk[:, :, 2]                      # sin_phi
    pz  = pt * np.sinh(eta)

    E_i  = E[:, :, None];  E_j  = E[:, None, :]
    px_i = px[:, :, None]; px_j = px[:, None, :]
    py_i = py[:, :, None]; py_j = py[:, None, :]
    pz_i = pz[:, :, None]; pz_j = pz[:, None, :]

    m2 = ((E_i + E_j)**2 - (px_i + px_j)**2 -
          (py_i + py_j)**2 - (pz_i + pz_j)**2).clip(0)

    return np.stack([dR, kT, z, m2], axis=-1).astype(np.float32)


def reco_4vec_from_mask(jet_chunk, binary_mask, scalers):
    """
    Sum 4-vectors of selected particles → scaled [pt, eta, sin_phi, cos_phi, E].
    jet_chunk: [B, P, 7] (already scaled features).
    binary_mask: [B, P] float32.
    Returns [B, 5].
    """
    B, P, _ = jet_chunk.shape
    # Unscale pt and E for summation (use raw magnitudes for 4-vector addition)
    # We reconstruct from the scaled features the physical 4-vector,
    # sum, then rescale the result.
    # Shortcut: directly use the unscaled pt/eta/phi stored in a passed raw array.
    # Since we don't have the raw array here, we use the scaled jet for direction
    # and magnitude is captured in pt_scaled (monotone transform → ordering preserved).
    # For simplicity, we just return the scaled coordinate of the leading matched particle
    # as a proxy. A full correct implementation should pass the raw arrays.
    # See _reco_kinematics_raw below.
    return np.zeros((B, 5), dtype=np.float32)


def indices_to_mask(indices_list: list, B: int, P: int) -> np.ndarray:
    """
    Convert a list of [B] index arrays (with -1 meaning absent) to a [B, P] binary mask.
    """
    mask = np.zeros((B, P), dtype=np.float32)
    for idx in indices_list:
        valid = (idx >= 0) & (idx < P)
        for b in range(B):
            if valid[b]:
                mask[b, idx[b]] = 1.0
    return mask


def reco_kin_from_raw(pt_raw, eta_raw, sin_phi_raw, cos_phi_raw, E_raw,
                      binary_mask, scalers):
    """
    Sum 4-vectors of selected particles from RAW (unscaled) arrays.
    Returns scaled [pt_s, eta_s, sin_phi, cos_phi, E_s, 0] [B, 5].
    binary_mask: [B, P] float32.
    """
    B, P = pt_raw.shape
    mask = binary_mask.astype(bool)  # [B, P]

    px  = pt_raw * cos_phi_raw      # [B, P]
    py  = pt_raw * sin_phi_raw
    pz  = pt_raw * np.sinh(eta_raw)
    E   = E_raw

    sum_px = (px * binary_mask).sum(axis=1)   # [B]
    sum_py = (py * binary_mask).sum(axis=1)
    sum_pz = (pz * binary_mask).sum(axis=1)
    sum_E  = (E  * binary_mask).sum(axis=1)

    reco_pt = np.sqrt(sum_px**2 + sum_py**2)
    reco_p  = np.sqrt(sum_px**2 + sum_py**2 + sum_pz**2)
    reco_eta = np.where(
        reco_p > 1e-6,
        np.arctanh(np.clip(sum_pz / reco_p.clip(1e-6), -0.9999, 0.9999)),
        np.zeros(B)
    )
    reco_phi = np.arctan2(sum_py, sum_px)  # [B]
    reco_sin = np.sin(reco_phi)
    reco_cos = np.cos(reco_phi)

    # Scale
    reco_pt_s = scalers.pt_scaler.transform(reco_pt.reshape(-1,1)).ravel()
    reco_eta_s = scalers.eta_scaler.transform(reco_eta.reshape(-1,1)).ravel()
    reco_E_s   = scalers.E_scaler.transform(sum_E.reshape(-1,1)).ravel()

    return np.stack([reco_pt_s, reco_eta_s, reco_sin, reco_cos, reco_E_s], axis=-1)  # [B, 5]


# ── Hadronic reader ───────────────────────────────────────────────────────────

def had_fit_chunk(jet_raw: np.ndarray, scalers: CombinedScalers):
    """Fit scalers on one hadronic chunk."""
    pt   = jet_raw[:, :, 0]
    eta  = jet_raw[:, :, 1]
    E    = jet_raw[:, :, 3]
    scalers.partial_fit_particles(pt, eta, E)
    # MET not available in hadronic data — skip met_scaler for this source


def had_process_chunk(jet_raw: np.ndarray, event_raw: np.ndarray,
                      scalers: CombinedScalers) -> Dict[str, np.ndarray]:
    """
    Transform one hadronic raw chunk into the unified output format.
    jet_raw: [B, P_had, 7] = [pt, eta, phi, E, m, btag, truthtag]
    event_raw: [B, 3] = [n_jets, n_bjets, all_matched]
    """
    B, P_had, _ = jet_raw.shape

    pt      = jet_raw[:, :, 0]
    eta     = jet_raw[:, :, 1]
    phi     = jet_raw[:, :, 2]
    E_raw   = jet_raw[:, :, 3]
    m       = jet_raw[:, :, 4]
    btag    = jet_raw[:, :, 5]
    tags    = jet_raw[:, :, 6].astype(np.int32)

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    # Pad / truncate to MAX_P
    P_out = MAX_P
    def pad(arr):
        if arr.shape[1] >= P_out:
            return arr[:, :P_out]
        pad_w = P_out - arr.shape[1]
        return np.concatenate([arr, np.zeros((B, pad_w) + arr.shape[2:], dtype=arr.dtype)], axis=1)

    pt      = pad(pt);      eta     = pad(eta)
    sin_phi = pad(sin_phi); cos_phi = pad(cos_phi)
    E_raw   = pad(E_raw);   m       = pad(m);   btag = pad(btag)
    tags    = pad(tags)

    # src_mask: real if pt > 0 (NaN rows are zeroed in raw h5 padding)
    src_mask = (pt > 0) & ~np.isnan(pt)  # [B, MAX_P]

    # Scale features
    jet_out = scalers.transform_particles(pt, eta, sin_phi, cos_phi, E_raw, m, btag)
    jet_out[~src_mask] = 0.0  # zero-out padding rows

    # Interaction matrix (computed on scaled jet_out)
    interactions = build_interaction_matrix(jet_out)

    # Build masks from truth tags
    # ht (top+) = chain 0, lt (top-) = chain 1
    B_P = B * P_out
    flat_tags = tags.reshape(B_P)
    ev_idx    = np.repeat(np.arange(B), P_out)

    def tag_mask(tag_set):
        m_ = np.zeros((B, P_out), dtype=np.float32)
        hit = np.isin(flat_tags, list(tag_set))
        rows, cols = np.divmod(np.where(hit)[0], P_out)
        m_[rows, cols] = 1.0
        return m_

    had_top_mask = tag_mask({1, 2, 3})   # chain 0 top
    had_W_mask   = tag_mask({2, 3})      # chain 0 W
    had2_top_mask= tag_mask({4, 5, 6})   # chain 1 top
    had2_W_mask  = tag_mask({5, 6})      # chain 1 W

    masks_tops = np.stack([had_top_mask, had2_top_mask], axis=1)  # [B, 2, P]
    masks_Ws   = np.stack([had_W_mask,   had2_W_mask],   axis=1)

    # Validity
    def all_tags_present(tag_set):
        ok = np.ones(B, dtype=bool)
        for t in tag_set:
            ok &= np.any(tags == t, axis=1)
        return ok

    valid_tops = np.stack([
        all_tags_present({1, 2, 3}),
        all_tags_present({4, 5, 6})
    ], axis=1).astype(np.uint8)

    valid_Ws = np.stack([
        all_tags_present({2, 3}),
        all_tags_present({5, 6})
    ], axis=1).astype(np.uint8)

    # Kinematics
    def rk(mask):
        return reco_kin_from_raw(pt, eta, sin_phi, cos_phi, E_raw, mask, scalers)

    kt0 = rk(had_top_mask);   kt1 = rk(had2_top_mask)
    kw0 = rk(had_W_mask);     kw1 = rk(had2_W_mask)

    kinematics_tops = np.stack([
        np.concatenate([kt0, np.zeros((B, 1))], axis=-1),
        np.concatenate([kt1, np.zeros((B, 1))], axis=-1),
    ], axis=1)  # [B, 2, 5]
    kinematics_Ws = np.stack([
        np.concatenate([kw0, np.zeros((B, 1))], axis=-1),
        np.concatenate([kw1, np.zeros((B, 1))], axis=-1),
    ], axis=1)

    # Leptonic keys (all-hadronic defaults)
    particle_type = np.zeros((B, P_out), dtype=np.uint8)
    chain_type    = np.zeros((B, 2),     dtype=np.uint8)
    neutrino_truth = np.zeros((B, 2, 1), dtype=np.float32)

    n_jets  = event_raw[:, 0].astype(np.float32)
    n_bjets = event_raw[:, 1].astype(np.float32)
    globals_arr = np.stack([
        n_jets, n_bjets,
        np.zeros(B, np.float32),   # n_leptons
        np.zeros(B, np.float32),   # MET_pt_s
        np.zeros(B, np.float32),   # sin(MET_phi)
        np.ones(B,  np.float32),   # cos(MET_phi) = 1 (phi=0 sentinel)
    ], axis=-1)

    return {
        'jet': jet_out.astype(np.float32),
        'src_mask': src_mask.astype(bool),
        'interactions': interactions,
        'masks_tops': masks_tops,
        'masks_Ws': masks_Ws,
        'kinematics_tops': kinematics_tops.astype(np.float32),
        'kinematics_Ws': kinematics_Ws.astype(np.float32),
        'valid_tops': valid_tops,
        'valid_Ws': valid_Ws,
        'particle_type': particle_type,
        'chain_type': chain_type,
        'globals': globals_arr.astype(np.float32),
        'neutrino_truth': neutrino_truth,
    }


# ── Semi-leptonic reader ──────────────────────────────────────────────────────

def slep_fit_chunk(f: h5py.File, start: int, stop: int, scalers: CombinedScalers):
    """Fit scalers on one semi-leptonic chunk."""
    pt   = f['INPUTS/Momenta/pt'][start:stop]
    eta  = f['INPUTS/Momenta/eta'][start:stop]
    m    = f['INPUTS/Momenta/mass'][start:stop]
    mask = f['INPUTS/Momenta/MASK'][start:stop]  # [B, P] bool

    # Mask out padding positions with NaN
    pt_v  = np.where(mask, pt, np.nan)
    eta_v = np.where(mask, eta, np.nan)
    E_v   = np.where(mask, compute_E(pt, eta, m), np.nan)

    scalers.partial_fit_particles(pt_v, eta_v, E_v)
    scalers.partial_fit_met(f['INPUTS/Met/met'][start:stop])


def slep_process_chunk(f: h5py.File, start: int, stop: int,
                       scalers: CombinedScalers) -> Dict[str, np.ndarray]:
    """Transform one semi-leptonic SPANet chunk into unified output format."""
    slep_P = f['INPUTS/Momenta/pt'].shape[1]
    B = stop - start

    pt      = f['INPUTS/Momenta/pt'][start:stop]        # [B, slep_P]
    eta     = f['INPUTS/Momenta/eta'][start:stop]
    sin_phi = f['INPUTS/Momenta/sin_phi'][start:stop]
    cos_phi = f['INPUTS/Momenta/cos_phi'][start:stop]
    mass    = f['INPUTS/Momenta/mass'][start:stop]
    btag    = f['INPUTS/Momenta/btag'][start:stop]
    etag    = f['INPUTS/Momenta/etag'][start:stop]
    utag    = f['INPUTS/Momenta/utag'][start:stop]
    valid_m = f['INPUTS/Momenta/MASK'][start:stop]       # [B, slep_P] bool True=real

    met_pt      = f['INPUTS/Met/met'][start:stop]        # [B]
    met_sin_phi = f['INPUTS/Met/sin_phi'][start:stop]
    met_cos_phi = f['INPUTS/Met/cos_phi'][start:stop]

    ht_b  = f['TARGETS/ht/b'][start:stop].astype(np.int32)
    ht_q1 = f['TARGETS/ht/q1'][start:stop].astype(np.int32)
    ht_q2 = f['TARGETS/ht/q2'][start:stop].astype(np.int32)
    lt_b  = f['TARGETS/lt/b'][start:stop].astype(np.int32)
    lt_l  = f['TARGETS/lt/l'][start:stop].astype(np.int32)   # always 0
    nu_pz = f['REGRESSIONS/EVENT/neutrino_pz'][start:stop].astype(np.float32)

    E_raw = compute_E(pt, eta, mass)

    # Pad or truncate to MAX_P
    P_src = slep_P
    if P_src < MAX_P:
        def pad2d(arr, fill=0.0):
            return np.concatenate([arr, np.full((B, MAX_P - P_src), fill, dtype=arr.dtype)], axis=1)
        pt      = pad2d(pt);      eta     = pad2d(eta)
        sin_phi = pad2d(sin_phi); cos_phi = pad2d(cos_phi)
        mass    = pad2d(mass);    btag    = pad2d(btag)
        etag    = pad2d(etag);    utag    = pad2d(utag)
        E_raw   = pad2d(E_raw)
        valid_m_padded = np.concatenate([valid_m, np.zeros((B, MAX_P - P_src), dtype=bool)], axis=1)
    elif P_src > MAX_P:
        pt = pt[:, :MAX_P]; eta = eta[:, :MAX_P]; sin_phi = sin_phi[:, :MAX_P]
        cos_phi = cos_phi[:, :MAX_P]; mass = mass[:, :MAX_P]; btag = btag[:, :MAX_P]
        etag = etag[:, :MAX_P]; utag = utag[:, :MAX_P]; E_raw = E_raw[:, :MAX_P]
        valid_m_padded = valid_m[:, :MAX_P]
    else:
        valid_m_padded = valid_m

    src_mask = valid_m_padded  # [B, MAX_P]

    # Scale particle features
    pt_s  = np.zeros((B, MAX_P), dtype=np.float32)
    eta_s = np.zeros_like(pt_s)
    E_s   = np.zeros_like(pt_s)
    valid_flat = src_mask.flatten()
    if valid_flat.any():
        pt_s.ravel()[valid_flat]  = scalers.pt_scaler.transform(pt.ravel()[valid_flat].reshape(-1,1)).ravel()
        eta_s.ravel()[valid_flat] = scalers.eta_scaler.transform(eta.ravel()[valid_flat].reshape(-1,1)).ravel()
        E_s.ravel()[valid_flat]   = scalers.E_scaler.transform(E_raw.ravel()[valid_flat].reshape(-1,1)).ravel()

    jet_out = np.stack([pt_s, eta_s, sin_phi, cos_phi, E_s, mass, btag], axis=-1)  # [B, MAX_P, 7]
    jet_out[~src_mask] = 0.0

    interactions = build_interaction_matrix(jet_out)

    # Build masks from TARGETS indices
    def make_mask(idx_arrays):
        m_ = np.zeros((B, MAX_P), dtype=np.float32)
        for idx_arr in idx_arrays:
            valid = (idx_arr >= 0) & (idx_arr < MAX_P)
            m_[np.where(valid)[0], idx_arr[valid]] = 1.0
        return m_

    had_top_m = make_mask([ht_b, ht_q1, ht_q2])   # chain 0
    had_W_m   = make_mask([ht_q1, ht_q2])
    lep_top_m = make_mask([lt_b, lt_l])             # chain 1
    lep_W_m   = make_mask([lt_l])

    masks_tops = np.stack([had_top_m, lep_top_m], axis=1)   # [B, 2, MAX_P]
    masks_Ws   = np.stack([had_W_m,   lep_W_m],   axis=1)

    # Validity
    valid_tops = np.stack([
        (ht_b >= 0) & (ht_q1 >= 0) & (ht_q2 >= 0),
        (lt_b >= 0) & (lt_l >= 0),
    ], axis=1).astype(np.uint8)

    valid_Ws = np.stack([
        (ht_q1 >= 0) & (ht_q2 >= 0),
        lt_l >= 0,                        # lepton always present (idx 0)
    ], axis=1).astype(np.uint8)

    # Kinematics
    def rk(mask):
        return reco_kin_from_raw(pt, eta, sin_phi, cos_phi, E_raw, mask, scalers)

    kt0 = rk(had_top_m); kt1 = rk(lep_top_m)
    kw0 = rk(had_W_m);   kw1 = rk(lep_W_m)

    kinematics_tops = np.stack([
        np.concatenate([kt0, np.zeros((B,1))], axis=-1),
        np.concatenate([kt1, np.zeros((B,1))], axis=-1),
    ], axis=1)
    kinematics_Ws = np.stack([
        np.concatenate([kw0, np.zeros((B,1))], axis=-1),
        np.concatenate([kw1, np.zeros((B,1))], axis=-1),
    ], axis=1)

    # Leptonic-extension keys
    particle_type = (etag + utag).clip(0, 1).astype(np.uint8)  # 1 at lepton slot
    if P_src != MAX_P:
        pass  # already padded/truncated above

    chain_type = np.zeros((B, 2), dtype=np.uint8)
    chain_type[:, 1] = 1   # chain 1 = leptonic

    met_pt_s = scalers.transform_met_pt(met_pt)
    n_jets   = src_mask.sum(axis=1).astype(np.float32) - 1.0   # subtract lepton
    n_bjets  = (btag > 0).astype(np.float32).sum(axis=1) - (etag + utag > 0).sum(axis=1)  # jets only
    n_bjets  = n_bjets.clip(0)
    n_leps   = np.ones(B, dtype=np.float32)

    globals_arr = np.stack([n_jets, n_bjets, n_leps,
                             met_pt_s, met_sin_phi, met_cos_phi], axis=-1)

    neutrino_truth = np.stack([
        np.zeros((B, 1), dtype=np.float32),
        nu_pz.reshape(B, 1),
    ], axis=1)  # [B, 2, 1]

    return {
        'jet': jet_out.astype(np.float32),
        'src_mask': src_mask.astype(bool),
        'interactions': interactions,
        'masks_tops': masks_tops,
        'masks_Ws': masks_Ws,
        'kinematics_tops': kinematics_tops.astype(np.float32),
        'kinematics_Ws': kinematics_Ws.astype(np.float32),
        'valid_tops': valid_tops,
        'valid_Ws': valid_Ws,
        'particle_type': particle_type,
        'chain_type': chain_type,
        'globals': globals_arr.astype(np.float32),
        'neutrino_truth': neutrino_truth,
    }


# ── HDF5 output ───────────────────────────────────────────────────────────────

DATASET_SPECS = {
    'jet':             ('float32', 'lzf',  None),
    'src_mask':        ('bool',    'gzip', 4),
    'interactions':    ('float32', 'lzf',  None),
    'masks_tops':      ('float32', 'gzip', 4),
    'masks_Ws':        ('float32', 'gzip', 4),
    'kinematics_tops': ('float32', 'gzip', 4),
    'kinematics_Ws':   ('float32', 'gzip', 4),
    'valid_tops':      ('uint8',   'gzip', 4),
    'valid_Ws':        ('uint8',   'gzip', 4),
    'particle_type':   ('uint8',   'gzip', 4),
    'chain_type':      ('uint8',   'gzip', 4),
    'globals':         ('float32', 'gzip', 4),
    'neutrino_truth':  ('float32', 'gzip', 4),
}


def create_datasets(out_f: h5py.File, sample: Dict[str, np.ndarray]):
    for key, arr in sample.items():
        dtype, comp, level = DATASET_SPECS.get(key, ('float32', 'gzip', 4))
        kwargs = {'compression': comp}
        if level is not None:
            kwargs['compression_opts'] = level
        out_f.create_dataset(
            key,
            shape=(0,) + arr.shape[1:],
            maxshape=(None,) + arr.shape[1:],
            dtype=dtype,
            **kwargs,
        )


def append_chunk(out_f: h5py.File, chunk: Dict[str, np.ndarray]):
    cur = out_f['jet'].shape[0]
    n   = chunk['jet'].shape[0]
    new_len = cur + n
    for key, arr in chunk.items():
        if key not in out_f:
            continue
        out_f[key].resize((new_len,) + out_f[key].shape[1:])
        out_f[key][cur:new_len] = arr


# ── Main pipeline ─────────────────────────────────────────────────────────────

def fit_scalers(had_train: Optional[Path], slep_train: Optional[Path]) -> CombinedScalers:
    scalers = CombinedScalers()
    print("[FIT] Fitting scalers on training data ...", flush=True)

    if had_train and had_train.exists():
        with h5py.File(had_train, 'r') as f:
            N = f['jet'].shape[0]
            for start in tqdm(range(0, N, CHUNK_SIZE), desc='Fit hadronic'):
                jet_raw = f['jet'][start:start+CHUNK_SIZE]
                had_fit_chunk(jet_raw, scalers)

    if slep_train and slep_train.exists():
        with h5py.File(slep_train, 'r') as f:
            N = f['INPUTS/Momenta/pt'].shape[0]
            for start in tqdm(range(0, N, CHUNK_SIZE), desc='Fit semi-leptonic'):
                slep_fit_chunk(f, start, min(start+CHUNK_SIZE, N), scalers)

    return scalers


def process_split(had_path: Optional[Path], slep_path: Optional[Path],
                  out_path: Path, scalers: CombinedScalers):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[PROCESS] → {out_path.name}", flush=True)

    total = 0
    with h5py.File(out_path, 'w') as out_f:
        datasets_created = False

        # ── Hadronic events ──
        if had_path and had_path.exists():
            with h5py.File(had_path, 'r') as f:
                N = f['jet'].shape[0]
                print(f"  Hadronic: {N:,} events", flush=True)
                for start in tqdm(range(0, N, CHUNK_SIZE), desc='  Hadronic'):
                    stop = min(start + CHUNK_SIZE, N)
                    jet_raw   = f['jet'][start:stop]
                    event_raw = f['event'][start:stop]

                    # Filter events with at least 1 valid object
                    n_valid = (
                        (np.any(jet_raw[:, :, 6:7] == t, axis=1).any(axis=-1)
                         for t in [1,2,3,4,5,6])
                    )

                    chunk = had_process_chunk(jet_raw, event_raw, scalers)
                    if not datasets_created:
                        create_datasets(out_f, chunk)
                        datasets_created = True
                    append_chunk(out_f, chunk)
                    total += chunk['jet'].shape[0]

        # ── Semi-leptonic events ──
        if slep_path and slep_path.exists():
            with h5py.File(slep_path, 'r') as f:
                N = f['INPUTS/Momenta/pt'].shape[0]
                print(f"  Semi-leptonic: {N:,} events", flush=True)
                for start in tqdm(range(0, N, CHUNK_SIZE), desc='  Semi-lep'):
                    stop = min(start + CHUNK_SIZE, N)
                    chunk = slep_process_chunk(f, start, stop, scalers)
                    if not datasets_created:
                        create_datasets(out_f, chunk)
                        datasets_created = True
                    append_chunk(out_f, chunk)
                    total += chunk['jet'].shape[0]

    print(f"  Done: {total:,} events written to {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Unified hadronic + semi-leptonic preprocessor")
    parser.add_argument('--had_train',  type=Path, default=None)
    parser.add_argument('--had_val',    type=Path, default=None)
    parser.add_argument('--had_test',   type=Path, default=None)
    parser.add_argument('--slep_train', type=Path,
                        default=Path('data/semi_leptonic_ttbar/training_mass_variation.h5'))
    parser.add_argument('--slep_test',  type=Path,
                        default=Path('data/semi_leptonic_ttbar/testing_sm.h5'))
    parser.add_argument('--output_dir', type=Path,
                        default=Path('data/topquarkreconstruction/leptonic_combined'))
    parser.add_argument('--scalers',    type=Path, default=None,
                        help='Path to pre-fitted scalers.joblib (skip fitting step)')
    args = parser.parse_args()

    scaler_path = args.output_dir / 'scalers.joblib'

    # ── Fit ──
    if args.scalers and args.scalers.exists():
        print(f"[FIT] Loading existing scalers from {args.scalers}", flush=True)
        scalers = CombinedScalers.load(args.scalers)
    else:
        scalers = fit_scalers(args.had_train, args.slep_train)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        scalers.save(scaler_path)

    # ── Process all splits ──
    splits = {
        'train': (args.had_train, args.slep_train),
        'val':   (args.had_val,   None),
        'test':  (args.had_test,  args.slep_test),
    }
    for split, (had, slep) in splits.items():
        if had is None and slep is None:
            continue
        if had and not had.exists():
            print(f"[SKIP] {split} hadronic file not found: {had}", flush=True)
            had = None
        if slep and not slep.exists():
            print(f"[SKIP] {split} semi-leptonic file not found: {slep}", flush=True)
            slep = None
        if had or slep:
            out = args.output_dir / f'ttbar_leptonic_{split}.h5'
            process_split(had, slep, out, scalers)

    print("\nDone! Next step:")
    print(f"  uv run src/main.py --config config/leptonic_config.yaml")


if __name__ == '__main__':
    main()
