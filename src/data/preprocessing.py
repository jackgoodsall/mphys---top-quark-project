import sys
import os

print("[START] Script initializing...", flush=True)

try:
    import numpy as np
    print("[OK] numpy imported", flush=True)
except Exception as e:
    print(f"[FAIL] numpy import: {e}", flush=True)
    sys.exit(1)

try:
    import h5py
    print("[OK] h5py imported", flush=True)
except Exception as e:
    print(f"[FAIL] h5py import: {e}", flush=True)
    sys.exit(1)

try:
    import joblib
    print("[OK] joblib imported", flush=True)
except Exception as e:
    print(f"[FAIL] joblib import: {e}", flush=True)
    sys.exit(1)

try:
    from pathlib import Path
    print("[OK] pathlib imported", flush=True)
except Exception as e:
    print(f"[FAIL] pathlib import: {e}", flush=True)
    sys.exit(1)

try:
    from abc import ABC, abstractmethod
    from typing import Tuple, Optional, Dict, Any
    print("[OK] abc/typing imported", flush=True)
except Exception as e:
    print(f"[FAIL] abc/typing import: {e}", flush=True)
    sys.exit(1)

try:
    import vector
    print("[OK] vector imported", flush=True)
except Exception as e:
    print(f"[FAIL] vector import: {e}", flush=True)

try:
    from tqdm import tqdm
    print("[OK] tqdm imported", flush=True)
except Exception as e:
    print(f"[FAIL] tqdm import: {e}", flush=True)
    sys.exit(1)

print("\n[ATTEMPTING] Custom imports from src/utils...", flush=True)

try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from src.data_utils.scalers import LogMinMaxScaler, StandardScaler, PhiTransformer
    print("[OK] Scalers imported", flush=True)
except Exception as e:
    print(f"[FAIL] Scalers import: {e}", flush=True)
    print("[WARN] Proceeding with dummy scalers", flush=True)
    
    class DummyScaler:
        def partial_fit(self, X):
            pass
        def transform(self, X):
            return X
    
    LogMinMaxScaler = DummyScaler
    StandardScaler = DummyScaler
    PhiTransformer = DummyScaler

try:
    from src.utils.utils import load_any_config
    print("[OK] load_any_config imported", flush=True)
except Exception as e:
    print(f"[FAIL] load_any_config import: {e}", flush=True)
    print("[WARN] Proceeding without config loader", flush=True)
    
    def load_any_config(path):
        return {}

try:
    from kinematics import (
        apply_mask,
        calculate_energy_value,
        convert_polar_to_cartesian,
        create_interaction_matrix,
        px_py_pz_from_pt_eta_phi,
    )
    print("[OK] kinematics functions imported", flush=True)
except Exception as e:
    print(f"[FAIL] kinematics import: {e}", flush=True)
    print("[WARN] Proceeding with dummy utils", flush=True)

    def apply_mask(arrays, mask):
        return tuple(a[mask] for a in arrays)

    def calculate_energy_value(x):
        return x[..., 3]

    def convert_polar_to_cartesian(x):
        return x[..., :4]

    def create_interaction_matrix(jet_chunk):
        B, P, F = jet_chunk.shape
        return np.zeros((B, P, P, 1))

    def px_py_pz_from_pt_eta_phi(X):
        pt, eta, phi = X[..., 0], X[..., 1], X[..., 2]
        return pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)

print("\n[SUCCESS] All imports completed\n", flush=True)


class TargetProcessor(ABC):
    """Abstract base for different target processing strategies."""

    @abstractmethod
    def get_target_count(self) -> int:
        pass

    @abstractmethod
    def init_target_transformers(self) -> tuple:
        pass

    @abstractmethod
    def process_targets(self, targets_chunk: np.ndarray, is_temp: bool) -> Dict[str, np.ndarray]:
        pass

    @abstractmethod
    def get_save_keys(self) -> list:
        pass

    @abstractmethod
    def reshape_targets(self, targets_chunk: np.ndarray) -> np.ndarray:
        pass


class IndividualParticleMaskAndKinematicsProcessor(TargetProcessor):
    """Processes masks and kinematics for individual tops and W bosons."""

    def __init__(self, include_leptonic_keys: bool = False):
        self.top_transformers = None
        self.W_transformers = None
        # When True: also save the four leptonic-extension keys with default values
        # (zeros/all-hadronic). Needed so hadronic files are compatible with the
        # combined leptonic training pipeline.
        self.include_leptonic_keys = include_leptonic_keys

    def get_target_count(self) -> int:
        return 2  # 2 tops and 2 Ws

    def init_target_transformers(self) -> tuple:
        """Initialize transformers for both tops and Ws kinematics."""
        # Transformers for top kinematics (pt, eta, phi, energy)
        top_trans = (
            LogMinMaxScaler(),      # pt
            StandardScaler(),       # eta
            PhiTransformer(),       # phi -> cos(phi), sin(phi)
            LogMinMaxScaler(),      # energy
        )

        # Transformers for W kinematics (same structure)
        W_trans = (
            LogMinMaxScaler(),      # pt
            StandardScaler(),       # eta
            PhiTransformer(),       # phi -> cos(phi), sin(phi)
            LogMinMaxScaler(),      # energy
        )

        self.top_transformers = top_trans
        self.W_transformers = W_trans

        return (top_trans, W_trans)

    def process_targets(self, targets_dict: Dict[str, np.ndarray], is_temp: bool) -> Dict[str, np.ndarray]:
        """Masks don't need processing, kinematics handled in _transform_targets."""
        return targets_dict

    def reshape_targets(self, targets_chunk: np.ndarray) -> np.ndarray:
        """Not used for this processor."""
        return targets_chunk

    def get_save_keys(self) -> list:
        keys = ["masks_tops", "masks_Ws", "kinematics_tops", "kinematics_Ws",
                "valid_tops", "valid_Ws"]
        if self.include_leptonic_keys:
            keys += ["particle_type", "chain_type", "globals", "neutrino_truth"]
        return keys


class InteractionProcessor(ABC):
    """Abstract base for interaction matrix handling."""

    @abstractmethod
    def needs_interaction(self) -> bool:
        pass

    @abstractmethod
    def init_interaction_transformer(self):
        pass


class NoInteractionProcessor(InteractionProcessor):
    def needs_interaction(self) -> bool:
        return False

    def init_interaction_transformer(self):
        return None


class WithInteractionProcessor(InteractionProcessor):
    def __init__(self, scaling: str = "logminmax"):
        # "logminmax" (default, legacy) = single shared LogMinMaxScaler over all 4
        # interaction features; "per_feature" = PerFeatureScaler with a physically
        # appropriate transform per column (ΔR/z standardised, kT/m² log-min-max).
        self.scaling = scaling

    def needs_interaction(self) -> bool:
        return True

    def init_interaction_transformer(self):
        if self.scaling == "per_feature":
            from src.data_utils.scalers import PerFeatureScaler
            return PerFeatureScaler()
        return LogMinMaxScaler()


class TargetExtractor(ABC):
    """Abstract base for different target extraction strategies."""

    @abstractmethod
    def extract_targets(
        self, jet_chunk: np.ndarray, targets_chunk: np.ndarray
    ) -> Dict[str, np.ndarray]:
        pass


class IndividualParticleMaskAndKinematicsExtractor(TargetExtractor):
    """
    Extracts binary masks (B, 2, P) and reconstructed kinematics (B, 2, 5) 
    for both tops and W bosons using fully vectorized operations.
    """

    def __init__(self,
                 tag_top1: np.ndarray = None,
                 tag_top2: np.ndarray = None,
                 tag_W1: np.ndarray = None,
                 tag_W2: np.ndarray = None,
                 num_jets: int = 20,
                 require_top_for_w: bool = False,
                 include_leptonic_keys: bool = False):
        # Set default truth-matching tags
        self.tag_top1 = tag_top1 if tag_top1 is not None else np.array([1, 2, 3])
        self.tag_top2 = tag_top2 if tag_top2 is not None else np.array([4, 5, 6])
        self.tag_W1 = tag_W1 if tag_W1 is not None else np.array([2, 3])
        self.tag_W2 = tag_W2 if tag_W2 is not None else np.array([5, 6])
        self.num_jets = num_jets
        self.require_top_for_w = require_top_for_w
        self.include_leptonic_keys = include_leptonic_keys

        # Define reconstruction tasks: (tags, particle_type, index)
        self.reco_tasks = [
            (self.tag_top1, "tops", 0),
            (self.tag_top2, "tops", 1),
            (self.tag_W1, "Ws", 0),
            (self.tag_W2, "Ws", 1),
        ]

    def extract_targets(
        self, jet_chunk: np.ndarray, targets_chunk: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Extract binary masks and reconstructed kinematics for tops and Ws.
        
        Args:
            jet_chunk (np.ndarray): Input jet array (B, P, F) where F includes 
                                    pt, eta, phi, energy, and truthmatch tag (index 6).
            targets_chunk (np.ndarray): Unused for reconstruction.

        Returns:
            Dict[str, np.ndarray]: Dictionary containing:
                - "masks_tops": (B, 2, P) binary masks for [top1, top2]
                - "masks_Ws": (B, 2, P) binary masks for [W1, W2]
                - "kinematics_tops": (B, 2, 5) kinematics for [top1, top2]
                - "kinematics_Ws": (B, 2, 5) kinematics for [W1, W2]
        """
        B, P, F = jet_chunk.shape

        # --- 1. Extract Binary Masks ---

        jet_tags = jet_chunk[..., 6]  # Shape (B, P)

        # Create masks for each particle (B, P) -> (B, 2, P)
        top1_mask = np.isin(jet_tags, self.tag_top1).astype(np.float32)
        top2_mask = np.isin(jet_tags, self.tag_top2).astype(np.float32)
        W1_mask = np.isin(jet_tags, self.tag_W1).astype(np.float32)
        W2_mask = np.isin(jet_tags, self.tag_W2).astype(np.float32)

        # Stack masks: (B, 2, P)
        masks_tops = np.stack([top1_mask, top2_mask], axis=1)  # (B, 2, P)
        masks_Ws = np.stack([W1_mask, W2_mask], axis=1)        # (B, 2, P)

        # --- 2. Reconstruct Kinematics (Vectorized) ---

        # Prepare flattened inputs
        flat_jets_vec = vector.zip({
            "pt": jet_chunk[..., 0].flatten(),
            "eta": jet_chunk[..., 1].flatten(),
            "phi": jet_chunk[..., 2].flatten(),
            "energy": jet_chunk[..., 3].flatten(),
        })

        event_indices = np.repeat(np.arange(B), P)  # (B*P,)
        flat_tags = jet_tags.flatten()

        # Convert to Cartesian for accurate summation
        flat_px = flat_jets_vec.px.to_numpy()
        flat_py = flat_jets_vec.py.to_numpy()
        flat_pz = flat_jets_vec.pz.to_numpy()
        flat_E = flat_jets_vec.energy.to_numpy()

        # Initialize output arrays (B, 2, 4) for [pt, eta, phi, energy]
        reco_tops = np.zeros((B, 2, 4), dtype=np.float32)
        reco_Ws = np.zeros((B, 2, 4), dtype=np.float32)

        # Track which objects are reconstructable: [B, 2] bool
        valid_tops = np.zeros((B, 2), dtype=bool)
        valid_Ws = np.zeros((B, 2), dtype=bool)

        # Vectorized reconstruction for all 4 particles
        for tags, particle_type, idx in self.reco_tasks:
            # Select jets matched to current particle
            tag_mask = np.isin(flat_tags, tags)
            matched_indices = event_indices[tag_mask]

            # Track which events have ALL decay products matched.
            # Each individual tag must have >= 1 matching jet.
            # e.g. top1 tags=[1,2,3]: need at least one jet with tag 1,
            #      at least one with tag 2, AND at least one with tag 3.
            all_tags_present = np.ones(B, dtype=bool)
            for tag in tags:
                tag_specific_mask = (flat_tags == tag)
                tag_matched_events = event_indices[tag_specific_mask]
                tag_counts = np.bincount(tag_matched_events, minlength=B)
                all_tags_present &= (tag_counts >= 1)

            if particle_type == "tops":
                valid_tops[:, idx] = all_tags_present
            else:
                valid_Ws[:, idx] = all_tags_present

            if matched_indices.size == 0:
                continue

            # Grouped reduction: sum Cartesian components by event
            sum_px = np.bincount(matched_indices, weights=flat_px[tag_mask], minlength=B)
            sum_py = np.bincount(matched_indices, weights=flat_py[tag_mask], minlength=B)
            sum_pz = np.bincount(matched_indices, weights=flat_pz[tag_mask], minlength=B)
            sum_E = np.bincount(matched_indices, weights=flat_E[tag_mask], minlength=B)

            # Reconstruct 4-vector
            reco_vec = vector.zip({
                "px": sum_px,
                "py": sum_py,
                "pz": sum_pz,
                "E": sum_E
            })

            # Convert to polar coordinates
            reco_polar = np.stack([
                reco_vec.pt.to_numpy(),
                reco_vec.eta.to_numpy(),
                reco_vec.phi.to_numpy(),
                reco_vec.E.to_numpy(),
            ], axis=-1)  # (B, 4)

            # Store in appropriate array
            if particle_type == "tops":
                reco_tops[:, idx, :] = reco_polar
            else:  # "Ws"
                reco_Ws[:, idx, :] = reco_polar

        # --- 3. Add Placeholder Column (5th feature = 0) ---

        kinematics_tops = np.concatenate([
            reco_tops,
            np.zeros((B, 2, 1), dtype=np.float32)
        ], axis=-1)  # (B, 2, 5)

        kinematics_Ws = np.concatenate([
            reco_Ws,
            np.zeros((B, 2, 1), dtype=np.float32)
        ], axis=-1)  # (B, 2, 5)

        # Optionally require parent top to be valid for W to count
        if self.require_top_for_w:
            valid_Ws &= valid_tops

        result = {
            "masks_tops": masks_tops,
            "masks_Ws": masks_Ws,
            "kinematics_tops": kinematics_tops,
            "kinematics_Ws": kinematics_Ws,
            "valid_tops": valid_tops.astype(np.uint8),  # [B, 2]
            "valid_Ws": valid_Ws.astype(np.uint8),      # [B, 2]
        }

        # Leptonic-extension defaults for hadronic data:
        # particle_type = all zeros (all jets), chain_type = all zeros (all hadronic),
        # globals = [n_jets_from_src_mask, n_bjets, 0, 0, 0, 0] built in the pipeline,
        # neutrino_truth = zeros.
        if self.include_leptonic_keys:
            result["particle_type"] = np.zeros((B, P), dtype=np.uint8)          # [B, P]
            result["chain_type"]    = np.zeros((B, 2), dtype=np.uint8)           # [B, 2] all hadronic
            result["neutrino_truth"] = np.zeros((B, 2, 1), dtype=np.float32)    # [B, 2, 1]
            # globals placeholder — filled properly in _transform_file when event data is available
            result["globals"]       = np.zeros((B, 6), dtype=np.float32)         # [B, 6]

        return result


class LeptonicMaskAndKinematicsExtractor(TargetExtractor):
    """
    Extracts masks, kinematics and leptonic-extension keys for semi-leptonic ttbar.

    Expected raw HDF5 keys (from root_to_h5_leptonic.py):
        jet          : [B, N_jets+1, 9]  unified jets+lepton, feature order:
                       [pt, eta, phi, E, m, btag, charge, lep_type, truthtag]
        particle_type: [B, N_jets+1]     0=jet, 1=lepton
        MET          : [B, 2]            [MET_pt, MET_phi]
        neutrino_pz_truth: [B, 1]        truth neutrino pz

    Truthtag convention:
        1 = b from hadronic top
        2,3 = hadronic W decay jets
        4 = b from leptonic top
        7 = lepton (in unified array, last particle slot)
        0 = unmatched

    Produces all hadronic keys plus: particle_type, chain_type, globals, neutrino_truth.
    chain_type [B, 2]: [0, 1] = [hadronic chain, leptonic chain] (chain 0 = had, chain 1 = lep).
    globals [B, 6]: [n_jets, n_bjets, n_leptons=1, MET_pt, sin(MET_phi), cos(MET_phi)].
    neutrino_truth [B, 2, 1]: [[0], [nu_pz]] — chain 0 has no neutrino.
    """

    HAD_B_TAG   = 1
    HAD_W1_TAG  = 2
    HAD_W2_TAG  = 3
    LEP_B_TAG   = 4
    LEPTON_TAG  = 7

    def __init__(self, num_jets: int = 20):
        self.num_jets = num_jets
        # hadronic top = chain 0, leptonic top = chain 1 (fixed convention)
        self.hadronic_top_tags = np.array([self.HAD_B_TAG, self.HAD_W1_TAG, self.HAD_W2_TAG])
        self.hadronic_W_tags   = np.array([self.HAD_W1_TAG, self.HAD_W2_TAG])
        self.leptonic_b_tag    = self.LEP_B_TAG
        self.lepton_tag        = self.LEPTON_TAG

    def extract_targets(
        self,
        jet_chunk: np.ndarray,        # [B, N_total, 9] — unified jet+lepton
        extra_chunks: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Extract all targets for semi-leptonic ttbar.

        Args:
            jet_chunk:    [B, N_total, 9] unified jet+lepton feature array
            extra_chunks: must contain 'particle_type' [B, N], 'MET' [B, 2],
                          'neutrino_pz_truth' [B, 1]

        Returns:
            Dict with masks_tops, masks_Ws, kinematics_*, valid_*, particle_type,
            chain_type, globals, neutrino_truth.
        """
        if extra_chunks is None:
            extra_chunks = {}

        B, N_total, F = jet_chunk.shape
        truthtag = jet_chunk[..., -1].astype(np.int32)  # last feature = truthtag

        particle_type_raw = extra_chunks.get("particle_type",
                                             np.zeros((B, N_total), dtype=np.uint8))
        met = extra_chunks.get("MET", np.zeros((B, 2), dtype=np.float32))
        nu_pz_truth = extra_chunks.get("neutrino_pz_truth",
                                        np.zeros((B, 1), dtype=np.float32))

        met_pt  = met[:, 0]
        met_phi = met[:, 1]

        # ── Binary masks ──
        def make_mask(tags):
            return np.isin(truthtag, tags).astype(np.float32)  # [B, N]

        had_top_mask = make_mask(self.hadronic_top_tags)   # [B, N] b + 2 jets
        had_W_mask   = make_mask(self.hadronic_W_tags)     # [B, N] 2 jets only
        lep_top_mask = (make_mask([self.leptonic_b_tag])   # b-jet
                        + make_mask([self.lepton_tag]))     # + lepton
        lep_top_mask = lep_top_mask.clip(0, 1)
        lep_W_mask   = make_mask([self.lepton_tag])         # [B, N] lepton only

        # Shape: [B, 2, N] — chain 0 = hadronic, chain 1 = leptonic
        masks_tops = np.stack([had_top_mask, lep_top_mask], axis=1)  # [B, 2, N]
        masks_Ws   = np.stack([had_W_mask,   lep_W_mask],   axis=1)  # [B, 2, N]

        # ── Kinematics (vectorised 4-vector summation) ──
        try:
            flat_jets_vec = vector.zip({
                "pt":     jet_chunk[..., 0].flatten(),
                "eta":    jet_chunk[..., 1].flatten(),
                "phi":    jet_chunk[..., 2].flatten(),
                "energy": jet_chunk[..., 3].flatten(),
            })
            ev_idx  = np.repeat(np.arange(B), N_total)
            flat_px = flat_jets_vec.px.to_numpy()
            flat_py = flat_jets_vec.py.to_numpy()
            flat_pz = flat_jets_vec.pz.to_numpy()
            flat_E  = flat_jets_vec.energy.to_numpy()
        except Exception:
            # Fallback without vector library
            flat_E  = jet_chunk[..., 3].flatten()
            flat_px = (jet_chunk[..., 0] * np.cos(jet_chunk[..., 2])).flatten()
            flat_py = (jet_chunk[..., 0] * np.sin(jet_chunk[..., 2])).flatten()
            flat_pz = (jet_chunk[..., 0] * np.sinh(jet_chunk[..., 1])).flatten()
            ev_idx  = np.repeat(np.arange(B), N_total)

        flat_tags = truthtag.flatten()

        def reco_4vec(tags):
            """Sum 4-vectors of particles matched to `tags` → [B, 4] polar."""
            tag_mask = np.isin(flat_tags, tags)
            if not tag_mask.any():
                return np.zeros((B, 4), dtype=np.float32)
            ev = ev_idx[tag_mask]
            sum_px = np.bincount(ev, weights=flat_px[tag_mask], minlength=B)
            sum_py = np.bincount(ev, weights=flat_py[tag_mask], minlength=B)
            sum_pz = np.bincount(ev, weights=flat_pz[tag_mask], minlength=B)
            sum_E  = np.bincount(ev, weights=flat_E[tag_mask],  minlength=B)
            try:
                v = vector.zip({"px": sum_px, "py": sum_py, "pz": sum_pz, "E": sum_E})
                return np.stack([v.pt.to_numpy(), v.eta.to_numpy(),
                                 v.phi.to_numpy(), v.E.to_numpy()], axis=-1).astype(np.float32)
            except Exception:
                pt  = np.sqrt(sum_px**2 + sum_py**2)
                p   = np.sqrt(sum_px**2 + sum_py**2 + sum_pz**2)
                eta = np.arctanh(np.clip(sum_pz / np.maximum(p, 1e-9), -0.9999, 0.9999))
                phi = np.arctan2(sum_py, sum_px)
                return np.stack([pt, eta, phi, sum_E], axis=-1).astype(np.float32)

        had_top_kin = reco_4vec(self.hadronic_top_tags)   # [B, 4]
        had_W_kin   = reco_4vec(self.hadronic_W_tags)
        lep_top_kin = reco_4vec([self.leptonic_b_tag, self.lepton_tag])
        lep_W_kin   = reco_4vec([self.lepton_tag])

        def with_placeholder(kin):  # [B, 4] → [B, 5] (adds zero column)
            return np.concatenate([kin, np.zeros((B, 1), dtype=np.float32)], axis=-1)

        kinematics_tops = np.stack([with_placeholder(had_top_kin),
                                     with_placeholder(lep_top_kin)], axis=1)  # [B, 2, 5]
        kinematics_Ws   = np.stack([with_placeholder(had_W_kin),
                                     with_placeholder(lep_W_kin)],   axis=1)  # [B, 2, 5]

        # ── Per-chain validity ──
        def all_tags_present(tags_list):
            ok = np.ones(B, dtype=bool)
            for tag in tags_list:
                ok &= np.any(truthtag == tag, axis=1)
            return ok

        had_top_valid = all_tags_present([self.HAD_B_TAG, self.HAD_W1_TAG, self.HAD_W2_TAG])
        had_W_valid   = all_tags_present([self.HAD_W1_TAG, self.HAD_W2_TAG])
        lep_b_valid   = all_tags_present([self.LEP_B_TAG])
        lep_lep_valid = all_tags_present([self.LEPTON_TAG])
        lep_top_valid = lep_b_valid & lep_lep_valid
        lep_W_valid   = lep_lep_valid

        valid_tops = np.stack([had_top_valid, lep_top_valid], axis=1).astype(np.uint8)
        valid_Ws   = np.stack([had_W_valid,   lep_W_valid],   axis=1).astype(np.uint8)

        # ── Leptonic-extension keys ──
        # chain_type: chain 0 = hadronic (0), chain 1 = leptonic (1)
        chain_type = np.zeros((B, 2), dtype=np.uint8)
        chain_type[:, 1] = 1   # second chain is always leptonic

        # globals: [n_jets, n_bjets, n_leptons, MET_pt, sin(MET_phi), cos(MET_phi)]
        n_jets  = (particle_type_raw == 0).sum(axis=1).astype(np.float32)
        n_bjets = (jet_chunk[..., 5] > 0).astype(np.float32).sum(axis=1)  # btag feature
        n_leps  = (particle_type_raw == 1).sum(axis=1).astype(np.float32)
        globals_arr = np.stack([n_jets, n_bjets, n_leps,
                                 met_pt,
                                 np.sin(met_phi),
                                 np.cos(met_phi)], axis=-1).astype(np.float32)  # [B, 6]

        # neutrino_truth: [B, 2, 1] — chain 0 has no neutrino (zeros), chain 1 has pz
        neutrino_truth = np.stack([
            np.zeros((B, 1), dtype=np.float32),        # hadronic chain: no neutrino
            nu_pz_truth.reshape(B, 1),                  # leptonic chain: truth pz
        ], axis=1)  # [B, 2, 1]

        return {
            "masks_tops":      masks_tops,
            "masks_Ws":        masks_Ws,
            "kinematics_tops": kinematics_tops,
            "kinematics_Ws":   kinematics_Ws,
            "valid_tops":      valid_tops,
            "valid_Ws":        valid_Ws,
            "particle_type":   particle_type_raw.astype(np.uint8),
            "chain_type":      chain_type,
            "globals":         globals_arr,
            "neutrino_truth":  neutrino_truth,
        }


class LeptonicTargetProcessor(IndividualParticleMaskAndKinematicsProcessor):
    """
    Extends the hadronic processor with leptonic-extension keys.
    Used when preprocessing semi-leptonic HDF5 files.
    """

    def __init__(self):
        super().__init__(include_leptonic_keys=True)

    def get_save_keys(self) -> list:
        return ["masks_tops", "masks_Ws", "kinematics_tops", "kinematics_Ws",
                "valid_tops", "valid_Ws",
                "particle_type", "chain_type", "globals", "neutrino_truth"]


class TopReconstructionDatasetFromH5:
    """Dataset preprocessor with support for individual particle masks and kinematics."""

    def __init__(
        self,
        config: Dict[str, Any],
        target_processor: TargetProcessor,
        interaction_processor: InteractionProcessor = None,
        target_extractor: TargetExtractor = None,
    ):
        print("[INIT] TopReconstructionDatasetFromH5 starting...", flush=True)
        
        self.raw_file_config = config.get("root_dataset_prepper", {})
        self.preprocessing_config = config.get("preprocessing", {})
        self.target_processor = target_processor
        self.interaction_processor = interaction_processor or NoInteractionProcessor()
        self.target_extractor = target_extractor or IndividualParticleMaskAndKinematicsExtractor()

        self.raw_file_prefix_and_path = self._construct_path(
            self.raw_file_config.get("save_path", "./data"),
            self.raw_file_config.get("save_file_prefix", "raw_"),
        )
        self.save_dir = Path(self.preprocessing_config.get("save_path", "./processed"))
        self.save_file_prefix_and_path = self._construct_path(
            self.preprocessing_config.get("save_path", "./processed"),
            self.preprocessing_config.get("save_file_prefix", "processed_"),
        )
        self.stream_size = self.preprocessing_config.get("stream_size", 1000)
        # Minimum number of reconstructable objects to keep an event.
        # Default 4 = old behaviour (fully-reconstructable events only).
        # Set to 1 to include all events with at least one object.
        self.min_objects = self.preprocessing_config.get("min_objects", 4)

        # Extra raw-HDF5 keys to read and pass to extract_targets as extra_chunks.
        # For semi-leptonic data: ['particle_type', 'MET', 'neutrino_pz_truth']
        self.extra_read_keys: list = self.preprocessing_config.get("extra_read_keys", [])

        # Keys from the extractor output that are stored as-is (no transformer scaling).
        # These are categorical/integer or pre-computed arrays.
        PASS_THROUGH_KEYS = {"particle_type", "chain_type", "globals", "neutrino_truth"}
        self._pass_through_keys = PASS_THROUGH_KEYS

        print(f"[CONFIG] Raw path: {self.raw_file_prefix_and_path}", flush=True)
        print(f"[CONFIG] Save path: {self.save_file_prefix_and_path}", flush=True)
        print(f"[CONFIG] Stream size: {self.stream_size}", flush=True)

        self.jet_transformers = self._init_jet_transformers()
        self.target_transformers = self.target_processor.init_target_transformers()
        self.interaction_transformers = self.interaction_processor.init_interaction_transformer()

        self._prepare_datasets()
        
        print("[INIT] Complete!", flush=True)

    def _save_transformers(self):
        """Save fitted transformers to disk."""
        transform_save_path = self.save_dir / "target_transforms.joblib"
        
        print(f"[SAVE] Saving transformers to {transform_save_path}", flush=True)
        
        transformers_dict = {
            "jet_transformers": self.jet_transformers,
            "target_transformers": self.target_transformers,
            "interaction_transformers": self.interaction_transformers,
        }
        transform_save_path.parent.mkdir(parents = True, exist_ok= True)
        joblib.dump(transformers_dict, transform_save_path)
        print(f"[SAVE] Transformers saved successfully!", flush=True)

    def _construct_path(self, directory: str, prefix: str) -> Path:
        """Helper to construct paths consistently."""
        return Path(directory) / prefix

    def _init_jet_transformers(self) -> tuple:
        """Initialize jet transformers (columns: pt, eta, phi, E, mass)."""
        return (
            LogMinMaxScaler(),   # pt
            StandardScaler(),    # eta
            PhiTransformer(),    # phi -> sin, cos
            LogMinMaxScaler(),   # E
            LogMinMaxScaler(),   # mass (>= 0); previously passed through raw. n_input stays 7.
        )

    def _get_file_pattern(self, prefix_path: Path, suffix: str) -> str:
        """Helper method to construct glob pattern."""
        return f"{prefix_path}*{suffix}.h5"

    def _prepare_datasets(self):
        """Prepare datasets by fitting transformers on training data only."""
        print("\n[FIT] Starting transformer fitting...", flush=True)
        
        raw_file_pattern = self._get_file_pattern(self.raw_file_prefix_and_path, "")
        raw_files = sorted(Path().glob(raw_file_pattern))
        
        if not raw_files:
            print(f"[WARN] No raw files found matching: {raw_file_pattern}", flush=True)
            return

        # Separate train files from test/val files
        train_files = [f for f in raw_files if "train" in f.name.lower()]
        non_train_files = [f for f in raw_files if "train" not in f.name.lower()]
        
        if not train_files:
            print(f"[ERROR] No training files found! Cannot fit transformers.", flush=True)
            print(f"[ERROR] Looking for files with 'train' in filename.", flush=True)
            return
        
        print(f"[FIT] Found {len(train_files)} training files (will fit transformers)", flush=True)
        print(f"[FIT] Found {len(non_train_files)} test/val files (will only transform)", flush=True)
        
        # Only fit on training files
        for raw_file in train_files:
            print(f"[FIT] Fitting on {raw_file.name}...", flush=True)
            self._fit_file(raw_file)
        
        print("[FIT] Transformer fitting complete!", flush=True)
        self._save_transformers()
        print("[FIT] Fitted transformers will be applied to all files during transformation.", flush=True)
        self._transform_all()

    def _read_extra_chunks(self, f: "h5py.File", start: int, stop: int) -> Dict[str, np.ndarray]:
        """Read optional extra keys from an open raw HDF5 file."""
        extra = {}
        for key in self.extra_read_keys:
            if key in f:
                extra[key] = f[key][start:stop].copy()
        return extra

    def _fit_file(self, raw_path: Path):
        """Fit transformers on a single file."""
        with h5py.File(raw_path, "r") as f:
            file_len = f["jet"].shape[0]
            print(f"[FIT] File length: {file_len}", flush=True)

            for i in tqdm(
                range(0, file_len, self.stream_size),
                desc=f"Fit {os.path.basename(raw_path)}",
            ):
                jet_chunk = f["jet"][i : i + self.stream_size].copy()
                event_chunk = f["event"][i : i + self.stream_size].copy()
                extra_chunks = self._read_extra_chunks(f, i, i + self.stream_size)

                if jet_chunk.shape[0] == 0:
                    continue

                # Extract targets first (validity needed to compute the event filter)
                targets_dict = self.target_extractor.extract_targets(jet_chunk, extra_chunks if extra_chunks else None)

                # Compute per-event filter from validity arrays
                if "valid_tops" in targets_dict and "valid_Ws" in targets_dict:
                    n_valid = (targets_dict["valid_tops"].sum(axis=1)
                               + targets_dict["valid_Ws"].sum(axis=1))  # [B]
                    event_filter = n_valid >= self.min_objects
                else:
                    event_filter = event_chunk[:, 2] == 1

                jet_chunk = jet_chunk[event_filter]
                targets_dict = {k: v[event_filter] for k, v in targets_dict.items()}

                if jet_chunk.shape[0] == 0:
                    continue

                # Fit jet transformers
                self._fit_jet_transformers(jet_chunk)

                # Fit target transformers
                self._fit_target_transformers(targets_dict)

                # Fit interaction transformer on every chunk in sub-batches
                # to avoid OOM (full chunk would be 500k×20×20×4 ≈ 3 GB).
                if self.interaction_processor.needs_interaction():
                    try:
                        FIT_BATCH = 50_000
                        for b_start in range(0, jet_chunk.shape[0], FIT_BATCH):
                            b_end = min(b_start + FIT_BATCH, jet_chunk.shape[0])
                            int_batch = create_interaction_matrix(jet_chunk[b_start:b_end])
                            self._fit_interaction_transformers(int_batch)
                            del int_batch
                    except Exception as e:
                        print(f"[WARN] Interaction fit failed: {e}", flush=True)

    def _fit_jet_transformers(self, jet_chunk: np.ndarray):
        """Fit jet transformers on jet data."""
        N, P, F = jet_chunk.shape
        
        for i, transformer in enumerate(self.jet_transformers):
            var = jet_chunk[..., i]
            var_flat = var.reshape(-1, 1)
            transformer.partial_fit(var_flat)

    def _fit_target_transformers(self, targets_dict: Dict[str, np.ndarray]):
        """Fit target transformers on target kinematics data."""
        if isinstance(self.target_processor, IndividualParticleMaskAndKinematicsProcessor):
            # Fit top transformers (skip placeholder column at index 4)
            if "kinematics_tops" in targets_dict:
                tops_chunk = targets_dict["kinematics_tops"]
                N, M, F = tops_chunk.shape
                
                for i, transformer in enumerate(self.target_processor.top_transformers):
                    if i < F - 1:  # Skip placeholder
                        var = tops_chunk[..., i]
                        var_flat = var.reshape(-1, 1)
                        transformer.partial_fit(var_flat)
            
            # Fit W transformers (skip placeholder column at index 4)
            if "kinematics_Ws" in targets_dict:
                Ws_chunk = targets_dict["kinematics_Ws"]
                N, M, F = Ws_chunk.shape
                
                for i, transformer in enumerate(self.target_processor.W_transformers):
                    if i < F - 1:  # Skip placeholder
                        var = Ws_chunk[..., i]
                        var_flat = var.reshape(-1, 1)
                        transformer.partial_fit(var_flat)

    def _fit_interaction_transformers(self, interaction_chunk: np.ndarray):
        """Fit interaction transformers."""
        if interaction_chunk is not None:
            N, P, P2, F = interaction_chunk.shape
            interaction_flat = interaction_chunk.reshape(-1, F)
            self.interaction_transformers.partial_fit(interaction_flat)

    def _transform_all(self):
        """Transform all raw files and save processed versions."""
        print("\n[TRANSFORM] Starting transformation...", flush=True)
        
        raw_file_pattern = self._get_file_pattern(self.raw_file_prefix_and_path, "")
        raw_files = sorted(Path().glob(raw_file_pattern))

        self.save_dir.mkdir(parents=True, exist_ok=True)

        for raw_file in raw_files:
            save_file = self.save_dir / raw_file.name.replace(
                self.raw_file_config.get("save_file_prefix", "raw_"),
                self.preprocessing_config.get("save_file_prefix", "processed_"),
            )
            print(f"[TRANSFORM] {raw_file.name} -> {save_file.name}", flush=True)
            self._transform_file(raw_file, save_file)

    def _create_and_transform_interactions_batched(
        self, jet_chunk: np.ndarray, batch_size: int = 50_000
    ) -> np.ndarray:
        """Compute and transform the interaction matrix in batches.

        Processing the full chunk at once allocates N×P²×4×8 bytes (float64
        intermediates in LogMinMaxScaler), which can exceed available RAM for
        large N (e.g. 500 K events → ~25 GB).  Batching at ``batch_size``
        events keeps the peak allocation to batch_size×P²×4×8 bytes (~2.5 GB
        at the default of 50 K events with P=20).
        """
        N, P, _ = jet_chunk.shape
        n_int_feat = 4  # ΔR, kT, z, m²
        result = np.empty((N, P, P, n_int_feat), dtype=np.float32)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = jet_chunk[start:end]
            int_batch = create_interaction_matrix(batch)        # [b, P, P, 4]
            b, P_, P2_, F_ = int_batch.shape
            flat = int_batch.reshape(-1, F_)
            flat = self.interaction_transformers.transform(flat)
            result[start:end] = flat.reshape(b, P_, P2_, F_)
        return result

    def _transform_file(self, raw_path: Path, save_path: Path):
        """Transform a single file."""
        # Process interaction matrix in sub-chunks to avoid multi-GB allocations.
        INTERACTION_BATCH = 50_000

        with h5py.File(raw_path, "r") as read_f, h5py.File(save_path, "w") as write_f:
            file_len = read_f["jet"].shape[0]
            print(f"[TRANSFORM] Total events: {file_len}", flush=True)

            datasets_created = False

            for i in tqdm(
                range(0, file_len, self.stream_size),
                desc=f"Transform {os.path.basename(raw_path)}",
            ):
                jet_chunk = read_f["jet"][i : i + self.stream_size].copy()
                event_chunk = read_f["event"][i : i + self.stream_size].copy()
                extra_chunks = self._read_extra_chunks(read_f, i, i + self.stream_size)

                # Extract targets first (validity needed to compute the event filter)
                targets_dict = self.target_extractor.extract_targets(jet_chunk, extra_chunks if extra_chunks else None)

                # Fill globals from event array for hadronic data (particle_type=all-zeros path)
                # event layout for hadronic: [n_jets, n_bjets, all_matched]
                # For leptonic data with MET, globals are already set by the extractor.
                if "globals" in targets_dict and "MET" not in extra_chunks:
                    # Hadronic fallback: derive globals from event array
                    B = jet_chunk.shape[0]
                    n_jets  = event_chunk[:, 0].astype(np.float32)
                    n_bjets = event_chunk[:, 1].astype(np.float32) if event_chunk.shape[1] > 1 else np.zeros(B, dtype=np.float32)
                    targets_dict["globals"] = np.stack([
                        n_jets, n_bjets,
                        np.zeros(B, np.float32),  # n_leptons=0
                        np.zeros(B, np.float32),  # MET_pt=0
                        np.zeros(B, np.float32),  # sin(MET_phi)=0
                        np.ones( B, np.float32),  # cos(MET_phi)=1 (phi=0)
                    ], axis=-1)

                # Compute per-event filter from validity arrays
                if "valid_tops" in targets_dict and "valid_Ws" in targets_dict:
                    n_valid = (targets_dict["valid_tops"].sum(axis=1)
                               + targets_dict["valid_Ws"].sum(axis=1))  # [B]
                    event_filter = n_valid >= self.min_objects
                else:
                    event_filter = event_chunk[:, 2] == 1

                jet_chunk = jet_chunk[event_filter]
                event_chunk = event_chunk[event_filter]
                targets_dict = {k: v[event_filter] for k, v in targets_dict.items()}

                if jet_chunk.shape[0] == 0:
                    continue

                # Raw 4-vectors [E, px, py, pz] for the soft invariant-mass loss (D4).
                # Computed from the filtered RAW chunk before scaling; NaN padding -> 0.
                # Intentionally NOT augmented downstream (mass is rotation/flip invariant).
                _px, _py, _pz = px_py_pz_from_pt_eta_phi(jet_chunk[..., :3])  # (pt, eta, phi)
                jet_p4_raw = np.stack(
                    [jet_chunk[..., 3], _px, _py, _pz], axis=-1
                ).astype(np.float32)                                          # [B, P, 4]
                jet_p4_raw = np.nan_to_num(jet_p4_raw, nan=0.0)

                # Build interaction matrix from raw jets and transform in batches.
                # Must happen BEFORE jet transformation (interactions use raw kinematics).
                interaction_chunk = None
                if self.interaction_processor.needs_interaction():
                    try:
                        interaction_chunk = self._create_and_transform_interactions_batched(
                            jet_chunk, batch_size=INTERACTION_BATCH
                        )
                    except Exception as e:
                        print(f"[WARN] Interaction matrix creation failed: {e}", flush=True)

                # Transform jet features (interaction already handled above)
                jet_chunk, _ = self._transform_data(jet_chunk, None)
                
                # Transform target kinematics
                targets_dict = self._transform_targets(targets_dict)
                
                jet_chunk, src_mask, interaction_chunk = self._pad_and_src_mask(
                    jet_chunk, interaction_chunk
                )

                if not datasets_created:
                    self._create_datasets(
                        write_f,
                        jet_chunk.shape,
                        event_chunk.shape,
                        targets_dict,
                        interaction_chunk.shape if interaction_chunk is not None else None,
                        jet_p4_raw_shape=jet_p4_raw.shape,
                    )
                    datasets_created = True

                self._save_data_chunks(
                    write_f,
                    jet_chunk,
                    event_chunk,
                    src_mask,
                    targets_dict,
                    interaction_chunk,
                    jet_p4_raw=jet_p4_raw,
                )
        
        print(f"[TRANSFORM] Saved to {save_path}", flush=True)

    def _transform_data(
        self,
        jet: np.ndarray,
        interactions: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Transform jets and interactions."""
        N, P, F = jet.shape
        num_transformed_jet_features = len(self.jet_transformers)
        
        jet_transformed_list = []
        for i, transformer in enumerate(self.jet_transformers):
            var = jet[..., i]
            var_reshaped = var.reshape(-1, 1)
            transformed_var = transformer.transform(var_reshaped).reshape(N, P, -1)
            jet_transformed_list.append(transformed_var)
            
        jet_transformed_array = np.concatenate(jet_transformed_list, axis=-1)
        non_transformed_jets = jet[..., num_transformed_jet_features:]
        jet = np.concatenate((jet_transformed_array, non_transformed_jets), axis=-1)

        interactions_transformed = None
        if interactions is not None and self.interaction_processor.needs_interaction():
            N, P, P2, F = interactions.shape
            interactions_flat = interactions.reshape(-1, F)
            interactions_flat = self.interaction_transformers.transform(interactions_flat)
            interactions_transformed = interactions_flat.reshape(N, P, P2, F)
        
        return jet, interactions_transformed

    def _transform_targets(self, targets_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Transform target kinematics (masks stay unchanged)."""
        if isinstance(self.target_processor, IndividualParticleMaskAndKinematicsProcessor):
            # Transform top kinematics
            if "kinematics_tops" in targets_dict:
                tops_chunk = targets_dict["kinematics_tops"]
                N, M, F = tops_chunk.shape
                
                tops_transformed_list = []
                for i, transformer in enumerate(self.target_processor.top_transformers):
                    if i < F - 1:  # Skip placeholder
                        var = tops_chunk[..., i]
                        var_reshaped = var.reshape(-1, 1)
                        transformed_var = transformer.transform(var_reshaped).reshape(N, M, -1)
                        tops_transformed_list.append(transformed_var)
                
                # Concatenate transformed features and add placeholder back
                tops_transformed = np.concatenate(tops_transformed_list, axis=-1)
                placeholder = tops_chunk[..., -1:]
                targets_dict["kinematics_tops"] = np.concatenate([tops_transformed, placeholder], axis=-1)
            
            # Transform W kinematics
            if "kinematics_Ws" in targets_dict:
                Ws_chunk = targets_dict["kinematics_Ws"]
                N, M, F = Ws_chunk.shape
                
                Ws_transformed_list = []
                for i, transformer in enumerate(self.target_processor.W_transformers):
                    if i < F - 1:  # Skip placeholder
                        var = Ws_chunk[..., i]
                        var_reshaped = var.reshape(-1, 1)
                        transformed_var = transformer.transform(var_reshaped).reshape(N, M, -1)
                        Ws_transformed_list.append(transformed_var)
                
                # Concatenate transformed features and add placeholder back
                Ws_transformed = np.concatenate(Ws_transformed_list, axis=-1)
                placeholder = Ws_chunk[..., -1:]
                targets_dict["kinematics_Ws"] = np.concatenate([Ws_transformed, placeholder], axis=-1)
        
        return targets_dict

    def _create_datasets(
        self,
        file: h5py.File,
        jet_shape: Tuple,
        event_shape: Tuple,
        targets_dict: Dict[str, np.ndarray],
        interaction_shape: Optional[Tuple] = None,
        jet_p4_raw_shape: Optional[Tuple] = None,
    ):
        """Create HDF5 dataset groups."""
        _, N_jets, jet_features = jet_shape
        _, event_features = event_shape

        file.create_dataset(
            "jet", 
            shape=(0, N_jets, jet_features - 1), 
            maxshape=(None, N_jets, jet_features - 1), 
            compression="gzip", 
            compression_opts=4,
            dtype="float32",
        )
        file.create_dataset(
            "event", 
            shape=(0, event_features), 
            maxshape=(None, event_features), 
            compression="gzip", 
            compression_opts=4,
            dtype="float32",
        )
        file.create_dataset(
            "src_mask", 
            shape=(0, N_jets), 
            maxshape=(None, N_jets), 
            compression="gzip", 
            compression_opts=4,
            dtype="float32",
        )

        # Create datasets for masks, kinematics, and validity arrays
        _validity_keys = {"valid_tops", "valid_Ws"}
        for key in self.target_processor.get_save_keys():
            if key in targets_dict:
                target_array = targets_dict[key]

                if key in _validity_keys:
                    # Validity arrays are 2D [B, 2] uint8
                    _, n_cols = target_array.shape
                    file.create_dataset(
                        key,
                        shape=(0, n_cols),
                        maxshape=(None, n_cols),
                        compression="gzip",
                        compression_opts=4,
                        dtype="uint8",
                    )
                else:
                    # Masks and kinematics are 3D [B, M, F] float32
                    _, M_targets, target_features = target_array.shape
                    file.create_dataset(
                        key,
                        shape=(0, M_targets, target_features),
                        maxshape=(None, M_targets, target_features),
                        compression="gzip",
                        compression_opts=4,
                        dtype="float32",
                    )

        if interaction_shape is not None:
            _, N, N, interaction_features = interaction_shape
            # lzf: ~4x faster write and ~3x faster read than gzip-4, ~12% larger files.
            # h5py ships with lzf built-in so no extra install is needed.
            file.create_dataset(
                "interactions",
                shape=(0, N, N, interaction_features),
                maxshape=(None, N, N, interaction_features),
                compression="lzf",
                dtype="float32",
            )

        # Raw 4-vectors for the invariant-mass loss (D4). Explicit, not in get_save_keys.
        if jet_p4_raw_shape is not None:
            _, N_p4, F_p4 = jet_p4_raw_shape
            file.create_dataset(
                "jet_p4_raw",
                shape=(0, N_p4, F_p4),
                maxshape=(None, N_p4, F_p4),
                compression="lzf",
                dtype="float32",
            )

        # Leptonic-extension pass-through keys
        for key in self._pass_through_keys:
            if key in targets_dict:
                arr = targets_dict[key]
                if arr.ndim == 1:
                    shape_rest = ()
                else:
                    shape_rest = arr.shape[1:]
                dtype = "uint8" if key in {"particle_type", "chain_type"} else "float32"
                file.create_dataset(
                    key,
                    shape=(0,) + shape_rest,
                    maxshape=(None,) + shape_rest,
                    compression="gzip",
                    compression_opts=4,
                    dtype=dtype,
                )

        print(f"[TRANSFORM] Datasets created in HDF5 file", flush=True)

    def _save_data_chunks(
        self,
        file: h5py.File,
        jet_chunk: np.ndarray,
        event_chunk: np.ndarray,
        src_mask_chunk: np.ndarray,
        targets_dict: Dict[str, np.ndarray],
        interaction_chunk: Optional[np.ndarray] = None,
        jet_p4_raw: Optional[np.ndarray] = None,
    ):
        """Save data chunks to HDF5."""
        cur_len = file["jet"].shape[0]
        n0, n1 = cur_len, cur_len + jet_chunk.shape[0]

        file["jet"].resize((n1,) + file["jet"].shape[1:])
        file["event"].resize((n1,) + file["event"].shape[1:])
        file["src_mask"].resize((n1,) + file["src_mask"].shape[1:])

        file["jet"][n0:n1] = jet_chunk[..., :-1].astype("float32")
        file["event"][n0:n1] = event_chunk.astype("float32")
        file["src_mask"][n0:n1] = src_mask_chunk.astype("float32")

        # Save all target types (masks, kinematics, and validity arrays)
        _validity_keys = {"valid_tops", "valid_Ws"}
        for key in self.target_processor.get_save_keys():
            if key in targets_dict:
                file[key].resize((n1,) + file[key].shape[1:])
                dtype = "uint8" if key in _validity_keys else "float32"
                file[key][n0:n1] = targets_dict[key].astype(dtype)

        if interaction_chunk is not None:
            file["interactions"].resize((n1,) + file["interactions"].shape[1:])
            file["interactions"][n0:n1] = interaction_chunk.astype("float32")

        if jet_p4_raw is not None and "jet_p4_raw" in file:
            file["jet_p4_raw"].resize((n1,) + file["jet_p4_raw"].shape[1:])
            file["jet_p4_raw"][n0:n1] = jet_p4_raw.astype("float32")

        # Leptonic-extension pass-through keys (no transformer scaling)
        for key in self._pass_through_keys:
            if key in targets_dict and key in file:
                file[key].resize((n1,) + file[key].shape[1:])
                dtype = "uint8" if key in {"particle_type", "chain_type"} else "float32"
                file[key][n0:n1] = targets_dict[key].astype(dtype)

    def _pad_and_src_mask(
        self, 
        jet_chunk: np.ndarray, 
        interaction_chunk: Optional[np.ndarray] = None, 
        pad_value: float = 0
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Pad NaN values and create source mask."""
        src_mask = ~np.any(
            np.isnan(jet_chunk[..., :]),
            axis=-1,
        )
        jet_chunk = np.nan_to_num(jet_chunk, nan=pad_value)

        if interaction_chunk is not None:
            interaction_chunk = np.nan_to_num(interaction_chunk, nan=pad_value)

        return jet_chunk, src_mask, interaction_chunk


if __name__ == "__main__":
    print("\n" + "="*60, flush=True)
    print("INDIVIDUAL PARTICLE MASKS + KINEMATICS PROCESSOR", flush=True)
    print("="*60 + "\n", flush=True)
    
    try:
        import argparse
        _p = argparse.ArgumentParser()
        _p.add_argument("--config", default="config/preprocessing_config.yaml")
        _args, _ = _p.parse_known_args()
        config = load_any_config(_args.config)
        _interaction_scaling = config.get("preprocessing", {}).get("interaction_scaling", "logminmax")

        if not config:
            print("[WARN] Config is empty, using defaults", flush=True)
            config = {
                "root_dataset_prepper": {
                    "save_path": "./data",
                    "save_file_prefix": "raw_",
                },
                "preprocessing": {
                    "save_path": "./processed",
                    "save_file_prefix": "processed_",
                    "stream_size": 1000,
                }
            }

        # Use IndividualParticleMaskAndKinematicsProcessor and Extractor
        processor = IndividualParticleMaskAndKinematicsProcessor()
        extractor = IndividualParticleMaskAndKinematicsExtractor(
            tag_top1=np.array([1, 2, 3]),
            tag_top2=np.array([4, 5, 6]),
            tag_W1=np.array([2, 3]),
            tag_W2=np.array([5, 6]),
            num_jets=20
        )

        dataset = TopReconstructionDatasetFromH5(
            config,
            target_processor=processor,
            interaction_processor=WithInteractionProcessor(scaling=_interaction_scaling),
            target_extractor=extractor,
        )
        
        print("\n" + "="*60, flush=True)
        print("SUCCESS!", flush=True)
        print("="*60, flush=True)
        
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}", flush=True)
        import traceback
        traceback.print_exc()