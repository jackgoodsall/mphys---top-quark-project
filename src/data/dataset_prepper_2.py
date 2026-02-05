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
    from src.data_utls.scalers import LogMinMaxScaler, StandardScaler, PhiTransformer
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
    from utils import (
        apply_mask,
        calculate_energy_value,
        convert_polar_to_cartesian,
        create_interaction_matrix,
    )
    print("[OK] utils functions imported", flush=True)
except Exception as e:
    print(f"[FAIL] utils import: {e}", flush=True)
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


class BinaryMaskTargetProcessor(TargetProcessor):
    """Processes targets as a binary mask (B, 20, 1) for classification."""

    def get_target_count(self) -> int:
        return 20 

    def init_target_transformers(self) -> tuple:
        return ()

    def process_targets(self, targets_chunk: np.ndarray, is_temp: bool) -> Dict[str, np.ndarray]:
        return {"targets": targets_chunk}

    def reshape_targets(self, targets_chunk: np.ndarray) -> np.ndarray:
        if targets_chunk.ndim == 2:
            return targets_chunk.reshape(-1, 20, 1)
        return targets_chunk

    def get_save_keys(self) -> list:
        return ["targets"]


class CombinedMaskAndKinematicsProcessor(TargetProcessor):
    """Processes both binary mask and di-top kinematics."""
    
    def __init__(self):
        self.ditop_transformers = None
    
    def get_target_count(self) -> int:
        return 1  # For di-top system (mask doesn't need count)
    
    def init_target_transformers(self) -> tuple:
        """Initialize transformers for di-top kinematics (pt, eta, phi, energy)."""
        ditop_trans = (
            LogMinMaxScaler(),      # pt
            StandardScaler(),       # eta
            PhiTransformer(),       # phi -> cos(phi), sin(phi)
            LogMinMaxScaler(),      # energy
        )
        self.ditop_transformers = ditop_trans
        return (ditop_trans,)  # Return as tuple of tuples for consistency
    
    def process_targets(self, targets_dict: Dict[str, np.ndarray], is_temp: bool) -> Dict[str, np.ndarray]:
        """Apply target-specific processing."""
        # Binary mask needs no processing
        # Di-top kinematics is processed in _transform_targets
        return targets_dict
    
    def reshape_targets(self, targets_chunk: np.ndarray) -> np.ndarray:
        """Not used for this combined processor."""
        return targets_chunk
    
    def get_save_keys(self) -> list:
        return ["targets", "target_kinematics"]


class DiTopKinematicsProcessor(TargetProcessor):
    """Processes combined di-top system kinematics (B, 1, 5)."""
    
    def __init__(self):
        self.ditop_transformers = None
    
    def get_target_count(self) -> int:
        return 1  # Single combined di-top system
    
    def init_target_transformers(self) -> tuple:
        """Initialize transformers for di-top kinematics (pt, eta, phi, energy)."""
        ditop_trans = (
            LogMinMaxScaler(),
            StandardScaler(),
            PhiTransformer(),
            LogMinMaxScaler(),
        )
        self.ditop_transformers = ditop_trans
        return (ditop_trans,)  # Return as tuple of tuples for consistency
    
    def process_targets(self, targets_dict: Dict[str, np.ndarray], is_temp: bool) -> Dict[str, np.ndarray]:
        """Apply target-specific processing for di-top system."""
        ditop_chunk = targets_dict["target_kinematics"]
        targets_dict["target_kinematics"] = ditop_chunk
        return targets_dict
    
    def reshape_targets(self, targets_chunk: np.ndarray) -> np.ndarray:
        """Reshape targets to (B, 1, 4) format."""
        if targets_chunk.ndim == 2:
            return targets_chunk.reshape(-1, 1, 4)
        return targets_chunk
    
    def get_save_keys(self) -> list:
        return ["target_kinematics"]


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
    def needs_interaction(self) -> bool:
        return True

    def init_interaction_transformer(self):
        return LogMinMaxScaler()


class TargetExtractor(ABC):
    """Abstract base for different target extraction strategies."""

    @abstractmethod
    def extract_targets(
        self, jet_chunk: np.ndarray, targets_chunk: np.ndarray
    ) -> Dict[str, np.ndarray]:
        pass


class BinaryMaskTargetExtractor(TargetExtractor):
    """Creates a binary mask (B, 20, 1) from jet_truthmatch."""

    def __init__(self, match_labels: np.ndarray = None):
        self.match_labels = match_labels if match_labels is not None else np.array([1, 2, 3, 4, 5, 6])

    def extract_targets(
        self, jet_chunk: np.ndarray, targets_chunk: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Creates a binary mask where 1 means the jet is a truth-matched constituent."""
        B, P, F = jet_chunk.shape
        jet_tags = jet_chunk[..., 6]

        target_mask_bool = np.isin(jet_tags, self.match_labels)
        targets_np = target_mask_bool.astype(np.float32)
        binary_mask = targets_np[..., np.newaxis] 

        return {"targets": binary_mask}


class ReconstructedTargetExtractor(TargetExtractor):
    """
    Reconstructs targets (Tops and W bosons) from jets using a fully 
    VECTORIZED grouped reduction approach (no Python loop over events).
    """

    def __init__(self, tag_top1: np.ndarray = None, tag_top2: np.ndarray = None,
                 tag_W1: np.ndarray = None, tag_W2: np.ndarray = None):
        # Set default truth-matching tags
        self.tag_top1 = tag_top1 if tag_top1 is not None else np.array([1, 2, 3])
        self.tag_top2 = tag_top2 if tag_top2 is not None else np.array([4, 5, 6])
        self.tag_W1 = tag_W1 if tag_W1 is not None else np.array([2, 3])
        self.tag_W2 = tag_W2 if tag_W2 is not None else np.array([5, 6])
        
        # Define tasks as a list of (tags, array_name, index) for simplified loop placement
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
        Reconstruct top quarks and W bosons by summing truth-matched jets 
        using fully vectorized NumPy Grouped Array Reduction (np.bincount).
        
        Args:
            jet_chunk (np.ndarray): Input jet array (B, P, F) where F includes 
                                    pt, eta, phi, mass, and truthmatch tag (index 6).
            targets_chunk (np.ndarray): Input truth targets (unused for reconstruction, 
                                        but included for method signature consistency).

        Returns:
            Dict[str, np.ndarray]: Dictionary containing {"targets": reco_tops_full, 
                                                         "W_targets": reco_Ws_full}.
        """
        B, P, F = jet_chunk.shape
        
        # --- 1. Prepare Flattened Inputs (B*P jets) ---
        
        jet_tags = jet_chunk[..., 6] # Tags (float/int)
        
        # Create a single vector array for ALL jets (B*P elements)
        flat_jets_vec = vector.zip({
            "pt": jet_chunk[..., 0].flatten(),
            "eta": jet_chunk[..., 1].flatten(),
            "phi": jet_chunk[..., 2].flatten(),
            "energy": jet_chunk[..., 3].flatten(),
        })
        
        # Create a group index for every jet (The event ID, 0 to B-1)
        event_indices = np.repeat(np.arange(B), P) # Shape (B*P,)
        flat_tags = jet_tags.flatten()
        
        # Convert to Cartesian (Px, Py, Pz, E) for accurate linear summation
        flat_px = flat_jets_vec.px.to_numpy()
        flat_py = flat_jets_vec.py.to_numpy()
        flat_pz = flat_jets_vec.pz.to_numpy()
        flat_E = flat_jets_vec.energy.to_numpy()
        
        # Initialize the output arrays (B events, 2 particles, 4 features: pt, eta, phi, mass)
        reco_tops = np.zeros((B, 2, 4), dtype=np.float32)
        reco_Ws = np.zeros((B, 2, 4), dtype=np.float32)
        
        # --- 2. Vectorized Reconstruction Loop (Over 4 Target Types) ---
        
        for tags, target_type, idx in self.reco_tasks:
            
            # 2a. Mask: Select jets matched to the current target 
            tag_mask = np.isin(flat_tags, tags)
            
            # 2b. Filter: Select only the indices for matched jets
            matched_indices = event_indices[tag_mask]
            
            if matched_indices.size == 0:
                continue
            
            # 2c. Grouped Reduction (np.bincount)
            # Sum Cartesian components (Px, Py, Pz, E) grouped by event ID
            sum_px = np.bincount(matched_indices, weights=flat_px[tag_mask], minlength=B)
            sum_py = np.bincount(matched_indices, weights=flat_py[tag_mask], minlength=B)
            sum_pz = np.bincount(matched_indices, weights=flat_pz[tag_mask], minlength=B)
            sum_E = np.bincount(matched_indices, weights=flat_E[tag_mask], minlength=B)
            
            # 2d. Reconstruct the resulting 4-vector from the summed components
            reco_cartesian = vector.zip({
                "px": sum_px, 
                "py": sum_py, 
                "pz": sum_pz, 
                "E": sum_E
            })
            
            # 2e. Convert back to Polar coordinates (pt, eta, phi, mass)
            reco_pteta_mass = np.stack([
                reco_cartesian.pt.to_numpy(),
                reco_cartesian.eta.to_numpy(),
                reco_cartesian.phi.to_numpy(),
                reco_cartesian.E.to_numpy(),
            ], axis=-1) # Shape (B, 4)
            
            # 2f. Placement: Store the result in the correct output array and index
            if target_type == "tops":
                reco_tops[:, idx, :] = reco_pteta_mass
            else: # target_type == "Ws"
                reco_Ws[:, idx, :] = reco_pteta_mass

        # --- 3. Final Output Formatting ---
        
        # Append the 5th column (placeholder ID/PDG=0) for consistency with the rest of the pipeline
        reco_tops_full = np.concatenate([reco_tops, np.full((B, 2, 1), 0, dtype=np.float32)], axis=-1)
        reco_Ws_full = np.concatenate([reco_Ws, np.full((B, 2, 1), 0, dtype=np.float32)], axis=-1)
        
        return {"targets": reco_tops_full, "W_targets": reco_Ws_full}


class CombinedMaskAndKinematicsExtractor(TargetExtractor):
    """
    Extracts both binary mask and combined di-top system kinematics.
    """

    def __init__(self, match_labels: np.ndarray = None, 
                 tag_top1: np.ndarray = None, tag_top2: np.ndarray = None):
        # Binary mask labels
        self.match_labels = match_labels if match_labels is not None else np.array([1, 2, 3, 4, 5, 6])
        
        # Di-top kinematics labels
        self.tag_top1 = tag_top1 if tag_top1 is not None else np.array([1, 2, 3])
        self.tag_top2 = tag_top2 if tag_top2 is not None else np.array([4, 5, 6])
        self.tag_ditop = np.concatenate([self.tag_top1, self.tag_top2])

    def extract_targets(
        self, jet_chunk: np.ndarray, targets_chunk: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Extract both binary mask and di-top kinematics.
        
        Returns:
            Dict with keys: "targets" (binary mask) and "target_kinematics" (di-top system)
        """
        B, P, F = jet_chunk.shape
        
        # --- 1. Extract Binary Mask ---
        jet_tags = jet_chunk[..., 6]
        target_mask_bool = np.isin(jet_tags, self.match_labels)
        targets_np = target_mask_bool.astype(np.float32)
        binary_mask = targets_np[..., np.newaxis]  # (B, P, 1)
        
        # --- 2. Extract Di-Top Kinematics ---
        # Create vector array for all jets
        flat_jets_vec = vector.zip({
            "pt": jet_chunk[..., 0].flatten(),
            "eta": jet_chunk[..., 1].flatten(),
            "phi": jet_chunk[..., 2].flatten(),
            "energy": jet_chunk[..., 3].flatten(),
        })
        
        # Event indices and tags
        event_indices = np.repeat(np.arange(B), P)
        flat_tags = jet_tags.flatten()
        
        # Convert to Cartesian for accurate summation
        flat_px = flat_jets_vec.px.to_numpy()
        flat_py = flat_jets_vec.py.to_numpy()
        flat_pz = flat_jets_vec.pz.to_numpy()
        flat_E = flat_jets_vec.energy.to_numpy()
        
        # Mask for all jets belonging to either top quark
        ditop_mask = np.isin(flat_tags, self.tag_ditop)
        matched_indices = event_indices[ditop_mask]
        
        if matched_indices.size == 0:
            # Return zeros if no matched jets found
            ditop_system = np.zeros((B, 1, 4), dtype=np.float32)
        else:
            # Sum all Cartesian components for the combined system
            sum_px = np.bincount(matched_indices, weights=flat_px[ditop_mask], minlength=B)
            sum_py = np.bincount(matched_indices, weights=flat_py[ditop_mask], minlength=B)
            sum_pz = np.bincount(matched_indices, weights=flat_pz[ditop_mask], minlength=B)
            sum_E = np.bincount(matched_indices, weights=flat_E[ditop_mask], minlength=B)
            
            # Reconstruct the combined 4-vector
            ditop_vec = vector.zip({
                "px": sum_px,
                "py": sum_py,
                "pz": sum_pz,
                "E": sum_E
            })
            
            # Convert to polar coordinates (pt, eta, phi, energy)
            ditop_kinematics = np.stack([
                ditop_vec.pt.to_numpy(),
                ditop_vec.eta.to_numpy(),
                ditop_vec.phi.to_numpy(),
                ditop_vec.E.to_numpy(),
            ], axis=-1)  # Shape (B, 4)
            
            # Reshape to (B, 1, 4) - no placeholder needed
            ditop_system = ditop_kinematics[:, np.newaxis, :]  # (B, 1, 4)
        
        return {
            "targets": binary_mask,
            "target_kinematics": ditop_system
        }


class DiTopKinematicsExtractor(TargetExtractor):
    """
    Extracts combined di-top system kinematics by reconstructing both top quarks
    and combining them into a single 4-vector.
    """

    def __init__(self, tag_top1: np.ndarray = None, tag_top2: np.ndarray = None):
        # Set default truth-matching tags for both tops
        self.tag_top1 = tag_top1 if tag_top1 is not None else np.array([1, 2, 3])
        self.tag_top2 = tag_top2 if tag_top2 is not None else np.array([4, 5, 6])
        # Combined tags for the full di-top system
        self.tag_ditop = np.concatenate([self.tag_top1, self.tag_top2])

    def extract_targets(
        self, jet_chunk: np.ndarray, targets_chunk: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Reconstruct the combined di-top system by summing all truth-matched jets
        from both top quarks using vectorized operations.
        
        Args:
            jet_chunk (np.ndarray): Input jet array (B, P, F) where F includes 
                                    pt, eta, phi, energy, and truthmatch tag (index 6).
            targets_chunk (np.ndarray): Input truth targets (unused).

        Returns:
            Dict[str, np.ndarray]: Dictionary containing {"target_kinematics": ditop_system}
                                   where ditop_system has shape (B, 1, 5).
        """
        B, P, F = jet_chunk.shape
        
        # --- 1. Prepare Flattened Inputs ---
        
        jet_tags = jet_chunk[..., 6]
        
        # Create vector array for all jets
        flat_jets_vec = vector.zip({
            "pt": jet_chunk[..., 0].flatten(),
            "eta": jet_chunk[..., 1].flatten(),
            "phi": jet_chunk[..., 2].flatten(),
            "energy": jet_chunk[..., 3].flatten(),
        })
        
        # Event indices and tags
        event_indices = np.repeat(np.arange(B), P)
        flat_tags = jet_tags.flatten()
        
        # Convert to Cartesian for accurate summation
        flat_px = flat_jets_vec.px.to_numpy()
        flat_py = flat_jets_vec.py.to_numpy()
        flat_pz = flat_jets_vec.pz.to_numpy()
        flat_E = flat_jets_vec.energy.to_numpy()
        
        # --- 2. Reconstruct Combined Di-Top System ---
        
        # Mask for all jets belonging to either top quark
        ditop_mask = np.isin(flat_tags, self.tag_ditop)
        matched_indices = event_indices[ditop_mask]
        
        if matched_indices.size == 0:
            # Return zeros if no matched jets found
            ditop_system = np.zeros((B, 1, 4), dtype=np.float32)
            return {"target_kinematics": ditop_system}
        
        # Sum all Cartesian components for the combined system
        sum_px = np.bincount(matched_indices, weights=flat_px[ditop_mask], minlength=B)
        sum_py = np.bincount(matched_indices, weights=flat_py[ditop_mask], minlength=B)
        sum_pz = np.bincount(matched_indices, weights=flat_pz[ditop_mask], minlength=B)
        sum_E = np.bincount(matched_indices, weights=flat_E[ditop_mask], minlength=B)
        
        # Reconstruct the combined 4-vector
        ditop_vec = vector.zip({
            "px": sum_px,
            "py": sum_py,
            "pz": sum_pz,
            "E": sum_E
        })
        
        # Convert to polar coordinates (pt, eta, phi, energy)
        ditop_kinematics = np.stack([
            ditop_vec.pt.to_numpy(),
            ditop_vec.eta.to_numpy(),
            ditop_vec.phi.to_numpy(),
            ditop_vec.E.to_numpy(),
        ], axis=-1)  # Shape (B, 4)
        
        # --- 3. Format Output ---
        
        # Reshape to (B, 1, 4) - no placeholder needed
        ditop_system = ditop_kinematics[:, np.newaxis, :]  # (B, 1, 4)
        
        return {"target_kinematics": ditop_system}


class TopReconstructionDatasetFromH5:
    """Dataset preprocessor for binary mask target processing."""

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
        self.target_extractor = target_extractor or BinaryMaskTargetExtractor()

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

        print(f"[CONFIG] Raw path: {self.raw_file_prefix_and_path}", flush=True)
        print(f"[CONFIG] Save path: {self.save_file_prefix_and_path}", flush=True)
        print(f"[CONFIG] Stream size: {self.stream_size}", flush=True)

        self.jet_transformers = self._init_jet_transformers()
        self.target_transformers = self.target_processor.init_target_transformers()
        self.interaction_transformers = self.interaction_processor.init_interaction_transformer()

        self._prepare_datasets()
        
        print("[INIT] Complete!", flush=True)

    def _construct_path(self, directory: str, prefix: str) -> Path:
        """Helper to construct paths consistently."""
        return Path(directory) / prefix

    def _init_jet_transformers(self) -> tuple:
        """Initialize jet transformers."""
        return (
            LogMinMaxScaler(),
            StandardScaler(),
            PhiTransformer(),
            LogMinMaxScaler(),
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
        print("[FIT] Fitted transformers will be applied to all files during transformation.", flush=True)
        self._transform_all()

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
                targets_chunk = f["targets"][i : i + self.stream_size].copy()
                
                # Filter events
                event_filter = event_chunk[:, 2] == 1
                jet_chunk = jet_chunk[event_filter]
                event_chunk = event_chunk[event_filter]
                targets_chunk = targets_chunk[event_filter]
                
                if jet_chunk.shape[0] == 0:
                    continue

                # Extract targets using the configured extractor
                targets_dict = self.target_extractor.extract_targets(jet_chunk, targets_chunk)

                # Fit jet transformers
                self._fit_jet_transformers(jet_chunk)

                # Fit target transformers
                self._fit_target_transformers(targets_dict)

                # Fit interaction transformers if needed
                if self.interaction_processor.needs_interaction():
                    try:
                        interaction_chunk = create_interaction_matrix(jet_chunk)
                        self._fit_interaction_transformers(interaction_chunk)
                    except Exception as e:
                        print(f"[WARN] Interaction fit failed: {e}", flush=True)

    def _fit_jet_transformers(self, jet_chunk: np.ndarray):
        """Fit jet transformers on jet data."""
        N, P, F = jet_chunk.shape
        num_features_to_transform = len(self.jet_transformers)
        
        for i, transformer in enumerate(self.jet_transformers):
            var = jet_chunk[..., i]
            var_flat = var.reshape(-1, 1)
            transformer.partial_fit(var_flat)

    def _fit_target_transformers(self, targets_dict: Dict[str, np.ndarray]):
        """Fit target transformers on target data."""
        # Handle different target processor types
        if isinstance(self.target_processor, (DiTopKinematicsProcessor, CombinedMaskAndKinematicsProcessor)):
            # Single set of transformers for di-top kinematics (binary mask needs no fitting)
            if "target_kinematics" in targets_dict:
                target_chunk = targets_dict["target_kinematics"]
                transformers = self.target_transformers[0]  # Unpack the tuple
                
                N, M, F = target_chunk.shape
                num_features_to_transform = len(transformers)
                
                # Fit all 4 features (pt, eta, phi, energy) - no placeholder to skip
                for i, transformer in enumerate(transformers):
                    var = target_chunk[..., i]
                    var_flat = var.reshape(-1, 1)
                    transformer.partial_fit(var_flat)
        
        elif hasattr(self.target_processor, 'top_transformers'):
            # Handle processors with multiple transformer sets (like WBosonTargetProcessor)
            # Fit top transformers
            if "targets" in targets_dict:
                tops_chunk = targets_dict["targets"]
                N, M, F = tops_chunk.shape
                for i, transformer in enumerate(self.target_processor.top_transformers):
                    if i < F - 1:
                        var = tops_chunk[..., i]
                        var_flat = var.reshape(-1, 1)
                        transformer.partial_fit(var_flat)
            
            # Fit W transformers
            if "W_targets" in targets_dict:
                W_chunk = targets_dict["W_targets"]
                N, M, F = W_chunk.shape
                for i, transformer in enumerate(self.target_processor.W_transformers):
                    if i < F - 1:
                        var = W_chunk[..., i]
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

    def _transform_file(self, raw_path: Path, save_path: Path):
        """Transform a single file."""
        
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
                targets_chunk = read_f["targets"][i : i + self.stream_size].copy()

                # Filter events where event[:, 2] == 1
                event_filter = event_chunk[:, 2] == 1
                
                jet_chunk = jet_chunk[event_filter]
                event_chunk = event_chunk[event_filter]
                targets_chunk = targets_chunk[event_filter]
                

                targets_dict = self.target_extractor.extract_targets(jet_chunk, targets_chunk)
                
                if jet_chunk.shape[0] == 0:
                    continue
                
                interaction_chunk = None
                if self.interaction_processor.needs_interaction():
                    try:
                        interaction_chunk = create_interaction_matrix(jet_chunk)
                    except Exception as e:
                        print(f"[WARN] Interaction matrix creation failed: {e}", flush=True)

                jet_chunk, interaction_chunk = self._transform_data(
                    jet_chunk, interaction_chunk
                )
                
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
                    )
                    datasets_created = True

                self._save_data_chunks(
                    write_f,
                    jet_chunk,
                    event_chunk,
                    src_mask,
                    targets_dict,
                    interaction_chunk,
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
        """Transform target kinematics."""
        if isinstance(self.target_processor, (DiTopKinematicsProcessor, CombinedMaskAndKinematicsProcessor)):
            # Transform di-top kinematics (binary mask stays unchanged)
            if "target_kinematics" in targets_dict:
                target_chunk = targets_dict["target_kinematics"]
                transformers = self.target_transformers[0]  # Unpack the tuple
                
                N, M, F = target_chunk.shape
                
                # Transform all 4 features (pt, eta, phi, energy) - no placeholder
                target_transformed_list = []
                for i, transformer in enumerate(transformers):
                    var = target_chunk[..., i]
                    var_reshaped = var.reshape(-1, 1)
                    transformed_var = transformer.transform(var_reshaped).reshape(N, M, -1)
                    target_transformed_list.append(transformed_var)
                
                # Concatenate all transformed features (phi becomes 2 features)
                targets_dict["target_kinematics"] = np.concatenate(target_transformed_list, axis=-1)
        
        elif hasattr(self.target_processor, 'top_transformers'):
            # Transform tops
            if "targets" in targets_dict:
                tops_chunk = targets_dict["targets"]
                N, M, F = tops_chunk.shape
                
                tops_transformed_list = []
                for i, transformer in enumerate(self.target_processor.top_transformers):
                    if i < F - 1:
                        var = tops_chunk[..., i]
                        var_reshaped = var.reshape(-1, 1)
                        transformed_var = transformer.transform(var_reshaped).reshape(N, M, -1)
                        tops_transformed_list.append(transformed_var)
                
                tops_transformed = np.concatenate(tops_transformed_list, axis=-1)
                placeholder = tops_chunk[..., -1:]
                targets_dict["targets"] = np.concatenate([tops_transformed, placeholder], axis=-1)
            
            # Transform Ws
            if "W_targets" in targets_dict:
                W_chunk = targets_dict["W_targets"]
                N, M, F = W_chunk.shape
                
                W_transformed_list = []
                for i, transformer in enumerate(self.target_processor.W_transformers):
                    if i < F - 1:
                        var = W_chunk[..., i]
                        var_reshaped = var.reshape(-1, 1)
                        transformed_var = transformer.transform(var_reshaped).reshape(N, M, -1)
                        W_transformed_list.append(transformed_var)
                
                W_transformed = np.concatenate(W_transformed_list, axis=-1)
                placeholder = W_chunk[..., -1:]
                targets_dict["W_targets"] = np.concatenate([W_transformed, placeholder], axis=-1)
        
        return targets_dict

    def _create_datasets(
        self,
        file: h5py.File,
        jet_shape: Tuple,
        event_shape: Tuple,
        targets_dict: Dict[str, np.ndarray],
        interaction_shape: Optional[Tuple] = None,
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

        # Create datasets for all target types in targets_dict
        for key in self.target_processor.get_save_keys():
            if key in targets_dict:
                target_array = targets_dict[key]
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
            file.create_dataset(
                "interactions", 
                shape=(0, N, N, interaction_features), 
                maxshape=(None, N, N, interaction_features), 
                compression="gzip", 
                compression_opts=4,
                dtype="float32",
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

        # Save all target types
        for key in self.target_processor.get_save_keys():
            if key in targets_dict:
                file[key].resize((n1,) + file[key].shape[1:])
                file[key][n0:n1] = targets_dict[key].astype("float32")

        if interaction_chunk is not None:
            file["interactions"].resize((n1,) + file["interactions"].shape[1:])
            file["interactions"][n0:n1] = interaction_chunk.astype("float32")

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
    print("COMBINED MASK + DI-TOP KINEMATICS PROCESSOR", flush=True)
    print("="*60 + "\n", flush=True)
    
    try:
        config = load_any_config("config/top_reconstruction_config.yaml")
        
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

        # Use CombinedMaskAndKinematicsProcessor and CombinedMaskAndKinematicsExtractor
        processor = CombinedMaskAndKinematicsProcessor()
        extractor = CombinedMaskAndKinematicsExtractor(
            match_labels=np.array([1, 2, 3, 4, 5, 6]),  # For binary mask
            tag_top1=np.array([1, 2, 3]),  # For di-top kinematics
            tag_top2=np.array([4, 5, 6])   # For di-top kinematics
        )

        dataset = TopReconstructionDatasetFromH5(
            config,
            target_processor=processor,
            interaction_processor=WithInteractionProcessor(),
            target_extractor=extractor,
        )
        
        print("\n" + "="*60, flush=True)
        print("SUCCESS!", flush=True)
        print("="*60, flush=True)
        
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}", flush=True)
        import traceback
        traceback.print_exc()