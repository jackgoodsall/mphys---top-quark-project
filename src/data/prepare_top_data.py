import numpy as np
import h5py
import joblib
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
import vector
from tqdm import tqdm # <-- ADDED PROGRESS BAR IMPORT
import awkward as ak
import os
import sys
# Assuming these imports set up your environment and dependencies
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))) 

from src.data_utls.scalers import LogMinMaxScaler, StandardScaler, PhiTransformer
from src.utils.utils import load_any_config
from utils import (
    apply_mask,
    calculate_energy_value,
    convert_polar_to_cartesian,
    create_interaction_matrix,
)

# --- TargetProcessor Definitions (Unchanged) ---

class TargetProcessor(ABC):
    """Abstract base for different target processing strategies."""

    @abstractmethod
    def get_target_count(self) -> int:
        """Number of targets per event (e.g., 2 for tops, 4 for interaction)."""
        pass

    @abstractmethod
    def init_target_transformers(self) -> tuple:
        """Initialize target transformers."""
        pass

    @abstractmethod
    def process_targets(self, targets_chunk: np.ndarray, is_temp: bool) -> Dict[str, np.ndarray]:
        """Apply target-specific processing (energy calc, coordinate transform, etc.)."""
        pass

    @abstractmethod
    def get_save_keys(self) -> list:
        """Return list of dataset keys to save (e.g., ['targets'] or ['targets', 'W_targets'])."""
        pass

    @abstractmethod
    def reshape_targets(self, targets_chunk: np.ndarray) -> np.ndarray:
        """Reshape targets for this strategy."""
        pass


class SimpleTargetProcessor(TargetProcessor):
    """Processes 2 top quarks with energy calculation."""

    def get_target_count(self) -> int:
        return 2

    def init_target_transformers(self) -> tuple:
        return (
            LogMinMaxScaler(),
            StandardScaler(),
            StandardScaler(),
            LogMinMaxScaler(),
        )

    def process_targets(self, targets_chunk: np.ndarray, is_temp: bool) -> Dict[str, np.ndarray]:
        # targets_chunk is (B, 2, 5). Feature 3 is mass. Feature 4 is PDG ID (or similar placeholder).
        if not is_temp:
            targets_chunk[..., 3] = calculate_energy_value(targets_chunk[..., :])
        return {"targets": targets_chunk}

    def reshape_targets(self, targets_chunk: np.ndarray) -> np.ndarray:
        return targets_chunk.reshape(-1, 2, 5)

    def get_save_keys(self) -> list:
        return ["targets"]


class CartesianTargetProcessor(TargetProcessor):
    """Processes 2 top quarks with polar-to-cartesian coordinate transformation."""

    def get_target_count(self) -> int:
        return 2

    def init_target_transformers(self) -> tuple:
        return (
            LogMinMaxScaler(),
            StandardScaler(),
            StandardScaler(),
            StandardScaler(),
        )

    def process_targets(self, targets_chunk: np.ndarray, is_temp: bool) -> Dict[str, np.ndarray]:
        if not is_temp:
            targets_chunk[..., :4] = convert_polar_to_cartesian(targets_chunk)
        return {"targets": targets_chunk}

    def reshape_targets(self, targets_chunk: np.ndarray) -> np.ndarray:
        return targets_chunk.reshape(-1, 2, 5)

    def get_save_keys(self) -> list:
        return ["targets"]


class InteractionTargetProcessor(TargetProcessor):
    """Processes 4 targets with interaction matrix."""

    def get_target_count(self) -> int:
        return 4

    def init_target_transformers(self) -> tuple:
        return (
            LogMinMaxScaler(),
            StandardScaler(),
            PhiTransformer(),
            LogMinMaxScaler(),
        )

    def process_targets(self, targets_chunk: np.ndarray, is_temp: bool) -> Dict[str, np.ndarray]:
        if not is_temp:
            targets_chunk[..., 3] = calculate_energy_value(targets_chunk[..., :])
        return {"targets": targets_chunk}

    def reshape_targets(self, targets_chunk: np.ndarray) -> np.ndarray:
        return targets_chunk.reshape(-1, 4, 5)

    def get_save_keys(self) -> list:
        return ["targets"]


class WBosonTargetProcessor(TargetProcessor):
    """Processes 2 tops and 2 W bosons separately."""

    def __init__(self):
        self.top_transformers = None
        self.W_transformers = None

    def get_target_count(self) -> int:
        return 2  # Returns per target type (Tops and Ws)

    def init_target_transformers(self) -> Tuple[tuple, tuple]:
        # This returns the TOP transformers
        top_trans = (
            LogMinMaxScaler(),
            StandardScaler(),
            PhiTransformer(),
            LogMinMaxScaler(),
        )
        # This returns the W transformers, initialized separately in TopReconstructionDatasetFromH5
        W_trans = (
            LogMinMaxScaler(),
            StandardScaler(),
            PhiTransformer(),
            LogMinMaxScaler(),
        )
        self.top_transformers = top_trans
        self.W_transformers = W_trans
        return top_trans, W_trans # The main method expects a tuple of tuples

    def process_targets(self, targets_dict: Dict[str, np.ndarray], is_temp: bool) -> Dict[str, np.ndarray]:
        """Apply target-specific processing for both Tops and Ws."""
        
        # NOTE: targets_dict contains both "targets" (tops) and "W_targets" (Ws)
        
        # Process Tops
        tops_chunk = targets_dict["targets"]
        #if not is_temp:
            # Assume feature 3 is mass (placeholder), calculate energy
            #tops_chunk[..., 3] = calculate_energy_value(tops_chunk[..., :])
        targets_dict["targets"] = tops_chunk

        # Process Ws
        W_chunk = targets_dict["W_targets"]
        #if not is_temp:
            # Assume feature 3 is mass (placeholder), calculate energy
            #W_chunk[..., 3] = calculate_energy_value(W_chunk[..., :])
        targets_dict["W_targets"] = W_chunk

        return targets_dict

    def reshape_targets(self, targets_chunk: np.ndarray) -> np.ndarray:
        # This method is not used directly by the main processor for this class, 
        # as targets_dict is handled directly. Leaving the original return for consistency.
        return targets_chunk.reshape(-1, 2, 5)

    def get_save_keys(self) -> list:
        return ["targets", "W_targets"]

# --- InteractionProcessor Definitions (Unchanged) ---

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


# --- TargetExtractor Definitions (Unchanged for Truth) ---

class TargetExtractor(ABC):
    """Abstract base for different target extraction strategies."""

    @abstractmethod
    def extract_targets(
        self, jet_chunk: np.ndarray, targets_chunk: np.ndarray
    ) -> Dict[str, np.ndarray]: # Changed return type to Dict[str, np.ndarray] for W_targets
        """Extract targets from data. Returns dictionary of targets."""
        pass


class TruthTargetExtractor(TargetExtractor):
    """Extracts targets directly from truth information."""

    def extract_targets(
        self, jet_chunk: np.ndarray, targets_chunk: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Simply return tops as 'targets'."""
        top_quark_mask = np.abs(targets_chunk[..., 4]) == 6
        extracted = targets_chunk[top_quark_mask]
        
        # Truth data often contains all particles, but for simplicity, we only extract tops here
        # For full truth extraction including Ws, this would need complex masking/sorting.
        # We assume the standard case returns only Tops unless Ws are explicitly matched/extracted.
        return {"targets": extracted}


# --- Optimized and Extended ReconstructedTargetExtractor (Crucial Changes) ---
import numpy as np
import vector
from typing import Dict, Any, Tuple

class ReconstructedTargetExtractor:
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

# --- TopReconstructionDatasetFromH5 (Core Logic) ---

class TopReconstructionDatasetFromH5:
    """Unified dataset preprocessor for top reconstruction variants."""

    def __init__(
        self,
        config: Dict[str, Any],
        target_processor: TargetProcessor,
        interaction_processor: InteractionProcessor = None,
        target_extractor: TargetExtractor = None,
    ):
        self.raw_file_config = config["root_dataset_prepper"]
        self.preprocessing_config = config.get("preprocessing", None)
        self.target_processor = target_processor
        self.interaction_processor = interaction_processor or NoInteractionProcessor()
        self.target_extractor = target_extractor or TruthTargetExtractor()

        self.raw_file_prefix_and_path = self._construct_path(
            self.raw_file_config["save_path"],
            self.raw_file_config["save_file_prefix"],
        )
        self.save_dir = Path(self.preprocessing_config["save_path"])
        self.save_file_prefix_and_path = self._construct_path(
            self.preprocessing_config["save_path"],
            self.preprocessing_config["save_file_prefix"],
        )
        self.stream_size = self.preprocessing_config["stream_size"]

        self._init_transformers()
        self._process_pipeline()

    @staticmethod
    def _construct_path(base_path: str, prefix: str) -> str:
        return f"{base_path}/{prefix}"

    # ... _init_transformers and _process_pipeline are unchanged ...
    # (Leaving them out for brevity, assuming they are in the class)
    def _init_transformers(self):
        self.jet_transformers = (
            LogMinMaxScaler(),
            StandardScaler(),
            PhiTransformer(),
            LogMinMaxScaler(),
            LogMinMaxScaler(),
        )
        
        is_WBoson_processor = isinstance(self.target_processor, WBosonTargetProcessor)
        
        if is_WBoson_processor:
            top_trans, W_trans = self.target_processor.init_target_transformers()
            self.target_transformers = top_trans
            self.W_target_transformers = W_trans
        else:
            self.target_transformers = self.target_processor.init_target_transformers()
            self.W_target_transformers = None
        
        self.invariant_mass_transformer = LogMinMaxScaler()

        if self.interaction_processor.needs_interaction():
            self.interaction_transformers = (
                self.interaction_processor.init_interaction_transformer()
            )

    def _process_pipeline(self):
        """Main processing pipeline: fit, then transform."""
        raw_train = f"{self.raw_file_prefix_and_path}train.h5"
        
        # Fit on training data only
        self._fit_transformers_from_file(raw_train)

        # Save transformers
        self.save_dir.mkdir(exist_ok=True, parents=True)
        joblib.dump(
            self.target_transformers,
            self.save_dir / "target_transforms.joblib",
        )
        if self.W_target_transformers is not None:
             joblib.dump(
                self.W_target_transformers,
                self.save_dir / "W_target_transforms.joblib",
            )
        joblib.dump(
            self.invariant_mass_transformer,
            self.save_dir / "invariant_mass_transform.joblib",
        )

        # Transform and save all splits
        file_pairs = [
            (raw_train, f"{self.save_file_prefix_and_path}train.h5"),
            (f"{self.raw_file_prefix_and_path}val.h5", f"{self.save_file_prefix_and_path}val.h5"),
            (f"{self.raw_file_prefix_and_path}test.h5", f"{self.save_file_prefix_and_path}test.h5"),
        ]

        for raw_file, save_file in file_pairs:
            self._transform_and_save_file(raw_file, save_file)


    # ----------------------------------------------------------------------
    # 1. CORE PROCESSING METHODS (Refactored)
    # ----------------------------------------------------------------------

    def _fit_transformers_from_file(self, raw_path: str):
        """Fit all transformers on raw training file (with TQDM progress)."""
        print(f"Fitting transformers on {raw_path}...")
        with h5py.File(raw_path, "r") as f:
            file_len = f["jet"].shape[0]
            
            for i in tqdm(
                range(0, file_len, self.stream_size),
                desc="Fitting Chunks",
                unit="chunk"
            ):
                jet_chunk = f["jet"][i : i + self.stream_size]
                event_chunk = f["event"][i : i + self.stream_size]
                targets_chunk = f["targets"][i : i + self.stream_size] # raw truth targets

                # Extract targets (truth or reconstructed) into a dictionary
                targets_dict = self.target_extractor.extract_targets(jet_chunk, targets_chunk)
                
                # --- Step 1: Separate and Process Tops/Ws ---
                tops_chunk = targets_dict["targets"]
                ws_chunk = targets_dict.get("W_targets", None) # Get Ws if they exist

                if isinstance(self.target_processor, WBosonTargetProcessor):
                    # Process both tops and Ws
                    processed_dict = self.target_processor.process_targets(targets_dict, is_temp=True)
                    tops_chunk = processed_dict["targets"]
                    ws_chunk = processed_dict["W_targets"]
                else:
                    # Process tops only (W_targets remain None or extracted as is)
                    tops_only = self.target_processor.process_targets(tops_chunk, is_temp=True)
                    tops_chunk = tops_only["targets"]

                # --- Step 2: Apply Selection Cuts ---
                # Pass separated chunks to selection cuts
                mask = self._selection_cuts_separate(jet_chunk, event_chunk, tops_chunk, ws_chunk)
                
                # --- Step 3: Apply Masking ---
                # Pass separated chunks to apply_mask
                arrays_to_mask = [jet_chunk, event_chunk, tops_chunk]
                if ws_chunk is not None:
                    arrays_to_mask.append(ws_chunk)
                
                masked_chunks = apply_mask(tuple(arrays_to_mask), mask)
                
                # Assign masked results back
                jet_chunk, event_chunk, tops_chunk = masked_chunks[0:3]
                if ws_chunk is not None:
                    ws_chunk = masked_chunks[3]
                
                # --- Step 4: Reconstruct targets_dict for subsequent methods ---
                targets_dict = {"targets": tops_chunk}
                if ws_chunk is not None:
                    targets_dict["W_targets"] = ws_chunk

                # Calculate invariant mass of ttbar system
                invariant_masses = self._calculate_ttbar_invariant_mass(targets_dict)

                interaction_chunk = (
                    create_interaction_matrix(jet_chunk)
                    if self.interaction_processor.needs_interaction()
                    else None
                )

                self._fit_transformers(jet_chunk, targets_dict, invariant_masses, interaction_chunk)


    def _transform_and_save_file(self, raw_path: str, save_path: str):
        """Transform and save a complete file (with TQDM progress)."""
        print(f"Transforming and saving {raw_path} to {save_path}...")
        with h5py.File(raw_path, "r") as read_f, h5py.File(save_path, "w") as write_f:
            file_len = read_f["jet"].shape[0]
            datasets_created = False

            for i in tqdm(
                range(0, file_len, self.stream_size),
                desc=f"Processing {raw_path.split('/')[-1]}",
                unit="chunk"
            ):
                jet_chunk = read_f["jet"][i : i + self.stream_size]
                event_chunk = read_f["event"][i : i + self.stream_size]
                targets_chunk = read_f["targets"][i : i + self.stream_size]

                # Extract targets (truth or reconstructed) into a dictionary
                targets_dict = self.target_extractor.extract_targets(jet_chunk, targets_chunk)
                
                # --- Step 1: Separate and Process Tops/Ws ---
                tops_chunk = targets_dict["targets"]
                ws_chunk = targets_dict.get("W_targets", None)

                if isinstance(self.target_processor, WBosonTargetProcessor):
                    processed_dict = self.target_processor.process_targets(targets_dict, is_temp=False)
                    tops_chunk = processed_dict["targets"]
                    ws_chunk = processed_dict["W_targets"]
                else:
                    tops_only = self.target_processor.process_targets(tops_chunk, is_temp=False)
                    tops_chunk = tops_only["targets"]
                    
                # --- Step 2: Apply Selection Cuts ---
                mask = self._selection_cuts_separate(jet_chunk, event_chunk, tops_chunk, ws_chunk)
                
                # --- Step 3: Apply Masking ---
                arrays_to_mask = [jet_chunk, event_chunk, tops_chunk]
                if ws_chunk is not None:
                    arrays_to_mask.append(ws_chunk)
                
                masked_chunks = apply_mask(tuple(arrays_to_mask), mask)
                
                # Assign masked results back
                jet_chunk, event_chunk, tops_chunk = masked_chunks[0:3]
                if ws_chunk is not None:
                    ws_chunk = masked_chunks[3]

                # --- Step 4: Reconstruct targets_dict for subsequent methods ---
                targets_dict = {"targets": tops_chunk}
                if ws_chunk is not None:
                    targets_dict["W_targets"] = ws_chunk

                # Calculate invariant mass of ttbar system
                invariant_masses = self._calculate_ttbar_invariant_mass(targets_dict)

                interaction_chunk = (
                    create_interaction_matrix(jet_chunk)
                    if self.interaction_processor.needs_interaction()
                    else None
                )

                # --- Step 5: Transformation ---
                jet_chunk, targets_dict, invariant_masses, interaction_chunk = self._transform_separate(
                    jet_chunk, tops_chunk, ws_chunk, invariant_masses, interaction_chunk
                )
                
                # Check for empty chunk after selection/transformation
                if jet_chunk.shape[0] == 0:
                    continue
                    
                jet_chunk, src_mask, interaction_chunk = self._pad_and_src_mask(
                    jet_chunk, interaction_chunk
                )

                if not datasets_created:
                    self._create_datagroups_separate(
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
                    targets_dict, # targets_dict is still required here for dynamic keys
                    invariant_masses,
                    interaction_chunk,
                )

    # ----------------------------------------------------------------------
    # 2. HELPER METHODS (Modified)
    # ----------------------------------------------------------------------

    def _selection_cuts_separate(
        self, 
        jet: np.ndarray, 
        event: np.ndarray, 
        tops: np.ndarray, 
        Ws: Optional[np.ndarray]
    ) -> np.ndarray:
        """Apply event selection cuts using separate target arrays."""
        
        # Require event[:, 2] == 1
        event_mask = (event[:, 2] == 1)
        
        # Filter out events with no valid targets (all zeros from reconstruction)
        has_valid_targets = np.ones(jet.shape[0], dtype=bool)
        
        # Check Tops
        valid_tops = np.any(tops[..., 0:4] != 0, axis=(1, 2)) # Check the 4-vector features
        has_valid_targets = has_valid_targets & valid_tops
        
        # Check Ws if present
        if Ws is not None:
            valid_Ws = np.any(Ws[..., 0:4] != 0, axis=(1, 2))
            has_valid_targets = has_valid_targets & valid_Ws
        
        return event_mask & has_valid_targets

    def _transform_separate(
        self,
        jet: np.ndarray,
        tops: np.ndarray,
        Ws: Optional[np.ndarray],
        invariant_masses: np.ndarray,
        interactions: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray, Optional[np.ndarray]]:
        """Transform data using separate target arrays."""
        N, P, F = jet.shape
        num_transformed_jet_features = len(self.jet_transformers) 
        
        # 1. Transform jets (Unchanged logic)
        jet_transformed_list = []
        for i, transformer in enumerate(self.jet_transformers):
            var = jet[..., i]
            transformed_var = transformer.transform(var.reshape(-1, 1)).reshape(N, P, -1)
            jet_transformed_list.append(transformed_var)
            
        jet_transformed_array = np.concatenate(jet_transformed_list, axis=-1)
        non_transformed_jets = jet[..., num_transformed_jet_features:]
        jet = np.concatenate((jet_transformed_array, non_transformed_jets), axis=-1)


        # 2. Transform Top targets (tops)
        M_targets = tops.shape[1]
        targets_4vec = tops[..., 0:4]
        targets_flat = targets_4vec.reshape(-1, 4)
        
        targets_transformed_list = []
        for i, transformer in enumerate(self.target_transformers):
            var = targets_flat[:, i]
            transformed_var = transformer.transform(var.reshape(-1, 1)).reshape(N, M_targets, -1)
            targets_transformed_list.append(transformed_var)
            
        targets_transformed_4vec = np.concatenate(targets_transformed_list, axis=-1)
        non_4vec_targets = tops[..., 4:]
        tops_transformed = np.concatenate((targets_transformed_4vec, non_4vec_targets), axis=-1)


        # 3. Transform W targets if present (Ws)
        Ws_transformed = None
        if Ws is not None and self.W_target_transformers is not None:
            M_W_targets = Ws.shape[1]
            W_targets_4vec = Ws[..., 0:4]
            W_targets_flat = W_targets_4vec.reshape(-1, 4)

            W_targets_transformed_list = []
            for i, transformer in enumerate(self.W_target_transformers):
                var = W_targets_flat[:, i]
                transformed_var = transformer.transform(var.reshape(-1, 1)).reshape(N, M_W_targets, -1)
                W_targets_transformed_list.append(transformed_var)
                
            W_targets_transformed_4vec = np.concatenate(W_targets_transformed_list, axis=-1)
            non_4vec_W_targets = Ws[..., 4:]
            Ws_transformed = np.concatenate((W_targets_transformed_4vec, non_4vec_W_targets), axis=-1)


        # 4. Transform invariant masses (Unchanged logic)
        invariant_masses_transformed = self.invariant_mass_transformer.transform(
            invariant_masses.reshape(-1, 1)
        ).reshape(-1)

        # 5. Transform interactions (Unchanged logic)
        interactions_transformed = interactions
        if interactions is not None and self.interaction_processor.needs_interaction():
            interactions_flat = interactions.reshape(-1, 1)
            interactions_transformed = self.interaction_transformers.transform(
                interactions_flat
            ).reshape(N, P, P, -1)
            
        # Reconstruct targets_dict for return consistency
        targets_dict = {"targets": tops_transformed}
        if Ws_transformed is not None:
            targets_dict["W_targets"] = Ws_transformed


        return jet, targets_dict, invariant_masses_transformed, interactions_transformed

    # ... _create_datagroups and _save_data_chunks are unchanged ...
    # (They must remain in the class, using the reconstructed targets_dict for dynamic HDF5 keys)
    def _create_datagroups_separate(
        self,
        file: h5py.File,
        jet_shape: Tuple,
        event_shape: Tuple,
        targets_dict: Dict[str, np.ndarray], # targets_dict is needed here
        interaction_shape: Optional[Tuple] = None,
    ):
        """Create HDF5 dataset groups."""
        _, N_jets, jet_features = jet_shape
        _, event_features = event_shape

        file.create_dataset(
            "jet", shape=(0, N_jets, jet_features -1 ), maxshape=(None, N_jets, jet_features -1), compression="gzip", compression_opts=4,
        )
        file.create_dataset(
            "event", shape=(0, event_features), maxshape=(None, event_features), compression="gzip", compression_opts=4,
        )
        file.create_dataset(
            "src_mask", shape=(0, N_jets), maxshape=(None, N_jets), compression="gzip", compression_opts=4,
        )

        # Create target datasets dynamically based on keys in targets_dict
        for key, targets in targets_dict.items():
            _, M, Q = targets.shape
            file.create_dataset(
                key, shape=(0, M, Q), maxshape=(None, M, Q), compression="gzip", compression_opts=4,
            )

        # Create invariant mass dataset
        file.create_dataset(
            "inv_mass", shape=(0,), maxshape=(None,), compression="gzip", compression_opts=4,
        )

        # Create interaction dataset if needed
        if interaction_shape is not None:
            _, N, N, interaction_features = interaction_shape
            file.create_dataset(
                "interactions", shape=(0, N, N, interaction_features), maxshape=(None, N, N, interaction_features), compression="gzip", compression_opts=4,
            )

    def _save_data_chunks(
        self,
        file: h5py.File,
        jet_chunk: np.ndarray,
        event_chunk: np.ndarray,
        src_mask_chunk: np.ndarray,
        targets_dict: Dict[str, np.ndarray], # targets_dict is needed here
        invariant_masses: np.ndarray,
        interaction_chunk: Optional[np.ndarray] = None,
    ):
        """Save data chunks to HDF5."""
        cur_len = file["jet"].shape[0]
        n0, n1 = cur_len, cur_len + jet_chunk.shape[0]

        file["jet"].resize((n1,) + file["jet"].shape[1:])
        file["event"].resize((n1,) + file["event"].shape[1:])
        file["src_mask"].resize((n1,) + file["src_mask"].shape[1:])
        file["inv_mass"].resize((n1,))

        file["jet"][n0:n1] = jet_chunk[... , :-1].astype("float32")
        file["event"][n0:n1] = event_chunk.astype("float32")
        file["src_mask"][n0:n1] = src_mask_chunk.astype("float32")
        file["inv_mass"][n0:n1] = invariant_masses.astype("float32")

        # Save all target types (tops, Ws, etc.)
        for key, targets in targets_dict.items():
            file[key].resize((n1,) + file[key].shape[1:])
            file[key][n0:n1] = targets.astype("float32")

        if interaction_chunk is not None:
            file["interactions"].resize((n1,) + file["interactions"].shape[1:])
            file["interactions"][n0:n1] = interaction_chunk.astype("float32")

# ... (other utility methods remain unchanged) ...
    def _calculate_ttbar_invariant_mass(self, targets_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate invariant mass of ttbar system (sum of two tops) using 
        vectorized operations by zipping arrays in their (B, 2) shape.
        """
        tops = targets_dict["targets"]  # Shape (B, 2, 5)
        B = tops.shape[0]
        

        # 2. Extract components (Shape: B, 2)
        pt = tops[:, :, 0]
        eta = tops[:, :, 1]
        phi = tops[:, :, 2]
        # Note: Use 'mass' here, as the input array uses the Polar basis.
        energy = tops[:, :, 3] 

        # 3. Zip directly in the (B, 2) structure
        # The 'vector' library is smart enough to structure the resulting VectorArray
        # into B events, each containing a list of 2 4-vectors.
        all_tops_vec = vector.zip({
            "pt": pt,
            "eta": eta,
            "phi": phi,
            "energy": energy,
        })
        # all_tops_vec now has the structure ak.Array[B * 2 * {pt, eta, phi, mass}]
        
        # 4. Sum the two tops for each event
        # This element-wise addition sums top 0 and top 1 across all B events.
        ttbar_4vec = all_tops_vec[:, 0] + all_tops_vec[:, 1]
        # ttbar_4vec now has the structure ak.Array[B * {pt, eta, phi, mass}] (the summed ttbar system)
        
        # 5. Extract mass and apply mask
        invariant_masses = ttbar_4vec.mass.to_numpy()
        
        return invariant_masses.astype(np.float32)

    def _pad_and_src_mask(
        self, jet_chunk: np.ndarray, interaction_chunk: Optional[np.ndarray] = None, pad_value: float = 0
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Pad and create source mask."""
        src_mask = np.all(jet_chunk[..., 0:len(self.jet_transformers)] == 0, axis=-1)
        jet_chunk = np.nan_to_num(jet_chunk, nan=pad_value)

        if interaction_chunk is not None:
            interaction_chunk = np.nan_to_num(interaction_chunk, nan=0)

        return jet_chunk, src_mask, interaction_chunk


        # --- Transformer methods ---
    def _fit_transformers(
        self,
        jet: np.ndarray,
        targets_dict: Dict[str, np.ndarray],
        invariant_masses: np.ndarray,
        interactions: Optional[np.ndarray] = None,
    ):
        """Fit transformers on data."""
        N, P, F = jet.shape
        num_transformed_jet_features = len(self.jet_transformers) 

        # 1. Fit jet transformers
        jet_flat = jet[..., 0:num_transformed_jet_features].reshape(-1, num_transformed_jet_features)
        for i, transformer in enumerate(self.jet_transformers):
            var = jet_flat[:, i]
            valid_mask = var != 0
            # Ensure array is not empty before fitting
            if var[valid_mask].size > 0:
                transformer.partial_fit(var[valid_mask].reshape(-1, 1))

        # 2. Fit Top target transformers
        targets = targets_dict["targets"]
        targets_4vec = targets[..., 0:4]
        targets_flat = targets_4vec.reshape(-1, 4)
        
        for i, transformer in enumerate(self.target_transformers):
            var = targets_flat[:, i]
            valid_mask = var != 0
            if var[valid_mask].size > 0:
                transformer.partial_fit(var[valid_mask].reshape(-1, 1))

        # 3. Fit W target transformers
        if "W_targets" in targets_dict and self.W_target_transformers is not None:
            W_targets = targets_dict["W_targets"]
            W_targets_4vec = W_targets[..., 0:4]
            W_targets_flat = W_targets_4vec.reshape(-1, 4)

            for i, transformer in enumerate(self.W_target_transformers):
                var = W_targets_flat[:, i]
                valid_mask = var != 0
                if var[valid_mask].size > 0:
                    transformer.partial_fit(var[valid_mask].reshape(-1, 1))

        # 4. Fit invariant mass transformer
        valid_mass_mask = invariant_masses != 0
        if invariant_masses[valid_mass_mask].size > 0:
            self.invariant_mass_transformer.partial_fit(invariant_masses[valid_mass_mask].reshape(-1, 1))

        # 5. Fit interaction transformer
        if interactions is not None and self.interaction_processor.needs_interaction():
            interactions_flat = interactions.flatten()
            valid_int_mask = interactions_flat != 0
            if interactions_flat[valid_int_mask].size > 0:
                self.interaction_transformers.partial_fit(interactions_flat[valid_int_mask].reshape(-1, 1))

# Usage examples:
if __name__ == "__main__":
    # NOTE: This part requires a valid 'config/top_reconstruction_config.yaml'
    # and local files/directories set up for execution.
    try:
        config = load_any_config("config/top_reconstruction_config.yaml")

        # ============================================
        # Option 3: Reconstructed Tops AND W Bosons (Full System)
        # ============================================
        print("\n--- Processing Reconstructed Tops and W Bosons ---")
        dataset3 = TopReconstructionDatasetFromH5(
            config,
            WBosonTargetProcessor(), # Processor handles both Tops and Ws
            WithInteractionProcessor(),
            target_extractor=ReconstructedTargetExtractor(
                tag_top1=np.array([1, 2, 3]),
                tag_top2=np.array([4, 5, 6]),
                tag_W1=np.array([2, 3]),
                tag_W2=np.array([5, 6]),
            ),
        )

    except FileNotFoundError as e:
        print(f"\n[ERROR] Configuration or data file not found: {e}")
        print("Please ensure 'config/top_reconstruction_config.yaml' and the HDF5 data files exist.")