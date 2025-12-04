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

        self._init_transformers()
        self._process_pipeline()

    @staticmethod
    def _construct_path(base_path: str, prefix: str) -> str:
        return f"{base_path}/{prefix}"

    def _init_transformers(self):
        """Initialize transformers for jets and interactions."""
        print("[INIT] Initializing transformers...", flush=True)
        
        self.jet_transformers = (
            LogMinMaxScaler(),
            StandardScaler(),
            PhiTransformer(),
            LogMinMaxScaler(),
            LogMinMaxScaler(),
        )
        
        if self.interaction_processor.needs_interaction():
            self.interaction_transformers = self.interaction_processor.init_interaction_transformer()
        else:
            self.interaction_transformers = None
        
        print(f"[INIT] Transformers ready: {len(self.jet_transformers)} jet transformers", flush=True)

    def _process_pipeline(self):
        """Main processing pipeline: fit, then transform."""
        raw_train = f"{self.raw_file_prefix_and_path}train.h5"
        
        print(f"\n[PIPELINE] Starting...", flush=True)
        print(f"[PIPELINE] Looking for: {raw_train}", flush=True)
        
        if not os.path.exists(raw_train):
            print(f"[ERROR] Training file not found: {raw_train}", flush=True)
            print(f"[ERROR] Current directory: {os.getcwd()}", flush=True)
            print(f"[ERROR] Files in current dir: {os.listdir('.')}", flush=True)
            return

        print(f"[PIPELINE] File found. Starting fit phase...", flush=True)
        self._fit_transformers_from_file(raw_train)
        
        print(f"[PIPELINE] Fit complete. Saving transformers...", flush=True)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        joblib.dump(
            self.jet_transformers,
            self.save_dir / "jet_transforms.joblib",
        )
        print(f"[PIPELINE] Transformers saved to {self.save_dir}", flush=True)
        
        print(f"[PIPELINE] Starting transform and save phase...", flush=True)
        file_pairs = [
            (raw_train, f"{self.save_file_prefix_and_path}train.h5"),
            (f"{self.raw_file_prefix_and_path}val.h5", f"{self.save_file_prefix_and_path}val.h5"),
            (f"{self.raw_file_prefix_and_path}test.h5", f"{self.save_file_prefix_and_path}test.h5"),
        ]

        for raw_file, save_file in file_pairs:
            if os.path.exists(raw_file):
                self._transform_and_save_file(raw_file, save_file)
            else:
                print(f"[WARN] File not found, skipping: {raw_file}", flush=True)
        
        print(f"[PIPELINE] All phases complete!", flush=True)

    def _fit_transformers_from_file(self, raw_path: str):
        """Fit jet and interaction transformers on raw training file."""
        print(f"[FIT] Starting fit on {raw_path}...", flush=True)
        
        with h5py.File(raw_path, "r") as f:
            print(f"[FIT] HDF5 file opened. Datasets: {list(f.keys())}", flush=True)
            
            if "jet" not in f:
                print(f"[ERROR] 'jet' not found in HDF5", flush=True)
                return
            
            file_len = f["jet"].shape[0]
            print(f"[FIT] Total events: {file_len}", flush=True)
            print(f"[FIT] Processing in chunks of {self.stream_size}", flush=True)
            
            chunk_count = 0
            for i in tqdm(
                range(0, file_len, self.stream_size),
                desc="Fitting",
            ):
                jet_chunk = f["jet"][i : i + self.stream_size].copy()
                event_chunk = f["event"][i : i + self.stream_size].copy()
                self._fit_jet_transformers(jet_chunk)

                # Filter events where event[:, 2] == 1
                event_filter = event_chunk[:, 2] == 1
                
                
                jet_chunk = jet_chunk[event_filter]
                
                if self.interaction_processor.needs_interaction():
                    try:
                        interaction_chunk = create_interaction_matrix(jet_chunk)
                        self._fit_interaction_transformer(interaction_chunk)
                    except Exception as e:
                        print(f"[WARN] Interaction matrix creation failed: {e}", flush=True)
                
                chunk_count += 1
        
        print(f"[FIT] Fit complete! Processed {chunk_count} chunks", flush=True)

    def _fit_jet_transformers(self, jet: np.ndarray):
        """Fit jet transformers on a chunk."""
        N, P, F = jet.shape
        num_transformed_jet_features = len(self.jet_transformers)

        jet_flat = jet[..., 0:num_transformed_jet_features].reshape(-1, num_transformed_jet_features)
        
        for i, transformer in enumerate(self.jet_transformers):
            var = jet_flat[:, i]
            valid_mask = ~np.isnan(var) & (var != 0)
            if valid_mask.sum() > 0:
                transformer.partial_fit(var[valid_mask].reshape(-1, 1))

    def _fit_interaction_transformer(self, interactions: np.ndarray):
        """Fit interaction transformer on a chunk."""
        if interactions is None or self.interaction_transformers is None:
            return
            
        N, P, P2, F = interactions.shape
        interactions_flat = interactions.reshape(-1, F)
        valid_rows = ~(np.isnan(interactions_flat).any(axis=1)) & ~(interactions_flat == 0).all(axis=1)

        if valid_rows.sum() > 0:
            self.interaction_transformers.partial_fit(interactions_flat[valid_rows])

    def _transform_and_save_file(self, raw_path: str, save_path: str):
        """Transform and save a complete file."""
        print(f"\n[TRANSFORM] Processing {os.path.basename(raw_path)} -> {os.path.basename(save_path)}...", flush=True)
        
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
                targets_mask = targets_dict["targets"]
                

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
                
                jet_chunk, src_mask, interaction_chunk = self._pad_and_src_mask(
                    jet_chunk, interaction_chunk
                )

                if not datasets_created:
                    self._create_datasets(
                        write_f,
                        jet_chunk.shape,
                        event_chunk.shape,
                        targets_mask.shape,
                        interaction_chunk.shape if interaction_chunk is not None else None,
                    )
                    datasets_created = True

                self._save_data_chunks(
                    write_f,
                    jet_chunk,
                    event_chunk,
                    src_mask,
                    targets_mask,
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

    def _create_datasets(
        self,
        file: h5py.File,
        jet_shape: Tuple,
        event_shape: Tuple,
        targets_shape: Tuple,
        interaction_shape: Optional[Tuple] = None,
    ):
        """Create HDF5 dataset groups."""
        _, N_jets, jet_features = jet_shape
        _, event_features = event_shape
        _, M_targets, target_features = targets_shape

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
        file.create_dataset(
            "targets", 
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
        targets_chunk: np.ndarray,
        interaction_chunk: Optional[np.ndarray] = None,
    ):
        """Save data chunks to HDF5."""
        cur_len = file["jet"].shape[0]
        n0, n1 = cur_len, cur_len + jet_chunk.shape[0]

        file["jet"].resize((n1,) + file["jet"].shape[1:])
        file["event"].resize((n1,) + file["event"].shape[1:])
        file["src_mask"].resize((n1,) + file["src_mask"].shape[1:])
        file["targets"].resize((n1,) + file["targets"].shape[1:])

        file["jet"][n0:n1] = jet_chunk[..., :-1].astype("float32")
        file["event"][n0:n1] = event_chunk.astype("float32")
        file["src_mask"][n0:n1] = src_mask_chunk.astype("float32")
        file["targets"][n0:n1] = targets_chunk.astype("float32")

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
    print("BINARY MASK PROCESSOR", flush=True)
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

        processor = BinaryMaskTargetProcessor()
        extractor = BinaryMaskTargetExtractor(match_labels=np.array([1, 2, 3, 4, 5, 6]))

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