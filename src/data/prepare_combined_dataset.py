"""
Prepare combined hadronic + semi-leptonic processed HDF5 files.

Two use-cases:
  1. HADRONIC ONLY (original):   concatenate old + inclusive hadronic processed files.
  2. LEPTONIC COMBINED (new):    merge hadronic processed + semi-leptonic processed.
     Hadronic source files that lack the leptonic-extension keys
     (particle_type, chain_type, globals, neutrino_truth) get them filled with
     sensible all-hadronic defaults.

Usage:
    # Combined hadronic (original behaviour)
    python src/data/prepare_combined_dataset.py --mode hadronic

    # Combined hadronic + semi-leptonic (new)
    python src/data/prepare_combined_dataset.py --mode leptonic
"""

import argparse
import numpy as np
import h5py
from pathlib import Path

CHUNK_SIZE = 500_000

# ── Standard processed keys (both modes) ─────────────────────────────────────
BASE_KEYS = ["jet", "src_mask", "interactions",
             "masks_tops", "masks_Ws", "kinematics_tops", "kinematics_Ws",
             "valid_tops", "valid_Ws"]

# ── Leptonic-extension keys ───────────────────────────────────────────────────
LEPTONIC_KEYS = ["particle_type", "chain_type", "globals", "neutrino_truth"]

# ── Defaults when a hadronic file lacks a leptonic key ───────────────────────
def _make_leptonic_defaults(n_events: int, N_particles: int) -> dict:
    """
    Build all-hadronic default arrays for an event block of size n_events.
    N_particles = number of particle slots in the jet array for this file.
    """
    return {
        "particle_type":  np.zeros((n_events, N_particles), dtype=np.uint8),
        "chain_type":     np.zeros((n_events, 2),           dtype=np.uint8),
        "globals":        np.zeros((n_events, 6),           dtype=np.float32),
        "neutrino_truth": np.zeros((n_events, 2, 1),        dtype=np.float32),
    }


def concatenate_to_output(src_paths: list, dst_path: Path,
                          keys: list, fill_leptonic_defaults: bool = False):
    """
    Concatenate multiple processed HDF5 files into dst_path.
    Reads only the given keys. If fill_leptonic_defaults=True, files that
    lack LEPTONIC_KEYS get them filled with zeros (all-hadronic defaults).
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Determine output shapes from first source ──
    ref_path = src_paths[0]
    with h5py.File(ref_path, "r") as ref:
        shapes = {}
        dtypes = {}
        for k in keys:
            if k in ref:
                shapes[k] = ref[k].shape[1:]
                dtypes[k] = ref[k].dtype
            elif fill_leptonic_defaults and k in LEPTONIC_KEYS:
                # shape determined per-file from jet N_particles; defer to write time
                shapes[k] = None
                dtypes[k] = "uint8" if k in {"particle_type", "chain_type"} else "float32"

    with h5py.File(dst_path, "w") as out:
        # Create resizable datasets (using None shapes for deferred keys)
        for k in keys:
            if shapes.get(k) is not None:
                out.create_dataset(
                    k,
                    shape=(0,) + shapes[k],
                    maxshape=(None,) + shapes[k],
                    dtype=dtypes[k],
                    compression="gzip" if k != "interactions" else "lzf",
                    compression_opts=4 if k != "interactions" else None,
                )

        total = 0
        for src_path in src_paths:
            print(f"  Reading {src_path.name} ...", flush=True)
            with h5py.File(src_path, "r") as src:
                n = src["jet"].shape[0]
                N_particles = src["jet"].shape[1]

                for start in range(0, n, CHUNK_SIZE):
                    end       = min(start + CHUNK_SIZE, n)
                    chunk_len = end - start

                    for k in keys:
                        if k in src:
                            chunk = src[k][start:end]
                        elif fill_leptonic_defaults and k in LEPTONIC_KEYS:
                            defaults = _make_leptonic_defaults(chunk_len, N_particles)
                            chunk    = defaults[k]
                        else:
                            continue

                        # Create dataset on first encounter if shape was deferred
                        if k not in out:
                            shape_rest = chunk.shape[1:]
                            out.create_dataset(
                                k,
                                shape=(0,) + shape_rest,
                                maxshape=(None,) + shape_rest,
                                dtype=dtypes.get(k, chunk.dtype),
                                compression="gzip",
                                compression_opts=4,
                            )

                        cur     = out[k].shape[0]
                        new_len = cur + chunk_len
                        out[k].resize((new_len,) + out[k].shape[1:])
                        out[k][cur:new_len] = chunk

                    total += chunk_len

            print(f"    -> {total:,} events so far", flush=True)

    print(f"  Done: {dst_path.name} ({total:,} events)\n", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Mode: hadronic (original two-dataset merge)
# ─────────────────────────────────────────────────────────────────────────────
def run_hadronic():
    """Concatenate old + inclusive hadronic processed files (original behaviour)."""
    DATA_DIR   = Path("data/topquarkreconstruction/masked_targets_combined")
    OUTPUT_DIR = Path("data/topquarkreconstruction/masked_targets_combined/merged")

    SPLITS = {
        "train": ["ttbar_preprocessed_train.h5",           "ttbar_preprocessed_inclusive_train.h5"],
        "val":   ["ttbar_preprocessed_val.h5",             "ttbar_preprocessed_inclusive_val.h5"],
        "test":  ["ttbar_preprocessed_test.h5",            "ttbar_preprocessed_inclusive_test.h5"],
    }

    keys = BASE_KEYS + LEPTONIC_KEYS  # include leptonic defaults so output is combinable later

    for split, filenames in SPLITS.items():
        src_paths = [DATA_DIR / f for f in filenames if (DATA_DIR / f).exists()]
        if not src_paths:
            print(f"[SKIP] No files found for split={split}")
            continue
        dst = OUTPUT_DIR / f"ttbar_preprocessed_{split}.h5"
        print(f"=== {split} ({len(src_paths)} files) ===")
        concatenate_to_output(src_paths, dst, keys, fill_leptonic_defaults=True)


# ─────────────────────────────────────────────────────────────────────────────
# Mode: leptonic (hadronic + semi-leptonic merge)
# ─────────────────────────────────────────────────────────────────────────────
def run_leptonic():
    """Merge hadronic + semi-leptonic processed files into combined training set."""
    HAD_DIR   = Path("data/topquarkreconstruction/masked_targets_combined")
    LEP_DIR   = Path("data/topquarkreconstruction/masked_targets_semileptonic")
    OUTPUT_DIR = Path("data/topquarkreconstruction/leptonic_combined")

    all_keys = BASE_KEYS + LEPTONIC_KEYS

    SPLITS = {
        "train": {
            "hadronic": [HAD_DIR / "ttbar_preprocessed_train.h5"],
            "leptonic": [LEP_DIR / "ttbar_semilepprocessed_train.h5"],
        },
        "val": {
            "hadronic": [HAD_DIR / "ttbar_preprocessed_val.h5"],
            "leptonic": [LEP_DIR / "ttbar_semilepprocessed_val.h5"],
        },
        "test": {
            "hadronic": [HAD_DIR / "ttbar_preprocessed_test.h5"],
            "leptonic": [LEP_DIR / "ttbar_semilepprocessed_test.h5"],
        },
    }

    for split, source_map in SPLITS.items():
        had_paths = [p for p in source_map["hadronic"] if p.exists()]
        lep_paths = [p for p in source_map["leptonic"] if p.exists()]

        if not had_paths and not lep_paths:
            print(f"[SKIP] No source files for split={split}")
            continue

        dst = OUTPUT_DIR / f"ttbar_leptonic_{split}.h5"
        print(f"=== {split}: {len(had_paths)} hadronic + {len(lep_paths)} leptonic files ===")

        all_src = had_paths + lep_paths
        concatenate_to_output(all_src, dst, all_keys, fill_leptonic_defaults=True)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Prepare combined HDF5 datasets")
    parser.add_argument("--mode", choices=["hadronic", "leptonic"], default="hadronic",
                        help="hadronic: concatenate two hadronic datasets; "
                             "leptonic: merge hadronic + semi-leptonic")
    args = parser.parse_args()

    if args.mode == "hadronic":
        print("Mode: hadronic-only combination\n")
        run_hadronic()
    else:
        print("Mode: hadronic + semi-leptonic combination\n")
        run_leptonic()

    print("Next steps:")
    if args.mode == "hadronic":
        print("  Update config/top_reconstruction_config.yaml data_modules.input_path")
    else:
        print("  Run: uv run src/main.py --config config/leptonic_config.yaml")


if __name__ == "__main__":
    main()
