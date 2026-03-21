"""
Prepare combined raw HDF5 files for preprocessing.py.

Concatenates old + inclusive raw datasets into:
  - ttbar_h5py_raw_train.h5   (combined, for transformer fitting + training)
  - ttbar_h5py_raw_val.h5     (combined)
  - ttbar_h5py_raw_test_old.h5       (old only, for evaluation)
  - ttbar_h5py_raw_test_inclusive.h5  (inclusive only, for evaluation)

The prepper glob 'ttbar_h5py_raw_*' finds all four files.  It fits
transformers on *train* files only, then transforms everything with
the same fitted scalers.

Usage:
    uv run src/data/prepare_combined_dataset.py
"""

import numpy as np
import h5py
from pathlib import Path

DATA_DIR = Path("data/topquarkreconstruction/h5py_data")
OUTPUT_DIR = Path("data/topquarkreconstruction/h5py_data/combined")
CHUNK_SIZE = 500_000

# Only these keys are used by the preprocessor (it ignores 'targets')
KEYS = ["jet", "event"]


def concatenate_files(src_paths: list, dst_path: Path):
    """Concatenate multiple HDF5 files into one, reading in chunks."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # Gather shapes from the first source that exists
    ref_path = src_paths[0]
    with h5py.File(ref_path, "r") as ref:
        shapes = {k: ref[k].shape[1:] for k in KEYS}
        dtypes = {k: ref[k].dtype for k in KEYS}

    with h5py.File(dst_path, "w") as out:
        # Create resizable datasets
        for key in KEYS:
            out.create_dataset(
                key,
                shape=(0,) + shapes[key],
                maxshape=(None,) + shapes[key],
                dtype=dtypes[key],
                compression="gzip",
                compression_opts=4,
            )

        total = 0
        for src_path in src_paths:
            with h5py.File(src_path, "r") as src:
                n = src[KEYS[0]].shape[0]
                print(f"  {src_path.name}: {n:,} events")

                for start in range(0, n, CHUNK_SIZE):
                    end = min(start + CHUNK_SIZE, n)
                    chunk_len = end - start
                    cur = out[KEYS[0]].shape[0]
                    new_len = cur + chunk_len

                    for key in KEYS:
                        out[key].resize((new_len,) + shapes[key])
                        out[key][cur:new_len] = src[key][start:end]

                    total += chunk_len

        print(f"  -> {dst_path.name}: {total:,} events total\n")


def copy_file(src_path: Path, dst_path: Path):
    """Copy a single HDF5 file, keeping only the keys the prepper needs."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(src_path, "r") as src:
        n = src[KEYS[0]].shape[0]
        print(f"  {src_path.name}: {n:,} events -> {dst_path.name}")

        with h5py.File(dst_path, "w") as out:
            for key in KEYS:
                out.create_dataset(
                    key,
                    data=src[key][:],
                    compression="gzip",
                    compression_opts=4,
                )


def main():
    old_train = DATA_DIR / "ttbar_h5py_raw_train.h5"
    old_val = DATA_DIR / "ttbar_h5py_raw_val.h5"
    old_test = DATA_DIR / "ttbar_h5py_raw_test.h5"
    inc_train = DATA_DIR / "ttbar_h5py_raw_inclusive_train.h5"
    inc_val = DATA_DIR / "ttbar_h5py_raw_inclusive_val.h5"
    inc_test = DATA_DIR / "ttbar_h5py_raw_inclusive_test.h5"

    # Verify all source files exist
    for p in [old_train, old_val, old_test, inc_train, inc_val, inc_test]:
        if not p.exists():
            print(f"[ERROR] Missing: {p}")
            return

    # 1. Combined train (old + inclusive) — for fitting + training
    print("=== Combined train ===")
    concatenate_files(
        [old_train, inc_train],
        OUTPUT_DIR / "ttbar_h5py_raw_train.h5",
    )

    # 2. Combined val
    print("=== Combined val ===")
    concatenate_files(
        [old_val, inc_val],
        OUTPUT_DIR / "ttbar_h5py_raw_val.h5",
    )

    # 3. Separate test files for independent evaluation
    print("=== Test (old) ===")
    copy_file(old_test, OUTPUT_DIR / "ttbar_h5py_raw_test_old.h5")

    print("\n=== Test (inclusive) ===")
    copy_file(inc_test, OUTPUT_DIR / "ttbar_h5py_raw_test_inclusive.h5")

    print("\nDone! Output directory:", OUTPUT_DIR)
    print("\nNext steps:")
    print("  1. Update config to point at the combined directory:")
    print(f'     save_path: "{OUTPUT_DIR}"')
    print("  2. Run src/data/preprocessing.py")
    print("  3. Outputs will include:")
    print("       ttbar_preprocessed_train.h5          (combined, for training)")
    print("       ttbar_preprocessed_val.h5            (combined)")
    print("       ttbar_preprocessed_test_old.h5       (old only, for eval)")
    print("       ttbar_preprocessed_test_inclusive.h5  (inclusive only, for eval)")


if __name__ == "__main__":
    main()
