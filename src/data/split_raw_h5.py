"""
Split a raw HDF5 file into train/val/test using sequential reads.

Assigns each chunk of events to a split based on pre-shuffled assignment,
reading sequentially to avoid random HDF5 access patterns.

Usage:
    uv run src/data/split_raw_h5.py
"""

import numpy as np
import h5py
from pathlib import Path

INPUT_PATH = Path("data/topquarkreconstruction/h5py_data/ttbar_h5py_raw_train.h5")
OUTPUT_DIR = Path("data/topquarkreconstruction/h5py_data")
PREFIX = "ttbar_h5py_raw_inclusive_"
TRAIN_FRAC = 0.90
VAL_FRAC = 0.05
# test gets the remainder
CHUNK_SIZE = 500_000


def main():
    with h5py.File(INPUT_PATH, "r") as f:
        n_total = f["jet"].shape[0]
        keys = list(f.keys())
        print(f"Total events: {n_total}")
        print(f"Keys: {keys}")

        # Pre-assign every event to a split
        rng = np.random.default_rng(seed=42)
        assignment = rng.random(n_total)  # uniform [0, 1)
        # train: [0, 0.90), val: [0.90, 0.95), test: [0.95, 1.0)

        split_names = ["train", "val", "test"]
        split_files = {}
        split_counts = {"train": 0, "val": 0, "test": 0}

        # Open all output files
        for name in split_names:
            out_path = OUTPUT_DIR / f"{PREFIX}{name}.h5"
            hf = h5py.File(out_path, "w")
            # Create resizable datasets
            for key in keys:
                shape = f[key].shape
                hf.create_dataset(
                    key,
                    shape=(0,) + shape[1:],
                    maxshape=(None,) + shape[1:],
                    dtype=f[key].dtype,
                    compression="gzip",
                    compression_opts=4,
                )
            split_files[name] = hf

        # Read sequentially, distribute to splits
        for start in range(0, n_total, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, n_total)
            chunk_assign = assignment[start:end]

            train_mask = chunk_assign < TRAIN_FRAC
            val_mask = (chunk_assign >= TRAIN_FRAC) & (chunk_assign < TRAIN_FRAC + VAL_FRAC)
            test_mask = chunk_assign >= TRAIN_FRAC + VAL_FRAC

            masks = {"train": train_mask, "val": val_mask, "test": test_mask}

            # Read chunk data once
            chunk_data = {}
            for key in keys:
                chunk_data[key] = f[key][start:end]

            for name, mask in masks.items():
                n_keep = int(mask.sum())
                if n_keep == 0:
                    continue

                hf = split_files[name]
                cur = hf[keys[0]].shape[0]
                new_len = cur + n_keep

                for key in keys:
                    hf[key].resize((new_len,) + hf[key].shape[1:])
                    hf[key][cur:new_len] = chunk_data[key][mask]

                split_counts[name] += n_keep

            print(f"  Processed {end:,} / {n_total:,} | "
                  f"train={split_counts['train']:,} val={split_counts['val']:,} test={split_counts['test']:,}")

        # Close all files
        for name in split_names:
            split_files[name].close()

        print("\nFinal counts:")
        for name, count in split_counts.items():
            print(f"  {name}: {count:,} events")
        print("Done!")


if __name__ == "__main__":
    main()
