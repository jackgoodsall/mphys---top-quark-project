"""
Convert interactions dataset in HDF5 files from float32 → float16.

Halves the in-memory footprint (~88 GB → ~44 GB for train) so two DDP
processes fit within the 200 GB SLURM allocation.

All other datasets are copied as-is. Compression and chunk settings are
preserved (chunks are element-indexed so they carry over unchanged).

Usage:
    python src/data/convert_interactions_fp16.py
    python src/data/convert_interactions_fp16.py --dry-run   # print sizes only
"""

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np

DATA_DIR = Path("data/topquarkreconstruction/masked_targets_combined")
CHUNK_ROWS = 50_000  # rows to process at a time; tune to available RAM


def convert_file(src: Path, dry_run: bool = False) -> None:
    with h5py.File(src, "r") as f:
        if "interactions" not in f:
            print(f"  skip {src.name} — no interactions dataset")
            return

        ds = f["interactions"]
        if ds.dtype == np.float16:
            print(f"  skip {src.name} — already float16")
            return

        n_rows = ds.shape[0]
        orig_gb = ds.id.get_storage_size() / 1e9
        est_ram_gb = np.prod(ds.shape) * 4 / 1e9
        est_new_ram_gb = est_ram_gb / 2
        print(f"  {src.name}: {n_rows:,} events, {orig_gb:.1f} GB on disk, "
              f"~{est_ram_gb:.0f} GB RAM (fp32) → ~{est_new_ram_gb:.0f} GB RAM (fp16)")

        if dry_run:
            return

        tmp_path = src.with_suffix(".tmp.h5")
        try:
            with h5py.File(tmp_path, "w") as out:
                # copy all datasets except interactions
                for key in f.keys():
                    if key == "interactions":
                        continue
                    f.copy(key, out)

                # rewrite interactions as float16 in chunks
                ds_out = out.create_dataset(
                    "interactions",
                    shape=ds.shape,
                    dtype=np.float16,
                    chunks=ds.chunks,
                    compression=ds.compression,
                    compression_opts=ds.compression_opts,
                )
                for start in range(0, n_rows, CHUNK_ROWS):
                    end = min(start + CHUNK_ROWS, n_rows)
                    ds_out[start:end] = ds[start:end].astype(np.float16)
                    print(f"    {end:>10,} / {n_rows:,}", end="\r", flush=True)
                print()

                # copy file-level attributes if any
                for attr_name, attr_val in f.attrs.items():
                    out.attrs[attr_name] = attr_val

            new_gb = tmp_path.stat().st_size / 1e9
            print(f"  → {new_gb:.1f} GB on disk (was {orig_gb:.1f} GB)")
            shutil.move(str(tmp_path), str(src))
            print(f"  replaced {src.name}")

        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print sizes without modifying files")
    parser.add_argument("--dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    files = sorted(args.dir.glob("ttbar_preprocessed_*.h5"))
    # skip symlinks — they point to already-handled real files
    files = [f for f in files if not f.is_symlink()]

    if not files:
        print(f"No .h5 files found in {args.dir}")
        return

    for f in files:
        print(f"Processing {f.name} ...")
        convert_file(f, dry_run=args.dry_run)

    print("Done.")


if __name__ == "__main__":
    main()
