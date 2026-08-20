import json
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np


_SPEC = importlib.util.spec_from_file_location(
    "g2_target_audit", Path(__file__).with_name("target_audit.py")
)
target_audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(target_audit)


def _write_fixture(path: Path):
    # Four events, two chains, six particle slots.
    tops = np.zeros((4, 2, 6), dtype=np.float32)
    ws = np.zeros_like(tops)
    valid_tops = np.ones((4, 2), dtype=np.uint8)
    valid_ws = np.ones((4, 2), dtype=np.uint8)

    # Eligible: both Ws have two jets and each top has one b extension.
    ws[0, 0, [0, 1]] = 1
    ws[0, 1, [3, 4]] = 1
    tops[0, 0, [0, 1, 2]] = 1
    tops[0, 1, [3, 4, 5]] = 1

    # Excluded for W cardinality (three jets in W chain 0).
    ws[1, 0, [0, 1, 2]] = 1
    ws[1, 1, [3, 4]] = 1
    tops[1, 0, [0, 1, 2, 5]] = 1
    tops[1, 1, [3, 4, 5]] = 1

    # Excluded for full-top b-extension cardinality (two b jets in chain 0).
    ws[2, 0, [0, 1]] = 1
    ws[2, 1, [3, 4]] = 1
    tops[2, 0, [0, 1, 2, 5]] = 1
    tops[2, 1, [3, 4, 5]] = 1

    # Invalid targets are allowed to be absent/censored; no valid mask is bad.
    valid_tops[3] = 0
    valid_ws[3] = 0
    ws[3, 0, [0, 1, 2]] = 1

    with h5py.File(path, "w") as handle:
        handle.create_dataset("masks_tops", data=tops)
        handle.create_dataset("masks_Ws", data=ws)
        handle.create_dataset("valid_tops", data=valid_tops)
        handle.create_dataset("valid_Ws", data=valid_ws)
        handle.attrs["source_hash"] = "source"
        handle.attrs["contract_hash"] = "contract"


class _TrackingDataset:
    def __init__(self, dataset, reads):
        self._dataset = dataset
        self._reads = reads

    def __getitem__(self, key):
        self._reads.append(key)
        return self._dataset[key]

    def __getattr__(self, name):
        return getattr(self._dataset, name)


class _TrackingFile:
    def __init__(self, handle, reads):
        self._handle = handle
        self._reads = reads
        self.attrs = handle.attrs

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return self._handle.__exit__(*exc)

    def __contains__(self, key):
        return key in self._handle

    def __getitem__(self, key):
        return _TrackingDataset(self._handle[key], self._reads)


class TargetAuditTests(unittest.TestCase):
    def test_counts_reasons_indices_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.h5"
            _write_fixture(source)

            result = target_audit.audit_file(source, root / "out", "train", chunk_size=2)
            self.assertEqual(result["counts"]["total"], 4)
            self.assertEqual(result["counts"]["eligible"], 2)
            self.assertEqual(result["counts"]["excluded"], 2)
            self.assertEqual(result["exclusion_reasons"]["w_cardinality"], 1)
            self.assertEqual(result["exclusion_reasons"]["full_top_b_cardinality"], 1)
            self.assertEqual(result["provenance"]["source_hash"], "source")
            np.testing.assert_array_equal(
                np.load(root / "out" / "train_eligible_indices.npy"), [0, 3]
            )

    def test_reads_are_bounded_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.h5"
            _write_fixture(source)
            reads = []
            real_file = h5py.File

            def tracking_file(*args, **kwargs):
                return _TrackingFile(real_file(*args, **kwargs), reads)

            with mock.patch.object(target_audit.h5py, "File", tracking_file):
                target_audit.audit_file(source, root / "out", "train", chunk_size=2)

            self.assertTrue(reads)
            self.assertTrue(all(isinstance(key, slice) for key in reads))
            self.assertTrue(all((key.stop - key.start) <= 2 for key in reads))

    def test_cli_writes_aggregate_manifest_and_explicit_stress(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            processed = root / "processed"
            processed.mkdir()
            for split in ("train", "val", "calibration", "test"):
                _write_fixture(processed / f"ttbar_contract_processed_{split}.h5")
            stress = root / "stress.h5"
            _write_fixture(stress)
            output = root / "audit"

            self.assertEqual(target_audit.main([
                "--processed-dir", str(processed), "--stress-file", str(stress),
                "--output-dir", str(output), "--chunk-size", "2",
            ]), 0)
            manifest = json.loads((output / "target_audit_manifest.json").read_text())
            self.assertEqual(set(manifest["splits"]),
                             {"train", "val", "calibration", "test", "stress"})


if __name__ == "__main__":
    unittest.main()
