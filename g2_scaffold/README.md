# Isolated G2 scaffold

This directory is intentionally disconnected from the live G1 model and
configuration. It contains the smallest CPU-testable pieces needed before a
real G2 run:

- three-state (`absent`, `W-only`, `full-top`) candidate scoring;
- permutation-invariant hard-min and marginal losses;
- exact two-chain decoding with distinct W jets, a distinct b jet, and global
  chain disjointness.

Run the checks from the repository root:

```bash
.transformer_env/bin/python -m unittest g2_scaffold.test_scaffold -v
```

No G1 file imports this package; the pilot submission below is the only training
entrypoint and does not resume or share a G1 run directory.

## Parallel pilot

The from-scratch pilot reuses only the existing encoder/data loader and writes
to `g2_scaffold/runs/G2_pilot`:

Build the mandatory G2 eligibility view first. This writes compact row-index
files and does not modify or copy the source HDF5 data:

```bash
.transformer_env/bin/python g2_scaffold/target_audit.py \
  --output-dir data/topquarkreconstruction/contract_v3/g2_eligibility \
  --stress-file data/topquarkreconstruction/contract_v3/processed_stress/ttbar_contract_processed_stress.h5
```

```bash
.transformer_env/bin/python g2_scaffold/train.py \
  --config g2_scaffold/g2_pilot.yaml --cpu-smoke
sbatch g2_scaffold/submit_g2.sbatch g2_scaffold/g2_pilot.yaml
```

The pilot is two epochs on the contract-v3 train/validation splits. Its gate is
finite loss, decreasing train/validation loss, and legal target-free decoding;
it does not depend on a G1 checkpoint.

After the pilot passes, the bounded screening config is
`g2_scaffold/g2_screen.yaml`; it runs 15 epochs in a separate directory.

Evaluate a checkpoint on HYPER test:

```bash
.transformer_env/bin/python g2_scaffold/evaluate.py \
  --config g2_scaffold/g2_pilot.yaml \
  --checkpoint PATH_TO_CHECKPOINT \
  --split test \
  --output-prefix g2_scaffold/runs/G2_pilot/hyper_test
```

Fit temperatures on calibration only, then evaluate the inclusive stress file
without copying it:

```bash
.transformer_env/bin/python g2_scaffold/evaluate.py \
  --config g2_scaffold/g2_pilot.yaml --checkpoint PATH_TO_CHECKPOINT \
  --split calibration --fit-calibration \
  --calibration-output g2_scaffold/runs/G2_pilot/calibration.json

.transformer_env/bin/python g2_scaffold/evaluate.py \
  --config g2_scaffold/g2_pilot.yaml --checkpoint PATH_TO_CHECKPOINT \
  --split test --stress-file data/topquarkreconstruction/contract_v3/processed_stress/ttbar_contract_processed_stress.h5 \
  --state-temperature CALIBRATED_STATE_TEMP \
  --candidate-temperature CALIBRATED_CANDIDATE_TEMP \
  --output-prefix g2_scaffold/runs/G2_pilot/inclusive_stress
```
