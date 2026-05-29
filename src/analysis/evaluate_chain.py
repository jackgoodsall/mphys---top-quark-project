"""
Evaluate top/W/ttbar reconstruction efficiency from saved test outputs
for chain_queries models.

In chain_queries mode there is no object_type task — type is implicit:
  - test_outputs_mask.h5           → top predictions       [N, Q, P]
  - test_outputs_mask_W.h5         → W predictions         [N, Q, P]
  - test_outputs_objectness.h5     → top objectness        [N, Q]
  - test_outputs_objectness_W.h5   → W objectness          [N, Q]

Each query token is a full chain (top + W pair).  A chain is "real" when
the W-mask target has ≥1 assigned particle (matching target_objectness).

Usage:
    python analysis/evaluate_chain.py --run_dir <path/to/version_X>
    python analysis/evaluate_chain.py --run_dir <path/to/version_X> --plot
    python analysis/evaluate_chain.py --run_dir <path/to/version_X> \\
        --data_file <path/to/ttbar_preprocessed_test.h5> --plot

Custom mask threshold (overrides the default 0.0 for logits / 0.5 for probs):
    python analysis/evaluate_chain.py --run_dir ... --threshold 0.3
    python analysis/evaluate_chain.py --run_dir ... --use_probs --threshold 0.6

Custom objectness threshold (real/fake classification; separate from mask threshold):
    python analysis/evaluate_chain.py --run_dir ... --obj_threshold 0.3
    python analysis/evaluate_chain.py --run_dir ... --use_probs --obj_threshold 0.6

Prior-based binarisation (top-k particles per mask type):
    python analysis/evaluate_chain.py --run_dir ... --prior top=3 W=2

Decision boundary / probability distribution plots:
    python analysis/evaluate_chain.py --run_dir ... --prob_dist
    python analysis/evaluate_chain.py --run_dir ... --prob_dist --obj_threshold 0.4
    python analysis/evaluate_chain.py --run_dir ... --prob_dist --prob_dist_bins 120
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _f1(precision: float, recall: float) -> float:
    """F1 score from precision (purity) and recall (efficiency)."""
    denom = precision + recall
    return 2 * precision * recall / denom if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_objectness(path: Path, use_probs: bool, task_name: str = "objectness"):
    """Load target and predicted objectness from an HDF5 file.

    Keys follow the task naming convention:
      target_{task_name}, predicted_{task_name}_logit, predicted_{task_name}_prob
    """
    with h5py.File(path, "r") as f:
        target_key = f"target_{task_name}"
        if target_key not in f:
            sys.exit(f"ERROR: key '{target_key}' not found in {path}")
        target = f[target_key][:]                          # [N, Q]
        pred_key = f"predicted_{task_name}_prob" if use_probs else f"predicted_{task_name}_logit"
        pred = f[pred_key][:] if pred_key in f else None
    return target, pred


def _infer_data_file_from_hparams(run_dir: Path):
    """Best-effort inference of test HDF5 path from run hparams.yaml."""
    hparams = run_dir / "hparams.yaml"
    if not hparams.exists():
        return None
    try:
        with hparams.open("r", encoding="utf-8") as f:
            hp = yaml.safe_load(f)
    except Exception:
        return None
    dmcfg = ((hp or {}).get("config", {}).get("data_modules", {}) or {})
    in_path = dmcfg.get("input_path")
    in_prefix = dmcfg.get("input_prefix")
    if not in_path or not in_prefix:
        return None
    p = Path(in_path) / f"{in_prefix}test.h5"
    return p if p.exists() else None


def load_run_data(run_dir: Path, data_file, use_probs: bool = False):
    """Load all arrays needed for efficiency evaluation."""

    mask_top_path = run_dir / "test_outputs_mask.h5"
    mask_W_path   = run_dir / "test_outputs_mask_W.h5"
    # New models use task name 'objectness_top'; old single-objectness models used 'objectness'
    obj_top_path_new = run_dir / "test_outputs_objectness_top.h5"
    obj_top_path_old = run_dir / "test_outputs_objectness.h5"
    obj_top_path  = obj_top_path_new if obj_top_path_new.exists() else obj_top_path_old
    obj_top_task_name = "objectness_top" if obj_top_path_new.exists() else "objectness"
    obj_W_path    = run_dir / "test_outputs_objectness_W.h5"

    for p in (mask_top_path, mask_W_path):
        if not p.exists():
            sys.exit(f"ERROR: required file not found: {p}")

    # At least one objectness file must exist
    if not obj_top_path.exists() and not obj_W_path.exists():
        sys.exit(
            f"ERROR: no objectness files found. Expected at least one of:\n"
            f"  {obj_top_path_new}\n  {obj_top_path_old}\n  {obj_W_path}"
        )

    scores_key = "predicted_masks_prob" if use_probs else "predicted_masks_logits"

    with h5py.File(mask_top_path, "r") as f:
        if scores_key not in f:
            sys.exit(f"ERROR: key '{scores_key}' not found in {mask_top_path}")
        pred_scores_top = f[scores_key][:]                # [N, Q, P]
        target_masks_top = f["target_masks"][:]            # [N, Q, P]
        jet_valid = f["jet_valid_mask"][:] if "jet_valid_mask" in f else None
        slot_valid_top = f["slot_valid"][:] if "slot_valid" in f else None

    with h5py.File(mask_W_path, "r") as f:
        if scores_key not in f:
            sys.exit(f"ERROR: key '{scores_key}' not found in {mask_W_path}")
        pred_scores_W = f[scores_key][:]                  # [N, Q, P]
        target_masks_W = f["target_masks"][:]              # [N, Q, P]
        slot_valid_W = f["slot_valid"][:] if "slot_valid" in f else None

    # Load separate top / W objectness
    target_obj_top, pred_obj_top = None, None
    target_obj_W,   pred_obj_W   = None, None

    if obj_top_path.exists():
        target_obj_top, pred_obj_top = _load_objectness(obj_top_path, use_probs, task_name=obj_top_task_name)
    if obj_W_path.exists():
        target_obj_W, pred_obj_W = _load_objectness(obj_W_path, use_probs, task_name="objectness_W")

    # If only one file exists, use it for both (backward compat)
    if target_obj_top is None:
        target_obj_top, pred_obj_top = target_obj_W, pred_obj_W
    if target_obj_W is None:
        target_obj_W, pred_obj_W = target_obj_top, pred_obj_top

    # Prefer explicit --data_file, else infer from hparams when possible.
    if data_file is None:
        data_file = _infer_data_file_from_hparams(run_dir)

    valid_tops_truth = None
    valid_Ws_truth = None

    # Fall back to external data file for src_mask, and optionally load truth validity masks.
    if jet_valid is None or data_file is not None:
        if jet_valid is None and data_file is None:
            sys.exit(
                "ERROR: 'jet_valid_mask' not found in test_outputs_mask.h5 and "
                "--data_file was not provided (and could not infer from hparams.yaml).\n"
                "Re-run with --data_file pointing to the corresponding HDF5 data file."
            )
        if data_file is not None and not data_file.exists():
            sys.exit(f"ERROR: data file not found: {data_file}")
        if data_file is not None:
            with h5py.File(data_file, "r") as f:
                if jet_valid is None:
                    if "src_mask" not in f:
                        sys.exit(f"ERROR: 'src_mask' key not found in {data_file}")
                    jet_valid = f["src_mask"][:]
                if "valid_tops" in f:
                    valid_tops_truth = f["valid_tops"][:].astype(bool)
                if "valid_Ws" in f:
                    valid_Ws_truth = f["valid_Ws"][:].astype(bool)
        N_run = pred_scores_top.shape[0]
        N_data = jet_valid.shape[0]
        if N_data != N_run:
            sys.exit(
                f"ERROR: event count mismatch — run has {N_run} events but "
                f"data file has {N_data}. Make sure you are using the correct test split."
            )
        if valid_tops_truth is not None and valid_tops_truth.shape[0] != N_run:
            sys.exit(
                f"ERROR: 'valid_tops' event count mismatch — run has {N_run} events but "
                f"data file has {valid_tops_truth.shape[0]}."
            )
        if valid_Ws_truth is not None and valid_Ws_truth.shape[0] != N_run:
            sys.exit(
                f"ERROR: 'valid_Ws' event count mismatch — run has {N_run} events but "
                f"data file has {valid_Ws_truth.shape[0]}."
            )

    # Load original (pre-signal-jet-filtering) multiplicities if available
    orig_mult_path = run_dir / "event_multiplicities.npz"
    original_mult = None
    if orig_mult_path.exists():
        d = np.load(orig_mult_path)
        original_mult = d["original_mult"].astype(int)

    return (pred_scores_top, target_masks_top, pred_scores_W, target_masks_W,
            jet_valid, target_obj_top, pred_obj_top, target_obj_W, pred_obj_W,
            original_mult, valid_tops_truth, valid_Ws_truth,
            slot_valid_top, slot_valid_W)


# ---------------------------------------------------------------------------
# Prediction binarisation
# ---------------------------------------------------------------------------

def binarise_predictions(scores, jet_valid, prior_k, use_probs, threshold=None):
    """
    Convert raw scores to a binary [N, Q, P] prediction array.

    Default threshold: 0.0 for logits, 0.5 for probs.
    Pass an explicit `threshold` to override the default.

    prior_k: int or None
        If given, select exactly the k highest-scoring *valid* particles
        per slot instead of thresholding.
    """
    N, Q, P = scores.shape
    valid = jet_valid.astype(bool)                        # [N, P]

    if threshold is None:
        threshold = 0.5 if use_probs else 0.0

    if prior_k is not None:
        # Mask padding positions to -inf so they are never top-k selected
        masked_scores = np.where(valid[:, np.newaxis, :], scores, -np.inf)
        topk_idx = np.argsort(masked_scores, axis=-1)[:, :, -prior_k:]  # [N, Q, k]
        pred_bin = np.zeros((N, Q, P), dtype=bool)
        np.put_along_axis(pred_bin, topk_idx, True, axis=-1)
        pred_bin &= valid[:, np.newaxis, :]
    else:
        pred_bin = (scores > threshold).copy()

    return pred_bin


# ---------------------------------------------------------------------------
# Core efficiency computation
# ---------------------------------------------------------------------------

def _compute_breakdown(multiplicity_arr, is_real, top_eff_gate, W_eff_gate, is_perfect_top, is_perfect_W,
                       all_tops_perfect, all_Ws_perfect, pred_real,
                       pred_real_W=None, is_real_W=None):
    """Compute per-multiplicity efficiency/purity rows for a given multiplicity array."""
    mult_values = sorted(np.unique(multiplicity_arr).tolist())
    breakdown = {}
    for m in mult_values:
        sel = multiplicity_arr == m
        n_sel = sel.sum()
        if n_sel == 0:
            continue

        ir  = is_real[sel]
        teg = top_eff_gate[sel]
        ipt = is_perfect_top[sel]
        ipw = is_perfect_W[sel]

        both_tops = ir.sum(axis=1) == 2

        n_top_correct = int((teg & ipt).sum())
        n_top_total   = int(teg.sum())
        n_W_correct   = int((W_eff_gate[sel] & ipw).sum())
        n_W_total     = int(W_eff_gate[sel].sum())
        bt = int(both_tops.sum())

        row = {
            "n_events":  int(n_sel),
            "top_eff":   float(n_top_correct / max(n_top_total, 1)),
            "W_eff":     float(n_W_correct / max(n_W_total, 1)),
            "ttbar_eff": float((all_tops_perfect[sel] & both_tops).sum() / max(bt, 1)),
            "n_top":     n_top_total,
            "n_W":       n_W_total,
            "n_ttbar":   bt,
        }

        if pred_real is not None:
            pr_top = pred_real[sel]                                                       # [n_sel, Q]
            pr_W   = pred_real_W[sel] if pred_real_W is not None else pr_top             # [n_sel, Q]
            irW    = is_real_W[sel]   if is_real_W   is not None else ir                 # [n_sel, Q]

            n_pr_top  = int(pr_top.sum())
            n_pr_W    = int(pr_W.sum())

            row["top_purity"]       = float((pr_top & ir  & ipt).sum() / max(n_pr_top, 1))
            row["W_purity"]         = float((pr_W   & irW & ipw).sum() / max(n_pr_W,   1))

            pr_both   = pr_top.sum(axis=1) >= 2
            n_pr_both = int(pr_both.sum())
            both_ok   = (pr_top & ir & ipt).sum(axis=1) == 2
            row["ttbar_purity"]     = float((pr_both & both_ok).sum() / max(n_pr_both, 1))
            row["n_pred_real"]      = n_pr_top
            row["n_pred_real_W"]    = n_pr_W
            row["n_pred_both_tops"] = n_pr_both
            row["top_f1"]   = _f1(row["top_purity"],   row["top_eff"])
            row["W_f1"]     = _f1(row["W_purity"],     row["W_eff"])
            row["ttbar_f1"] = _f1(row["ttbar_purity"], row["ttbar_eff"])

        breakdown[m] = row
    return breakdown


def compute_efficiencies(pred_scores_top, target_masks_top, pred_scores_W, target_masks_W,
                         jet_valid, target_obj_top, pred_obj_top=None,
                         target_obj_W=None, pred_obj_W=None,
                         prior_top=None, prior_W=None, use_probs=False,
                         threshold=None, obj_threshold=None, strict=False, joint=False,
                         original_mult=None, valid_tops_truth=None, valid_Ws_truth=None,
                         slot_valid_top=None, slot_valid_W=None,
                         legacy_output_targets=False,
                         require_complete_truth=False,
                         require_top_for_w=False):
    """
    Returns a dict with scalar efficiencies and per-multiplicity breakdowns.

    Chain queries: each query predicts both a top mask and a W mask.

    Per-type efficiency gates:
      - slot_valid_top / slot_valid_W from the output files (post-matching order,
        derived from the per-type valid_tops / valid_Ws before the chain AND).
      - Fallback: chain-level obj_valid_mask (both top & W valid).
    """
    N, Q, P = pred_scores_top.shape

    # Per-type slot validity from output files (preferred)
    if slot_valid_top is not None:
        is_real_top = slot_valid_top.astype(bool)
    else:
        # Fallback: chain-level objectness (both top & W valid)
        is_real_top = target_obj_top.astype(bool) if target_obj_top is not None else \
                      target_masks_top.astype(bool).any(axis=-1)

    if slot_valid_W is not None:
        is_real_W = slot_valid_W.astype(bool)
    else:
        is_real_W = target_obj_W.astype(bool) if target_obj_W is not None else \
                    target_masks_W.astype(bool).any(axis=-1)

    # Fallback for legacy/backward compat
    if legacy_output_targets:
        is_real = target_masks_W.astype(bool).any(axis=-1)
        is_real_top = is_real
        is_real_W = is_real

    # Optional physics-complete truth gate: require full expected multiplicity
    # for a real object (top=3 particles, W=2 particles).
    if require_complete_truth and not legacy_output_targets:
        top_counts = target_masks_top.astype(bool).sum(axis=-1)
        W_counts = target_masks_W.astype(bool).sum(axis=-1)
        is_real_top = is_real_top & (top_counts == 3)
        is_real_W = is_real_W & (W_counts == 2)

    # Optional physics-consistent gate: a W is only real if its parent top is also real.
    if require_top_for_w:
        is_real_W = is_real_W & is_real_top

    n_real_top = is_real_top.sum(axis=1)                   # [N]

    valid = jet_valid[:, np.newaxis, :].astype(bool)       # [N, 1, P]

    # Binarise top and W masks separately
    pred_bin_top = binarise_predictions(
        pred_scores_top, jet_valid, prior_top, use_probs, threshold=threshold
    )
    pred_bin_W = binarise_predictions(
        pred_scores_W, jet_valid, prior_W, use_probs, threshold=threshold
    )

    # Slot-level correctness
    target_top_b = target_masks_top.astype(bool)
    target_W_b   = target_masks_W.astype(bool)

    mismatch_top     = (pred_bin_top != target_top_b) & valid
    mismatch_W       = (pred_bin_W   != target_W_b)   & valid
    slot_perfect_top = mismatch_top.sum(axis=2) == 0       # [N, Q]
    slot_perfect_W   = mismatch_W.sum(axis=2) == 0         # [N, Q]

    # In joint mode, top is only correct if its associated W is also correct.
    # W condition is vacuously satisfied for chains with no real W target.
    if joint:
        slot_perfect_top = slot_perfect_top & (slot_perfect_W | ~is_real_W)

    # In strict mode, require the corresponding objectness head to predict real
    obj_thresh = obj_threshold if obj_threshold is not None else (0.5 if use_probs else 0.0)
    pred_real_top = None
    pred_real_W   = None
    if strict and pred_obj_top is not None:
        pred_real_top = pred_obj_top > obj_thresh
        detected_top = slot_perfect_top & pred_real_top
    else:
        detected_top = slot_perfect_top
    if strict and pred_obj_W is not None:
        pred_real_W = pred_obj_W > obj_thresh
        detected_W = slot_perfect_W & pred_real_W
    else:
        detected_W = slot_perfect_W

    # Event-level aggregates — use is_real (W-based) as the chain gate throughout
    all_tops_perfect = ((~is_real_top) | detected_top).all(axis=1)   # [N]
    all_Ws_perfect   = ((~is_real_W) | detected_W).all(axis=1)       # [N]
    perfect_all      = ((~is_real_top) | (detected_top & detected_W)).all(axis=1)

    both_tops = n_real_top == 2

    # Per-chain efficiencies use all real chains in the denominator.
    # Priors affect prediction binarisation only (numerator), not denominator.
    top_eff_gate = is_real_top
    W_eff_gate   = is_real_W

    n_top_correct = (top_eff_gate & detected_top).sum()
    n_top_total   = top_eff_gate.sum()

    n_W_correct = (W_eff_gate & detected_W).sum()
    n_W_total   = W_eff_gate.sum()

    # W efficiency: per-chain (real W chains)
    # ttbar efficiency: event-level — both tops correct among events with exactly 2 real chains
    top_eff   = n_top_correct / max(n_top_total, 1)
    W_eff     = n_W_correct / max(n_W_total, 1)
    ttbar_eff = (all_tops_perfect & both_tops).sum() / max(both_tops.sum(), 1)
    all_eff   = perfect_all.sum()                    / len(perfect_all)

    # ── Purity (from objectness predictions) ──
    # Use top objectness for top purity, W objectness for W purity.
    # "obj_purity" (chain-level) uses top objectness as the primary indicator.
    obj_purity       = None
    ttbar_purity     = None
    n_pred_real_top  = None
    n_pred_real_W    = None
    top_purity       = None
    W_purity         = None
    n_pred_both_tops = None

    if pred_obj_top is not None:
        pred_real_top = pred_obj_top > obj_thresh  # obj_thresh already set above  # [N, Q]
        n_pred_real_top = int(pred_real_top.sum())

        # Of predicted-real-top chains, fraction that are actually real
        obj_purity = float((pred_real_top & is_real_top).sum() / max(n_pred_real_top, 1))

        # Top purity: of predicted-real-top chains, fraction with perfect top mask
        top_purity = float((pred_real_top & is_real_top & slot_perfect_top).sum()
                           / max(n_pred_real_top, 1))

        # ttbar purity: events with ≥2 chains predicted real (top), both tops correct
        pred_both_tops   = pred_real_top.sum(axis=1) >= 2
        n_pred_both_tops = int(pred_both_tops.sum())
        both_tops_correct = (pred_real_top & is_real_top & slot_perfect_top).sum(axis=1) == 2
        ttbar_purity = float((pred_both_tops & both_tops_correct).sum()
                             / max(n_pred_both_tops, 1))

    if pred_obj_W is not None:
        pred_real_W = pred_obj_W > obj_thresh                  # [N, Q]
        n_pred_real_W = int(pred_real_W.sum())

        # W purity: of predicted-real-W chains, fraction with perfect W mask
        W_purity = float((pred_real_W & is_real_W & slot_perfect_W).sum()
                         / max(n_pred_real_W, 1))

    # For breakdown purity, use top objectness as the gating prediction
    pred_real_for_breakdown = pred_real_top

    # Per-multiplicity breakdown (by signal jet count)
    multiplicity = jet_valid.sum(axis=1).astype(int)
    breakdown = _compute_breakdown(
        multiplicity, is_real_top, top_eff_gate, W_eff_gate, detected_top, detected_W,
        all_tops_perfect, all_Ws_perfect, pred_real_for_breakdown,
        pred_real_W=pred_real_W, is_real_W=is_real_W,
    )

    # Per-original-multiplicity breakdown
    breakdown_orig = {}
    if original_mult is not None:
        breakdown_orig = _compute_breakdown(
            original_mult, is_real_top, top_eff_gate, W_eff_gate, detected_top, detected_W,
            all_tops_perfect, all_Ws_perfect, pred_real_for_breakdown,
            pred_real_W=pred_real_W, is_real_W=is_real_W,
        )

    top_f1   = _f1(top_purity,   float(top_eff))   if top_purity   is not None else None
    W_f1     = _f1(W_purity,     float(W_eff))     if W_purity     is not None else None
    ttbar_f1 = _f1(ttbar_purity, float(ttbar_eff)) if ttbar_purity is not None else None

    return {
        "N": N,
        "Q": Q,
        "P": P,
        "top_eff":   float(top_eff),
        "W_eff":     float(W_eff),
        "ttbar_eff": float(ttbar_eff),
        "all_eff":   float(all_eff),
        "obj_purity":       obj_purity,
        "ttbar_purity":     ttbar_purity,
        "top_purity":       top_purity,
        "W_purity":         W_purity,
        "top_f1":           top_f1,
        "W_f1":             W_f1,
        "ttbar_f1":         ttbar_f1,
        "n_pred_real_top":  n_pred_real_top,
        "n_pred_real_W":    n_pred_real_W,
        "n_pred_both_tops": n_pred_both_tops,
        "n_has_top":   int(n_top_total),   # chains with non-empty top target
        "n_has_W":     int(n_W_total),
        "n_both_tops": int(both_tops.sum()),
        "breakdown":      breakdown,
        "breakdown_orig": breakdown_orig,
        "prior_top":   prior_top,
        "prior_W":     prior_W,
        "use_probs":   use_probs,
        "threshold":     threshold if threshold is not None else (0.5 if use_probs else 0.0),
        "obj_threshold": obj_thresh,
        "strict":      strict,
        "joint":       joint,
    }


# ---------------------------------------------------------------------------
# Text output
# ---------------------------------------------------------------------------

def print_results(run_dir: Path, results: dict):
    N  = results["N"]
    Q  = results["Q"]
    P  = results["P"]
    br = results["breakdown"]

    prior_parts = []
    if results["prior_top"] is not None:
        prior_parts.append(f"top={results['prior_top']}")
    if results["prior_W"] is not None:
        prior_parts.append(f"W={results['prior_W']}")
    prior_str = ", ".join(prior_parts) if prior_parts else f"threshold={results['threshold']}"
    scores_str = "probs" if results["use_probs"] else "logits"

    print(f"\n=== Evaluation (chain_queries): {run_dir} ===")
    print(f"Events: {N:,}  |  Query slots Q: {Q}  |  Particles P: {P}")
    print(f"Scores: {scores_str}  |  Binarisation: {prior_str}\n")

    mode_parts = []
    if results.get("strict"):
        mode_parts.append("strict: pred_real & correct mask / N_real")
    else:
        mode_parts.append("recall: N_correct / N_real")
    if results.get("joint"):
        mode_parts.append("joint: top correct only if W also correct")
    print(f"EFFICIENCY SUMMARY ({'; '.join(mode_parts)})")
    print("─" * 70)
    top_label = "top+W masks correct" if results.get("joint") else "top mask correct"
    print(f"  Top efficiency     (per chain, {top_label}):  {results['top_eff']*100:6.2f}%   (N={results['n_has_top']:,} chains)")
    print(f"  W efficiency       (per chain, W mask correct):         {results['W_eff']*100:6.2f}%   (N={results['n_has_W']:,} chains)")
    print(f"  ttbar efficiency   (exactly 2 chains, both correct):   {results['ttbar_eff']*100:6.2f}%   (N={results['n_both_tops']:,})")
    print(f"  N_real used        (top / W / ttbar):                  {results['n_has_top']:,} / {results['n_has_W']:,} / {results['n_both_tops']:,}")
    print("─" * 70)
    print(f"  All-object efficiency:                                  {results['all_eff']*100:6.2f}%   (N={N:,})")

    has_top_purity = results.get("obj_purity") is not None
    has_W_purity   = results.get("W_purity") is not None
    if has_top_purity or has_W_purity:
        print(f"\nPURITY / F1 SUMMARY  (purity = precision, efficiency = recall)")
        print("─" * 80)
        if has_top_purity:
            npt = results['n_pred_real_top']
            print(f"  Object purity      (pred real & actual real / pred real):       {results['obj_purity']*100:5.2f}%   (N_pred_real_top={npt:,})")
            print(f"  Top purity         (pred real & perfect top / pred real):       {results['top_purity']*100:5.2f}%   (N_pred_real_top={npt:,})")
            print(f"  ttbar purity       (both tops correct / pred >=2 real):         {results['ttbar_purity']*100:5.2f}%   (N_pred_2t={results['n_pred_both_tops']:,})")
        if has_W_purity:
            npw = results['n_pred_real_W']
            print(f"  W purity           (pred real & perfect W / pred real):         {results['W_purity']*100:5.2f}%   (N_pred_real_W={npw:,})")
        print("─" * 80)
        if has_top_purity:
            print(f"  Top F1             (2·purity·eff / (purity+eff)):               {results['top_f1']*100:5.2f}%")
            print(f"  ttbar F1           :                                             {results['ttbar_f1']*100:5.2f}%")
        if has_W_purity:
            print(f"  W F1               :                                             {results['W_f1']*100:5.2f}%")
        print("─" * 80)

    def _print_mult_table(label, table):
        if not table:
            return
        has_f1 = "top_f1" in next(iter(table.values()))
        print(f"\n{label}")
        if has_f1:
            header = (
                f"  {'Jets':>5}   {'Top eff':>8}   {'W eff':>8}   {'ttbar eff':>9}   "
                f"{'Top F1':>8}   {'W F1':>8}   {'ttbar F1':>9}   {'Events':>8}"
            )
        else:
            header = (
                f"  {'Jets':>5}   {'Top eff':>8}   {'W eff':>8}   {'ttbar eff':>9}   "
                f"{'N_top':>8}   {'N_W':>8}   {'N_ttbar':>8}   {'Events':>8}"
            )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for m, row in sorted(table.items()):
            if has_f1:
                print(
                    f"  {m:>5}   {row['top_eff']*100:>7.2f}%   "
                    f"{row['W_eff']*100:>7.2f}%   "
                    f"{row['ttbar_eff']*100:>8.2f}%   "
                    f"{row['top_f1']*100:>7.2f}%   "
                    f"{row['W_f1']*100:>7.2f}%   "
                    f"{row['ttbar_f1']*100:>8.2f}%   "
                    f"{row['n_events']:>8,}"
                )
            else:
                print(
                    f"  {m:>5}   {row['top_eff']*100:>7.2f}%   "
                    f"{row['W_eff']*100:>7.2f}%   "
                    f"{row['ttbar_eff']*100:>8.2f}%   "
                    f"{row['n_top']:>8,}   "
                    f"{row['n_W']:>8,}   "
                    f"{row['n_ttbar']:>8,}   "
                    f"{row['n_events']:>8,}"
                )

    _print_mult_table("EFFICIENCY BY SIGNAL JET MULTIPLICITY (jets fed to model)", br)
    _print_mult_table("EFFICIENCY BY ORIGINAL JET MULTIPLICITY (all jets in event)", results.get("breakdown_orig", {}))
    print()


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _agg_bin(br: dict, mult_keys):
    """Aggregate efficiency across a set of multiplicity keys using weighted average."""
    n_top = sum(br[m]["n_top"]   for m in mult_keys if m in br)
    n_W   = sum(br[m]["n_W"]     for m in mult_keys if m in br)
    n_tt  = sum(br[m]["n_ttbar"] for m in mult_keys if m in br)
    top_eff   = sum(br[m]["top_eff"]   * br[m]["n_top"]   for m in mult_keys if m in br) / max(n_top, 1)
    W_eff     = sum(br[m]["W_eff"]     * br[m]["n_W"]     for m in mult_keys if m in br) / max(n_W,   1)
    ttbar_eff = sum(br[m]["ttbar_eff"] * br[m]["n_ttbar"] for m in mult_keys if m in br) / max(n_tt,  1)
    n_events  = sum(br[m]["n_events"]  for m in mult_keys if m in br)
    return {"top_eff": top_eff, "W_eff": W_eff, "ttbar_eff": ttbar_eff, "n_events": n_events,
            "n_top": n_top, "n_W": n_W, "n_ttbar": n_tt}


def _agg_purity_bin(br: dict, mult_keys):
    """Aggregate purity across a set of multiplicity keys using weighted average."""
    n_pr      = sum(br[m]["n_pred_real"]                                    for m in mult_keys if m in br)
    n_pr_W    = sum(br[m].get("n_pred_real_W", br[m]["n_pred_real"])        for m in mult_keys if m in br)
    n_pr_both = sum(br[m]["n_pred_both_tops"]                               for m in mult_keys if m in br)
    top   = sum(br[m]["top_purity"]   * br[m]["n_pred_real"]                                    for m in mult_keys if m in br) / max(n_pr,      1)
    W     = sum(br[m]["W_purity"]     * br[m].get("n_pred_real_W", br[m]["n_pred_real"])        for m in mult_keys if m in br) / max(n_pr_W,    1)
    ttbar = sum(br[m]["ttbar_purity"] * br[m]["n_pred_both_tops"]                               for m in mult_keys if m in br) / max(n_pr_both, 1)
    n_events = sum(br[m]["n_events"] for m in mult_keys if m in br)
    return {"top_purity": top, "W_purity": W, "ttbar_purity": ttbar,
            "n_pred_both_tops": n_pr_both, "n_events": n_events}


def make_plots(run_dir: Path, results: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not available — skipping plots.")
        return

    br = results["breakdown"]

    # 1. Grouped bar chart: Top / W / ttbar efficiency in bins 6, 7, >=8, All
    all_mults = sorted(br.keys())
    ge8_keys  = [m for m in all_mults if m >= 8]

    bins = {
        "6 jets":  _agg_bin(br, [6]),
        "7 jets":  _agg_bin(br, [7]),
        "\u22658 jets": _agg_bin(br, ge8_keys),
        "All":     {
            "top_eff":   results["top_eff"],
            "W_eff":     results["W_eff"],
            "ttbar_eff": results["ttbar_eff"],
            "n_events":  results["N"],
        },
    }

    bin_labels    = list(bins.keys())
    metric_labels = ["Top", "W", "ttbar"]
    metric_keys   = ["top_eff", "W_eff", "ttbar_eff"]
    colors        = ["#4c72b0", "#dd8452", "#55a868"]

    n_bins    = len(bin_labels)
    n_metrics = len(metric_labels)
    bar_width = 0.22
    x = np.arange(n_bins)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (metric, key, color) in enumerate(zip(metric_labels, metric_keys, colors)):
        offsets = x + (i - (n_metrics - 1) / 2) * bar_width
        vals    = [bins[b][key] * 100 for b in bin_labels]
        bars    = ax.bar(offsets, vals, width=bar_width, label=metric, color=color)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, val + 0.8,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=7, rotation=90,
            )

    bin_counts = [f"{bins[b]['n_events']:,}" for b in bin_labels]
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lbl}\n(N={n})" for lbl, n in zip(bin_labels, bin_counts)])
    ax.set_ylabel("Efficiency (%)")
    ax.set_title("Chain Queries: Reconstruction Efficiency by Jet Multiplicity Bin")
    ax.set_ylim(0, 115)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = run_dir / "eval_efficiency_summary.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")

    def _plot_eff_vs_mult(ax, table, xlabel, title):
        mults      = sorted(table.keys())
        xs         = np.array(mults)
        top_effs   = [table[m]["top_eff"]   * 100 for m in mults]
        W_effs     = [table[m]["W_eff"]     * 100 for m in mults]
        ttbar_effs = [table[m]["ttbar_eff"] * 100 for m in mults]
        ax.plot(xs, top_effs,   "o-", label=r"$\varepsilon_t$",              color="#4c72b0", lw=2, ms=8)
        ax.plot(xs, W_effs,     "s-", label=r"$\varepsilon_W$",              color="#dd8452", lw=2, ms=8)
        ax.plot(xs, ttbar_effs, "^-", label=r"$\varepsilon_{t\bar{t}}$",     color="#55a868", lw=2, ms=8)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(m) for m in mults], fontsize=16)
        ax.set_xlabel(xlabel, fontsize=17)
        ax.set_ylabel("Efficiency (%)", fontsize=17)
        ax.tick_params(axis="y", labelsize=16)
        ax.legend(fontsize=18, loc="upper right")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)

    # 2. Efficiency vs multiplicity
    br_orig = results.get("breakdown_orig", {})
    if br or br_orig:
        n_panels = (1 if br else 0) + (1 if br_orig else 0)
        fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 5), squeeze=False)
        panel = 0
        if br:
            _plot_eff_vs_mult(axes[0, panel], br,
                              "Jet multiplicity",
                              "Chain Queries: Efficiency vs Signal Jet Multiplicity")
            panel += 1
        if br_orig:
            _plot_eff_vs_mult(axes[0, panel], br_orig,
                              "Jet multiplicity (original)",
                              "Chain Queries: Efficiency vs Original Jet Multiplicity")
        fig.tight_layout()
        out = run_dir / "eval_efficiency_vs_multiplicity.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out}")

    # ── Purity plots (only when objectness predictions are available) ──
    has_purity = br and "ttbar_purity" in next(iter(br.values()))
    if has_purity:
        # 3. Grouped bar chart: Top / W / ttbar purity in bins 6, 7, >=8, All
        purity_bins = {
            "6 jets":  _agg_purity_bin(br, [6]),
            "7 jets":  _agg_purity_bin(br, [7]),
            "\u22658 jets": _agg_purity_bin(br, ge8_keys),
            "All":     {
                "top_purity":       results["top_purity"],
                "W_purity":         results["W_purity"],
                "ttbar_purity":     results["ttbar_purity"],
                "n_pred_both_tops": results["n_pred_both_tops"],
                "n_events":         results["N"],
            },
        }

        p_bin_labels    = list(purity_bins.keys())
        p_metric_labels = ["Top", "W", "ttbar"]
        p_metric_keys   = ["top_purity", "W_purity", "ttbar_purity"]
        p_colors        = ["#4c72b0", "#dd8452", "#55a868"]

        n_p_bins    = len(p_bin_labels)
        n_p_metrics = len(p_metric_labels)
        p_bar_width = 0.22
        xp = np.arange(n_p_bins)

        fig, ax = plt.subplots(figsize=(8, 5))
        for i, (metric, key, color) in enumerate(zip(p_metric_labels, p_metric_keys, p_colors)):
            offsets = xp + (i - (n_p_metrics - 1) / 2) * p_bar_width
            vals    = [purity_bins[b][key] * 100 for b in p_bin_labels]
            bars    = ax.bar(offsets, vals, width=p_bar_width, label=metric, color=color)
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, val + 0.8,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=7, rotation=90,
                )

        p_bin_counts = [f"{purity_bins[b]['n_events']:,}" for b in p_bin_labels]
        ax.set_xticks(xp)
        ax.set_xticklabels([f"{lbl}\n(N={n})" for lbl, n in zip(p_bin_labels, p_bin_counts)])
        ax.set_ylabel("Purity (%)")
        ax.set_title("Chain Queries: Reconstruction Purity by Jet Multiplicity Bin")
        ax.set_ylim(0, 115)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        out = run_dir / "eval_purity_summary.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out}")

        # 4. Purity vs multiplicity
        mults = sorted(br.keys())
        top_purities   = [br[m]["top_purity"]   * 100 for m in mults]
        W_purities     = [br[m]["W_purity"]     * 100 for m in mults]
        ttbar_purities = [br[m]["ttbar_purity"] * 100 for m in mults]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(mults, top_purities,   "o-", label="Top purity",   color="#4c72b0")
        ax.plot(mults, W_purities,     "s-", label="W purity",     color="#dd8452")
        ax.plot(mults, ttbar_purities, "^-", label="ttbar purity", color="#55a868")
        ax.set_xlabel("Number of valid jets (multiplicity)")
        ax.set_ylabel("Purity (%)")
        ax.set_title("Chain Queries: Reconstruction Purity vs Jet Multiplicity")
        ax.legend()
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = run_dir / "eval_purity_vs_multiplicity.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out}")

        # 5. F1 summary bar chart (per jet-multiplicity bin)
        f1_bins = {
            "6 jets":      {
                "top_f1":   _f1(_agg_purity_bin(br, [6])["top_purity"],   _agg_bin(br, [6])["top_eff"]),
                "W_f1":     _f1(_agg_purity_bin(br, [6])["W_purity"],     _agg_bin(br, [6])["W_eff"]),
                "ttbar_f1": _f1(_agg_purity_bin(br, [6])["ttbar_purity"], _agg_bin(br, [6])["ttbar_eff"]),
                "n_events": _agg_bin(br, [6])["n_events"],
            },
            "7 jets":      {
                "top_f1":   _f1(_agg_purity_bin(br, [7])["top_purity"],   _agg_bin(br, [7])["top_eff"]),
                "W_f1":     _f1(_agg_purity_bin(br, [7])["W_purity"],     _agg_bin(br, [7])["W_eff"]),
                "ttbar_f1": _f1(_agg_purity_bin(br, [7])["ttbar_purity"], _agg_bin(br, [7])["ttbar_eff"]),
                "n_events": _agg_bin(br, [7])["n_events"],
            },
            "\u22658 jets": {
                "top_f1":   _f1(_agg_purity_bin(br, ge8_keys)["top_purity"],   _agg_bin(br, ge8_keys)["top_eff"]),
                "W_f1":     _f1(_agg_purity_bin(br, ge8_keys)["W_purity"],     _agg_bin(br, ge8_keys)["W_eff"]),
                "ttbar_f1": _f1(_agg_purity_bin(br, ge8_keys)["ttbar_purity"], _agg_bin(br, ge8_keys)["ttbar_eff"]),
                "n_events": _agg_bin(br, ge8_keys)["n_events"],
            },
            "All":         {
                "top_f1":   results["top_f1"],
                "W_f1":     results["W_f1"],
                "ttbar_f1": results["ttbar_f1"],
                "n_events": results["N"],
            },
        }

        f1_bin_labels    = list(f1_bins.keys())
        f1_metric_labels = ["Top", "W", "ttbar"]
        f1_metric_keys   = ["top_f1", "W_f1", "ttbar_f1"]
        f1_colors        = ["#4c72b0", "#dd8452", "#55a868"]

        n_f1_bins    = len(f1_bin_labels)
        n_f1_metrics = len(f1_metric_labels)
        f1_bar_width = 0.22
        xf = np.arange(n_f1_bins)

        fig, ax = plt.subplots(figsize=(8, 5))
        for i, (metric, key, color) in enumerate(zip(f1_metric_labels, f1_metric_keys, f1_colors)):
            offsets = xf + (i - (n_f1_metrics - 1) / 2) * f1_bar_width
            vals    = [f1_bins[b][key] * 100 for b in f1_bin_labels]
            bars    = ax.bar(offsets, vals, width=f1_bar_width, label=metric, color=color)
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, val + 0.8,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=7, rotation=90,
                )

        f1_bin_counts = [f"{f1_bins[b]['n_events']:,}" for b in f1_bin_labels]
        ax.set_xticks(xf)
        ax.set_xticklabels([f"{lbl}\n(N={n})" for lbl, n in zip(f1_bin_labels, f1_bin_counts)])
        ax.set_ylabel("F1 Score (%)")
        ax.set_title("Chain Queries: Reconstruction F1 Score by Jet Multiplicity Bin")
        ax.set_ylim(0, 115)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        out = run_dir / "eval_f1_summary.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out}")

        # 6. F1 vs multiplicity line plot
        top_f1s   = [br[m]["top_f1"]   * 100 for m in mults]
        W_f1s     = [br[m]["W_f1"]     * 100 for m in mults]
        ttbar_f1s = [br[m]["ttbar_f1"] * 100 for m in mults]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(mults, top_f1s,   "o-", label="Top F1",   color="#4c72b0")
        ax.plot(mults, W_f1s,     "s-", label="W F1",     color="#dd8452")
        ax.plot(mults, ttbar_f1s, "^-", label="ttbar F1", color="#55a868")
        ax.set_xlabel("Number of valid jets (multiplicity)")
        ax.set_ylabel("F1 Score (%)")
        ax.set_title("Chain Queries: Reconstruction F1 vs Jet Multiplicity")
        ax.legend()
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = run_dir / "eval_f1_vs_multiplicity.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_csv(run_dir: Path, results: dict):
    """
    Write ttbar efficiency, purity, and F1 for the four standard bins
    (6 jets, 7 jets, ≥8 jets, All) to eval_ttbar_summary.csv.
    """
    import csv

    br = results["breakdown"]
    all_mults = sorted(br.keys())
    ge8_keys  = [m for m in all_mults if m >= 8]

    has_purity = br and "ttbar_purity" in next(iter(br.values()))

    def _row(bin_label, eff, purity, f1, n_events, n_ttbar):
        row = {
            "bin":          bin_label,
            "n_events":     n_events,
            "n_ttbar":      n_ttbar,
            "ttbar_eff":    f"{eff:.6f}",
        }
        if has_purity:
            row["ttbar_purity"] = f"{purity:.6f}"
            row["ttbar_f1"]     = f"{f1:.6f}"
        return row

    agg6   = _agg_bin(br, [6])
    agg7   = _agg_bin(br, [7])
    agg8p  = _agg_bin(br, ge8_keys)

    if has_purity:
        pur6   = _agg_purity_bin(br, [6])
        pur7   = _agg_purity_bin(br, [7])
        pur8p  = _agg_purity_bin(br, ge8_keys)
        rows = [
            _row("6 jets",   agg6["ttbar_eff"],   pur6["ttbar_purity"],
                 _f1(pur6["ttbar_purity"],  agg6["ttbar_eff"]),  agg6["n_events"],  agg6["n_ttbar"]),
            _row("7 jets",   agg7["ttbar_eff"],   pur7["ttbar_purity"],
                 _f1(pur7["ttbar_purity"],  agg7["ttbar_eff"]),  agg7["n_events"],  agg7["n_ttbar"]),
            _row(">=8 jets", agg8p["ttbar_eff"],  pur8p["ttbar_purity"],
                 _f1(pur8p["ttbar_purity"], agg8p["ttbar_eff"]), agg8p["n_events"], agg8p["n_ttbar"]),
            _row("All",      results["ttbar_eff"], results["ttbar_purity"],
                 results["ttbar_f1"],               results["N"],        results["n_both_tops"]),
        ]
    else:
        rows = [
            _row("6 jets",   agg6["ttbar_eff"],   None, None, agg6["n_events"],  agg6["n_ttbar"]),
            _row("7 jets",   agg7["ttbar_eff"],   None, None, agg7["n_events"],  agg7["n_ttbar"]),
            _row(">=8 jets", agg8p["ttbar_eff"],  None, None, agg8p["n_events"], agg8p["n_ttbar"]),
            _row("All",      results["ttbar_eff"], None, None, results["N"],      results["n_both_tops"]),
        ]

    fieldnames = list(rows[0].keys())
    out = run_dir / "eval_ttbar_summary.csv"
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Objectness threshold sweep
# ---------------------------------------------------------------------------

def sweep_obj_threshold(pred_scores_top, target_masks_top, pred_scores_W, target_masks_W,
                        jet_valid, pred_obj_top, pred_obj_W=None,
                        use_probs=False, threshold=None, prior_top=None, prior_W=None,
                        slot_valid_top=None, slot_valid_W=None,
                        legacy_output_targets=False,
                        require_complete_truth=False, require_top_for_w=False,
                        joint=False,
                        t_min=None, t_max=None, n_steps=100):
    """
    Sweep the objectness threshold in probability space and compute purity,
    efficiency, and F1 at each step.

    Objectness scores are always converted to probabilities via sigmoid before
    sweeping, so the threshold is always in [0, 1] regardless of --use_probs.
    Mask binarisation is fixed (uses `threshold` / `prior_top` / `prior_W`).

    Returns a dict of arrays, each of length n_steps:
        thresholds, top_eff, top_purity, top_f1,
        W_eff, W_purity, W_f1, ttbar_eff, ttbar_purity, ttbar_f1
    """
    if t_min is None:
        t_min = 0.0
    if t_max is None:
        t_max = 1.0

    obj_thresholds = np.linspace(t_min, t_max, n_steps)

    # Always work in probability space — sigmoid logits if needed
    if not use_probs:
        pred_obj_top = 1.0 / (1.0 + np.exp(-pred_obj_top))
        if pred_obj_W is not None:
            pred_obj_W = 1.0 / (1.0 + np.exp(-pred_obj_W))

    # ── Pre-compute masks (fixed for the whole sweep) ──────────────────────
    pred_bin_top = binarise_predictions(pred_scores_top, jet_valid, prior_top, use_probs, threshold=threshold)
    pred_bin_W   = binarise_predictions(pred_scores_W,   jet_valid, prior_W,   use_probs, threshold=threshold)

    valid        = jet_valid[:, np.newaxis, :].astype(bool)
    target_top_b = target_masks_top.astype(bool)
    target_W_b   = target_masks_W.astype(bool)

    slot_perfect_top = ((pred_bin_top != target_top_b) & valid).sum(axis=2) == 0  # [N, Q]
    slot_perfect_W   = ((pred_bin_W   != target_W_b)   & valid).sum(axis=2) == 0  # [N, Q]

    # ── Ground-truth validity ───────────────────────────────────────────────
    if legacy_output_targets:
        is_real_top = target_masks_W.astype(bool).any(axis=-1)
        is_real_W   = is_real_top
    elif slot_valid_top is not None:
        is_real_top = slot_valid_top.astype(bool)
        is_real_W   = slot_valid_W.astype(bool) if slot_valid_W is not None else is_real_top
    else:
        is_real_top = target_masks_top.astype(bool).any(axis=-1)
        is_real_W   = target_masks_W.astype(bool).any(axis=-1)

    if require_complete_truth and not legacy_output_targets and slot_valid_top is None:
        is_real_top = is_real_top & (target_masks_top.astype(bool).sum(axis=-1) == 3)
        is_real_W   = is_real_W   & (target_masks_W.astype(bool).sum(axis=-1)   == 2)

    if require_top_for_w:
        is_real_W = is_real_W & is_real_top

    if joint:
        slot_perfect_top = slot_perfect_top & (slot_perfect_W | ~is_real_W)

    n_real_top   = is_real_top.sum(axis=1)
    both_tops    = n_real_top == 2
    n_both_tops  = int(both_tops.sum())
    n_top_total  = int(is_real_top.sum())
    n_W_total    = int(is_real_W.sum())

    # ── Sweep ───────────────────────────────────────────────────────────────
    top_effs, top_purities, top_f1s, top_mask_acc       = [], [], [], []
    W_effs,   W_purities,   W_f1s,   W_mask_acc         = [], [], [], []
    ttbar_effs, ttbar_purities, ttbar_f1s               = [], [], []

    for obj_t in obj_thresholds:
        pred_real_top = pred_obj_top > obj_t                               # [N, Q]
        pred_real_W   = pred_obj_W   > obj_t if pred_obj_W is not None else pred_real_top

        detected_top = slot_perfect_top & pred_real_top
        detected_W   = slot_perfect_W   & pred_real_W

        # Efficiency (strict: objectness must fire AND mask must be correct)
        t_eff = float((is_real_top & detected_top).sum() / max(n_top_total, 1))
        w_eff = float((is_real_W   & detected_W).sum()   / max(n_W_total, 1))

        all_tops_ok = ((~is_real_top) | detected_top).all(axis=1)
        tt_eff = float((all_tops_ok & both_tops).sum() / max(n_both_tops, 1))

        # Purity (precision: of predicted-real, fraction that are real AND have correct mask)
        n_pr_top   = int(pred_real_top.sum())
        n_pr_W     = int(pred_real_W.sum())
        pred_both  = pred_real_top.sum(axis=1) >= 2
        n_pr_both  = int(pred_both.sum())

        t_pur  = float((pred_real_top & is_real_top & slot_perfect_top).sum() / max(n_pr_top, 1))
        w_pur  = float((pred_real_W   & is_real_W   & slot_perfect_W).sum()   / max(n_pr_W, 1))
        both_correct = (pred_real_top & is_real_top & slot_perfect_top).sum(axis=1) == 2
        tt_pur = float((pred_both & both_correct).sum() / max(n_pr_both, 1))

        # Mask accuracy: of predicted-real queries, fraction with a correct mask
        # (no is_real gate — purely "did we get the mask right for what we picked?")
        t_macc  = float((pred_real_top & slot_perfect_top).sum() / max(n_pr_top, 1))
        w_macc  = float((pred_real_W   & slot_perfect_W).sum()   / max(n_pr_W, 1))

        top_effs.append(t_eff);   top_purities.append(t_pur);   top_f1s.append(_f1(t_pur, t_eff));   top_mask_acc.append(t_macc)
        W_effs.append(w_eff);     W_purities.append(w_pur);     W_f1s.append(_f1(w_pur, w_eff));     W_mask_acc.append(w_macc)
        ttbar_effs.append(tt_eff); ttbar_purities.append(tt_pur); ttbar_f1s.append(_f1(tt_pur, tt_eff))

    return {
        "thresholds":     np.array(obj_thresholds),
        "top_eff":        np.array(top_effs),
        "top_purity":     np.array(top_purities),
        "top_f1":         np.array(top_f1s),
        "top_mask_acc":   np.array(top_mask_acc),
        "W_eff":          np.array(W_effs),
        "W_purity":       np.array(W_purities),
        "W_f1":           np.array(W_f1s),
        "W_mask_acc":     np.array(W_mask_acc),
        "ttbar_eff":      np.array(ttbar_effs),
        "ttbar_purity":   np.array(ttbar_purities),
        "ttbar_f1":       np.array(ttbar_f1s),
    }


def plot_obj_threshold_sweep(run_dir: Path, sweep: dict, use_probs: bool = False):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not available — skipping objectness threshold sweep plot.")
        return

    thresholds = sweep["thresholds"]
    x_label = "Objectness threshold (probability)"  # always prob space

    best_idx = int(np.argmax(sweep["ttbar_f1"]))
    best_t   = float(thresholds[best_idx])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

    for ax, key_prefix, title in zip(
        axes,
        ["top", "W", "ttbar"],
        ["Top", "W", "ttbar"],
    ):
        eff  = sweep[f"{key_prefix}_eff"]   * 100
        pur  = sweep[f"{key_prefix}_purity"] * 100
        f1   = sweep[f"{key_prefix}_f1"]    * 100

        ax.plot(thresholds, eff, "o-", label="Efficiency (recall)", color="#4c72b0", lw=1.5, ms=3)
        ax.plot(thresholds, pur, "s-", label="Purity (precision)",  color="#dd8452", lw=1.5, ms=3)
        ax.plot(thresholds, f1,  "^-", label="F1",                  color="#55a868", lw=2,   ms=3)

        if f"{key_prefix}_mask_acc" in sweep:
            macc = sweep[f"{key_prefix}_mask_acc"] * 100
            ax.plot(thresholds, macc, "D--", label="Mask accuracy (pred real)", color="#9467bd", lw=1.5, ms=3)

        ax.axvline(best_t, color="#55a868", linestyle="--", alpha=0.6,
                   label=f"Best ttbar F1 @ {best_t:.3f}")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Score (%)")
        ax.set_title(f"{title}: Eff / Purity / F1 vs Objectness Threshold")
        ax.set_ylim(0, 105)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Objectness Threshold Sweep", fontsize=13)
    fig.tight_layout()

    out = run_dir / "eval_obj_threshold_sweep.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")

    best_tt_f1  = float(sweep["ttbar_f1"][best_idx])  * 100
    best_tt_eff = float(sweep["ttbar_eff"][best_idx])  * 100
    best_tt_pur = float(sweep["ttbar_purity"][best_idx]) * 100
    print(f"  Best ttbar F1:  {best_tt_f1:.2f}%  "
          f"(eff={best_tt_eff:.2f}%, purity={best_tt_pur:.2f}%)  "
          f"at obj_threshold={best_t:.4f}")


# ---------------------------------------------------------------------------
# Probability distribution / decision boundary plots
# ---------------------------------------------------------------------------

def plot_probability_distributions(
    run_dir: Path,
    pred_obj_top, target_obj_top,
    pred_obj_W=None, target_obj_W=None,
    use_probs: bool = False,
    obj_threshold: float = None,
    slot_valid_top=None, slot_valid_W=None,
    n_bins: int = 80,
):
    """
    Plot the distribution of predicted objectness probabilities split by ground-truth
    class (real vs null).  This is the classic decision-boundary visualisation:
    well-separated modes → easy threshold; overlapping → harder.

    Objectness scores are always converted to probabilities via sigmoid so the
    x-axis is always [0, 1] regardless of --use_probs.

    Saves: eval_prob_distributions.png
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not available — skipping probability distribution plot.")
        return

    # --- convert to probabilities -------------------------------------------
    def _to_prob(arr):
        if arr is None:
            return None
        arr = np.array(arr, dtype=float)
        return arr if use_probs else 1.0 / (1.0 + np.exp(-arr))

    prob_top = _to_prob(pred_obj_top)   # [N, Q]
    prob_W   = _to_prob(pred_obj_W)     # [N, Q] or None

    # --- resolve ground-truth labels ----------------------------------------
    # Prefer slot_valid (post-matching) over raw target_objectness
    def _resolve_labels(slot_valid, target_obj):
        if slot_valid is not None:
            return slot_valid.astype(bool)
        if target_obj is not None:
            return target_obj.astype(bool)
        return None

    label_top = _resolve_labels(slot_valid_top, target_obj_top)  # [N, Q] bool
    label_W   = _resolve_labels(slot_valid_W,   target_obj_W)    # [N, Q] bool or None

    # --- default threshold ---------------------------------------------------
    if obj_threshold is None:
        obj_threshold = 0.5  # always in prob space

    # --- decide panel layout -------------------------------------------------
    has_W = (prob_W is not None) and (label_W is not None)
    n_panels = 2 if has_W else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5), squeeze=False)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    def _draw_panel(ax, probs, labels, title):
        flat_probs  = probs.ravel()
        flat_labels = labels.ravel()

        real_probs = flat_probs[flat_labels]
        null_probs = flat_probs[~flat_labels]

        n_real = len(real_probs)
        n_null = len(null_probs)

        ax.hist(real_probs, bins=bin_edges, density=True, alpha=0.55,
                color="#55a868", label=f"Real  (N={n_real:,})", edgecolor="none")
        ax.hist(null_probs, bins=bin_edges, density=True, alpha=0.55,
                color="#dd8452", label=f"Null  (N={n_null:,})", edgecolor="none")

        ax.axvline(obj_threshold, color="k", linestyle="--", lw=1.5,
                   label=f"Threshold = {obj_threshold:.3f}")

        # Shade the misclassification regions lightly
        ax.axvspan(0.0, obj_threshold, alpha=0.04, color="#dd8452")   # null side
        ax.axvspan(obj_threshold, 1.0, alpha=0.04, color="#55a868")   # real side

        ax.set_xlabel("Predicted objectness probability")
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.set_xlim(0.0, 1.0)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)

        # Annotate overlap region statistics
        real_fp = float((real_probs < obj_threshold).sum() / max(n_real, 1))
        null_fp = float((null_probs > obj_threshold).sum() / max(n_null, 1))
        ax.text(0.02, 0.97,
                f"Missed real:  {real_fp*100:.1f}%\nFalse real:  {null_fp*100:.1f}%",
                transform=ax.transAxes, va="top", ha="left", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    _draw_panel(axes[0, 0], prob_top, label_top,
                "Top objectness: predicted probability by class")
    if has_W:
        _draw_panel(axes[0, 1], prob_W, label_W,
                    "W objectness: predicted probability by class")

    fig.suptitle("Objectness Decision Boundary — predicted probability distributions",
                 fontsize=12)
    fig.tight_layout()

    out = run_dir / "eval_prob_distributions.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------

def sweep_threshold_efficiencies(pred_scores_top, target_masks_top, pred_scores_W, target_masks_W,
                                  jet_valid, use_probs=False,
                                  t_min=None, t_max=None, n_steps=100,
                                  prior_top=None, prior_W=None, joint=False,
                                  valid_tops_truth=None, valid_Ws_truth=None,
                                  slot_valid_top=None, slot_valid_W=None,
                                  legacy_output_targets=False,
                                  require_complete_truth=False,
                                  require_top_for_w=False):
    """
    Evaluate efficiency metrics across a range of thresholds.

    Returns arrays (thresholds, top_effs, W_effs, ttbar_effs) each of length n_steps.
    """
    if t_min is None:
        t_min = 0.0 if use_probs else -5.0
    if t_max is None:
        t_max = 1.0 if use_probs else 5.0

    thresholds = np.linspace(t_min, t_max, n_steps)

    # Pre-compute masks that don't depend on threshold
    is_real = target_masks_W.astype(bool).any(axis=-1)    # [N, Q]
    if legacy_output_targets:
        is_real_top = is_real
        is_real_W = is_real
    elif slot_valid_top is not None:
        is_real_top = slot_valid_top.astype(bool)
        is_real_W = slot_valid_W.astype(bool) if slot_valid_W is not None else is_real
    else:
        is_real_top = valid_tops_truth if valid_tops_truth is not None else is_real
        is_real_W = valid_Ws_truth if valid_Ws_truth is not None else is_real

    if require_complete_truth and not legacy_output_targets and slot_valid_top is None:
        top_counts = target_masks_top.astype(bool).sum(axis=-1)
        W_counts = target_masks_W.astype(bool).sum(axis=-1)
        is_real_top = is_real_top & (top_counts == 3)
        is_real_W = is_real_W & (W_counts == 2)

    if require_top_for_w:
        is_real_W = is_real_W & is_real_top
    n_real_top   = is_real_top.sum(axis=1)
    both_tops    = n_real_top == 2
    valid        = jet_valid[:, np.newaxis, :].astype(bool)
    target_top_b = target_masks_top.astype(bool)
    target_W_b   = target_masks_W.astype(bool)
    top_eff_gate = is_real_top
    W_eff_gate   = is_real_W
    n_top_total  = int(top_eff_gate.sum())
    n_W_total    = int(W_eff_gate.sum())

    top_effs, W_effs, ttbar_effs = [], [], []

    for t in thresholds:
        pred_bin_top = binarise_predictions(
            pred_scores_top, jet_valid, prior_top, use_probs, threshold=t
        )
        pred_bin_W = binarise_predictions(
            pred_scores_W, jet_valid, prior_W, use_probs, threshold=t
        )

        slot_perfect_top = ((pred_bin_top != target_top_b) & valid).sum(axis=2) == 0
        slot_perfect_W   = ((pred_bin_W   != target_W_b)   & valid).sum(axis=2) == 0

        if joint:
            slot_perfect_top = slot_perfect_top & (slot_perfect_W | ~is_real_W)

        all_tops_perfect = ((~is_real_top) | slot_perfect_top).all(axis=1)
        n_top_correct = (top_eff_gate & slot_perfect_top).sum()
        n_W_correct   = (W_eff_gate & slot_perfect_W).sum()

        top_effs.append(float(n_top_correct / max(n_top_total, 1)))
        W_effs.append(float(n_W_correct / max(n_W_total, 1)))
        ttbar_effs.append(float((all_tops_perfect & both_tops).sum() / max(both_tops.sum(), 1)))

    return np.array(thresholds), np.array(top_effs), np.array(W_effs), np.array(ttbar_effs)


def plot_threshold_sweep(run_dir: Path, thresholds, top_effs, W_effs, ttbar_effs,
                         use_probs=False):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not available — skipping threshold sweep plot.")
        return

    x_label = "Threshold (probability)" if use_probs else "Threshold (logit)"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, ttbar_effs * 100, "^-", label="ttbar efficiency", color="#55a868", lw=1.5)
    ax.plot(thresholds, top_effs   * 100, "o-", label="Top efficiency",   color="#4c72b0", lw=1.5)
    ax.plot(thresholds, W_effs     * 100, "s-", label="W efficiency",     color="#dd8452", lw=1.5)

    best_idx = np.argmax(ttbar_effs)
    ax.axvline(thresholds[best_idx], color="#55a868", linestyle="--", alpha=0.6,
               label=f"Best ttbar @ {thresholds[best_idx]:.3f} ({ttbar_effs[best_idx]*100:.1f}%)")

    ax.set_xlabel(x_label)
    ax.set_ylabel("Efficiency (%)")
    ax.set_title("Chain Queries: Efficiency vs Binarisation Threshold")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = run_dir / "eval_threshold_sweep.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")
    print(f"  Best ttbar efficiency: {ttbar_effs[best_idx]*100:.2f}% at threshold={thresholds[best_idx]:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute top/W/ttbar reconstruction efficiency from chain_queries test outputs."
    )
    parser.add_argument(
        "--run_dir", required=True, type=Path,
        help="Path to lightning_logs/version_X directory containing test_outputs_*.h5"
    )
    parser.add_argument(
        "--data_file", default=None, type=Path,
        help="HDF5 data file with 'src_mask' key (required if jet_valid_mask absent in outputs)"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Save PNG efficiency plots to --run_dir"
    )
    parser.add_argument(
        "--csv", action="store_true",
        help="Save ttbar efficiency/purity/F1 summary to eval_ttbar_summary.csv in --run_dir"
    )
    parser.add_argument(
        "--use_probs", action="store_true",
        help="Use saved sigmoid probabilities instead of raw logits (default threshold 0.5)"
    )
    parser.add_argument(
        "--threshold", type=float, default=None, metavar="T",
        help=(
            "Binarisation threshold applied to scores (logits or probs). "
            "Overrides the default (0.0 for logits, 0.5 for probs). "
            "Ignored when --prior is used for the corresponding mask type."
        ),
    )
    parser.add_argument(
        "--prob_dist", action="store_true",
        help=(
            "Plot predicted objectness probability distributions split by class "
            "(real vs null) to visualise the decision boundary. "
            "Saved to eval_prob_distributions.png in --run_dir."
        ),
    )
    parser.add_argument(
        "--prob_dist_bins", type=int, default=80, metavar="N",
        help="Number of histogram bins for --prob_dist (default: 80)",
    )
    parser.add_argument(
        "--obj_threshold_sweep", action="store_true",
        help="Sweep objectness threshold and plot purity/efficiency/F1 curves (saved to --run_dir)"
    )
    parser.add_argument(
        "--obj_sweep_range", nargs=2, type=float, metavar=("MIN", "MAX"), default=None,
        help="Threshold range for objectness sweep in probability space (default: [0, 1])."
    )
    parser.add_argument(
        "--obj_sweep_steps", type=int, default=100, metavar="N",
        help="Number of threshold steps in the objectness sweep (default: 100)"
    )
    parser.add_argument(
        "--threshold_sweep", action="store_true",
        help="Plot efficiency vs threshold curve (saved to --run_dir)"
    )
    parser.add_argument(
        "--sweep_range", nargs=2, type=float, metavar=("MIN", "MAX"), default=None,
        help=(
            "Threshold range for sweep. "
            "Defaults: [0, 1] for probs, [-5, 5] for logits."
        ),
    )
    parser.add_argument(
        "--sweep_steps", type=int, default=100, metavar="N",
        help="Number of threshold steps in the sweep (default: 100)"
    )
    parser.add_argument(
        "--obj_threshold", type=float, default=None, metavar="T",
        help=(
            "Threshold applied to objectness scores to classify a query as real. "
            "Overrides the default (0.0 for logits, 0.5 for probs). "
            "Affects purity metrics and --strict gating."
        ),
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Require objectness prediction to also mark the slot as real for efficiency metrics"
    )
    parser.add_argument(
        "--joint", action="store_true",
        help="Count a top as correctly reconstructed only if its associated W is also correct"
    )
    parser.add_argument(
        "--prior", nargs="+", metavar="TYPE=K", default=[],
        help=(
            "Per-type top-k binarisation prior, e.g. --prior top=3 W=2. "
            "For each mask type, select the K highest-scoring valid particles "
            "instead of thresholding."
        ),
    )
    parser.add_argument(
        "--fixed_slot_eval", action="store_true",
        help=(
            "Compatibility flag. This evaluator already uses fixed slot index matching, "
            "so enabling this does not change behaviour."
        ),
    )
    parser.add_argument(
        "--legacy_output_targets", action="store_true",
        help=(
            "Evaluate using legacy output targets/objectness gates for both top and W denominators."
        ),
    )
    parser.add_argument(
        "--require_complete_truth", action="store_true",
        help=(
            "In truth-based mode, only count real tops/Ws with complete truth multiplicity "
            "(top=3 particles, W=2 particles)."
        ),
    )
    parser.add_argument(
        "--require_top_for_w", action="store_true",
        help="Only count a W as real when its corresponding top is also real",
    )
    args = parser.parse_args()

    # Parse --prior top=3 W=2
    prior_top = None
    prior_W   = None
    for item in args.prior:
        if "=" not in item:
            sys.exit(f"ERROR: --prior entries must be TYPE=K, got '{item}'")
        name, k_str = item.split("=", 1)
        if name not in ("top", "W"):
            sys.exit(f"ERROR: unknown type '{name}' in --prior. Valid: top, W")
        try:
            k = int(k_str)
        except ValueError:
            sys.exit(f"ERROR: K must be an integer in --prior, got '{k_str}'")
        if name == "top":
            prior_top = k
        else:
            prior_W = k

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        sys.exit(f"ERROR: --run_dir does not exist or is not a directory: {run_dir}")

    (pred_scores_top, target_masks_top, pred_scores_W, target_masks_W,
     jet_valid, target_obj_top, pred_obj_top, target_obj_W, pred_obj_W,
     original_mult, valid_tops_truth, valid_Ws_truth,
     slot_valid_top, slot_valid_W) = load_run_data(
        run_dir, args.data_file, use_probs=args.use_probs
    )

    results = compute_efficiencies(
        pred_scores_top, target_masks_top, pred_scores_W, target_masks_W,
        jet_valid, target_obj_top,
        pred_obj_top=pred_obj_top, target_obj_W=target_obj_W,
        pred_obj_W=pred_obj_W, prior_top=prior_top, prior_W=prior_W,
        use_probs=args.use_probs, threshold=args.threshold, obj_threshold=args.obj_threshold,
        strict=args.strict,
        joint=args.joint, original_mult=original_mult,
        valid_tops_truth=valid_tops_truth,
        valid_Ws_truth=valid_Ws_truth,
        slot_valid_top=slot_valid_top,
        slot_valid_W=slot_valid_W,
        legacy_output_targets=args.legacy_output_targets,
        require_complete_truth=args.require_complete_truth,
        require_top_for_w=args.require_top_for_w,
    )
    print_results(run_dir, results)

    if args.plot:
        make_plots(run_dir, results)

    if args.prob_dist:
        if pred_obj_top is None:
            print("WARNING: no objectness predictions found — skipping probability distribution plot.")
        else:
            print("\nPlotting objectness probability distributions...")
            plot_probability_distributions(
                run_dir,
                pred_obj_top, target_obj_top,
                pred_obj_W=pred_obj_W, target_obj_W=target_obj_W,
                use_probs=args.use_probs,
                obj_threshold=args.obj_threshold if args.obj_threshold is not None
                              else (0.5 if args.use_probs else None),
                slot_valid_top=slot_valid_top,
                slot_valid_W=slot_valid_W,
                n_bins=args.prob_dist_bins,
            )

    if args.csv:
        save_csv(run_dir, results)

    if args.threshold_sweep:
        t_min, t_max = args.sweep_range if args.sweep_range else (None, None)
        print(f"\nRunning threshold sweep ({args.sweep_steps} steps)...")
        thresholds, top_effs, W_effs, ttbar_effs = sweep_threshold_efficiencies(
            pred_scores_top, target_masks_top, pred_scores_W, target_masks_W,
            jet_valid,
            use_probs=args.use_probs,
            t_min=t_min, t_max=t_max,
            n_steps=args.sweep_steps,
            prior_top=prior_top, prior_W=prior_W,
            joint=args.joint,
            valid_tops_truth=valid_tops_truth,
            valid_Ws_truth=valid_Ws_truth,
            slot_valid_top=slot_valid_top,
            slot_valid_W=slot_valid_W,
            legacy_output_targets=args.legacy_output_targets,
            require_complete_truth=args.require_complete_truth,
            require_top_for_w=args.require_top_for_w,
        )
        plot_threshold_sweep(run_dir, thresholds, top_effs, W_effs, ttbar_effs,
                             use_probs=args.use_probs)

    if args.obj_threshold_sweep:
        if pred_obj_top is None:
            print("WARNING: no objectness predictions found — skipping objectness threshold sweep.")
        else:
            obj_t_min, obj_t_max = args.obj_sweep_range if args.obj_sweep_range else (None, None)
            print(f"\nRunning objectness threshold sweep ({args.obj_sweep_steps} steps)...")
            sweep = sweep_obj_threshold(
                pred_scores_top, target_masks_top, pred_scores_W, target_masks_W,
                jet_valid, pred_obj_top, pred_obj_W=pred_obj_W,
                use_probs=args.use_probs,
                threshold=args.threshold,
                prior_top=prior_top, prior_W=prior_W,
                slot_valid_top=slot_valid_top, slot_valid_W=slot_valid_W,
                legacy_output_targets=args.legacy_output_targets,
                require_complete_truth=args.require_complete_truth,
                require_top_for_w=args.require_top_for_w,
                joint=args.joint,
                t_min=obj_t_min, t_max=obj_t_max,
                n_steps=args.obj_sweep_steps,
            )
            plot_obj_threshold_sweep(run_dir, sweep, use_probs=args.use_probs)


if __name__ == "__main__":
    main()