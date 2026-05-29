"""
Evaluate top/W/ttbar reconstruction efficiency from saved test outputs.

Usage:
    python analysis/evaluate.py --run_dir <path/to/version_X>
    python analysis/evaluate.py --run_dir <path/to/version_X> --plot
    python analysis/evaluate.py --run_dir <path/to/version_X> \\
        --data_file <path/to/ttbar_preprocessed_test.h5> --plot

Custom threshold (overrides the default 0.0 for logits / 0.5 for probs):
    python analysis/evaluate.py --run_dir ... --threshold 0.3
    python analysis/evaluate.py --run_dir ... --use_probs --threshold 0.6

Prior-based binarisation (top-k particles per slot type):
    python analysis/evaluate.py --run_dir ... --prior top=3 W=2
    python analysis/evaluate.py --run_dir ... --prior top=3 W=2 --use_probs

Save a plain-text summary:
    python analysis/evaluate.py --run_dir ... --save_summary
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Quick per-task metrics (objectness precision/recall/F1, type accuracy, mask IoU)
# ---------------------------------------------------------------------------

def compute_objectness_metrics(obj_path: Path) -> dict:
    """Precision, recall, F1 for the objectness head."""
    with h5py.File(obj_path, "r") as f:
        logits  = f["predicted_objectness_logit"][:]  # [N, Q]
        targets = f["target_objectness"][:]            # [N, Q]
    probs     = 1.0 / (1.0 + np.exp(-logits))
    preds     = (probs > 0.5).astype(int).ravel()
    tgts      = targets.astype(int).ravel()
    tp = int(((preds == 1) & (tgts == 1)).sum())
    fp = int(((preds == 1) & (tgts == 0)).sum())
    fn = int(((preds == 0) & (tgts == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-8)
    return {"obj_precision": precision, "obj_recall": recall, "obj_f1": f1,
            "obj_tp": tp, "obj_fp": fp, "obj_fn": fn}


def compute_type_accuracy(type_path: Path) -> dict:
    """Binary top-vs-W accuracy evaluated only on real (non-null) query slots."""
    with h5py.File(type_path, "r") as f:
        logits  = f["predicted_type_logit"][:]   # [N, Q]
        targets = f["target_type"][:]             # [N, Q]
        classes = f["target_classes"][:]          # [N, Q]
    real_mask = classes > 0
    if not real_mask.any():
        return {"type_accuracy": 0.0, "type_n_real": 0}
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs > 0.5).astype(float)
    accuracy = float((preds[real_mask] == targets[real_mask]).mean())
    return {"type_accuracy": accuracy, "type_n_real": int(real_mask.sum())}


def compute_mask_iou(mask_path: Path) -> dict:
    """Mean IoU between predicted and target masks over slots with ≥1 target particle."""
    with h5py.File(mask_path, "r") as f:
        pred_logits = f["predicted_masks_logits"][:]  # [N, Q, P]
        targets     = f["target_masks"][:]             # [N, Q, P]
    pred_bin     = (pred_logits > 0).astype(float)
    intersection = (pred_bin * targets).sum(axis=-1)
    union        = np.clip(pred_bin + targets, 0, 1).sum(axis=-1)
    has_particles = targets.sum(axis=-1) > 0
    if not has_particles.any():
        return {"mask_mean_iou": 0.0, "mask_n_objects": 0}
    iou = intersection[has_particles] / (union[has_particles] + 1e-8)
    return {"mask_mean_iou": float(iou.mean()), "mask_n_objects": int(has_particles.sum())}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_run_data(run_dir: Path, data_file, use_probs: bool = False):
    """Load all arrays needed for efficiency evaluation."""

    mask_path  = run_dir / "test_outputs_mask.h5"
    obj_path   = run_dir / "test_outputs_objectness.h5"
    type_path  = run_dir / "test_outputs_object_type.h5"

    for p in (mask_path, obj_path, type_path):
        if not p.exists():
            sys.exit(f"ERROR: required file not found: {p}")

    scores_key = "predicted_masks_prob" if use_probs else "predicted_masks_logits"
    with h5py.File(mask_path, "r") as f:
        if scores_key not in f:
            sys.exit(f"ERROR: key '{scores_key}' not found in {mask_path}")
        pred_scores  = f[scores_key][:]                  # [N, Q, P]
        target_masks = f["target_masks"][:]              # [N, Q, P]
        jet_valid    = f["jet_valid_mask"][:] if "jet_valid_mask" in f else None  # [N, P]

    with h5py.File(obj_path, "r") as f:
        target_obj = f["target_objectness"][:]           # [N, Q]
        pred_obj_key = "predicted_objectness_prob" if use_probs else "predicted_objectness_logit"
        pred_obj = f[pred_obj_key][:] if pred_obj_key in f else None  # [N, Q]

    with h5py.File(type_path, "r") as f:
        target_cls = f["target_classes"][:]              # [N, Q]
        type_scores_key = "predicted_type_prob" if use_probs else "predicted_type_logit"
        pred_type = f[type_scores_key][:] if type_scores_key in f else None  # [N, Q]

    # Fall back to external data file for src_mask
    if jet_valid is None:
        if data_file is None:
            sys.exit(
                "ERROR: 'jet_valid_mask' not found in test_outputs_mask.h5 and "
                "--data_file was not provided.\n"
                "Re-run with --data_file pointing to the corresponding HDF5 data file."
            )
        if not data_file.exists():
            sys.exit(f"ERROR: data file not found: {data_file}")
        with h5py.File(data_file, "r") as f:
            if "src_mask" not in f:
                sys.exit(f"ERROR: 'src_mask' key not found in {data_file}")
            jet_valid = f["src_mask"][:]                 # [N, P]
        N_run = pred_scores.shape[0]
        N_data = jet_valid.shape[0]
        if N_data != N_run:
            sys.exit(
                f"ERROR: event count mismatch — run has {N_run} events but "
                f"data file has {N_data}. Make sure you are using the correct test split."
            )

    # Load original (pre-signal-jet-filtering) multiplicities if available
    orig_mult_path = run_dir / "event_multiplicities.npz"
    original_mult = None
    if orig_mult_path.exists():
        d = np.load(orig_mult_path)
        original_mult = d["original_mult"].astype(int)

    return pred_scores, target_masks, jet_valid, target_obj, target_cls, pred_obj, pred_type, original_mult


# ---------------------------------------------------------------------------
# Prediction binarisation
# ---------------------------------------------------------------------------

# Human-readable type names → integer class IDs stored in target_classes
TYPE_IDS = {"top": 1, "W": 2, "null": 0}


def binarise_predictions(scores, jet_valid, target_cls, priors, use_probs, threshold=None):
    """
    Convert raw scores to a binary [N, Q, P] prediction array.

    Default threshold: 0.0 for logits, 0.5 for probs.
    Pass an explicit `threshold` to override the default.

    priors: dict {type_id (int) -> k (int)}
        For slots of that type, select exactly the k highest-scoring
        *valid* particles instead of thresholding.  Padding positions
        (jet_valid == 0) are never selected regardless of k.
    """
    N, Q, P = scores.shape
    valid = jet_valid.astype(bool)                              # [N, P]

    if threshold is None:
        threshold = 0.5 if use_probs else 0.0
    pred_bin  = (scores > threshold).copy()                    # [N, Q, P]

    if not priors:
        return pred_bin

    # Mask padding positions to -inf so they are never top-k selected
    masked_scores = np.where(valid[:, np.newaxis, :], scores, -np.inf)  # [N, Q, P]

    for q in range(Q):
        for type_id, k in priors.items():
            sel = target_cls[:, q] == type_id                  # [N] events with this type at slot q
            if not sel.any():
                continue
            slot_scores = masked_scores[sel, q, :]             # [n_sel, P]
            # argsort ascending → take last k for top-k
            topk_idx    = np.argsort(slot_scores, axis=1)[:, -k:]   # [n_sel, k]
            new_bin     = np.zeros((sel.sum(), P), dtype=bool)
            np.put_along_axis(new_bin, topk_idx, True, axis=1)
            new_bin    &= valid[sel]                           # never select padding
            pred_bin[sel, q, :] = new_bin

    return pred_bin


# ---------------------------------------------------------------------------
# Core efficiency computation
# ---------------------------------------------------------------------------

def _compute_breakdown(multiplicity_arr, is_top, is_W, is_real, slot_detected_perfect,
                        all_tops_perfect, all_Ws_perfect, pred_real, pred_is_top, pred_is_W):
    """Compute per-multiplicity efficiency/purity rows for a given multiplicity array."""
    mult_values = sorted(np.unique(multiplicity_arr).tolist())
    breakdown = {}
    for m in mult_values:
        sel = multiplicity_arr == m
        n_sel = sel.sum()
        if n_sel == 0:
            continue
        hw = (is_W[sel] & is_real[sel]).any(axis=1).sum()
        bt = ((is_top[sel] & is_real[sel]).sum(axis=1) == 2).sum()
        n_top_correct = (is_top[sel] & is_real[sel] & slot_detected_perfect[sel]).sum()
        n_top_total   = (is_top[sel] & is_real[sel]).sum()
        row = {
            "n_events":  int(n_sel),
            "top_eff":   float(n_top_correct / max(n_top_total, 1)),
            "W_eff":     float((all_Ws_perfect[sel]   & (is_W[sel] & is_real[sel]).any(axis=1)).sum() / max(hw, 1)),
            "ttbar_eff": float((all_tops_perfect[sel] & ((is_top[sel] & is_real[sel]).sum(axis=1) == 2)).sum() / max(bt, 1)),
            "n_top":     int(n_top_total),
            "n_W":       int(hw),
            "n_ttbar":   int(bt),
        }
        if pred_real is not None:
            pr   = pred_real[sel]
            ir   = is_real[sel]; it = is_top[sel]; iw = is_W[sel]
            sdp  = slot_detected_perfect[sel]
            pit  = pred_is_top[sel]
            piw  = pred_is_W[sel]
            n_pr_top  = int((pr & pit).sum())
            n_pr_W    = int((pr & piw).sum())
            row["top_purity"]   = float((pr & pit & it & ir & sdp).sum() / max(n_pr_top, 1))
            row["W_purity"]     = float((pr & piw & iw & ir & sdp).sum() / max(n_pr_W, 1))
            pr_both = (pr & pit).sum(axis=1) >= 2
            n_pr_both = int(pr_both.sum())
            pr_both_correct = (pr & pit & it & ir & sdp).sum(axis=1) == 2
            row["ttbar_purity"]     = float((pr_both & pr_both_correct).sum() / max(n_pr_both, 1))
            row["n_pred_top"]       = n_pr_top
            row["n_pred_W"]         = n_pr_W
            row["n_pred_both_tops"] = n_pr_both
        breakdown[m] = row
    return breakdown


def compute_efficiencies(pred_scores, target_masks, jet_valid, target_obj, target_cls,
                         pred_obj=None, pred_type=None, priors=None, use_probs=False,
                         threshold=None, strict=False, original_mult=None):
    """
    Returns a dict with scalar efficiencies and per-multiplicity breakdowns.

    Efficiency definition: fraction of events (with ≥1 object of that type)
    where ALL objects of that type are exactly reconstructed (zero token errors
    on valid / non-padded particle positions).
    """
    is_real = target_obj > 0.5                               # [N, Q]
    is_top  = target_cls == 1                                # [N, Q]
    is_W    = target_cls == 2                                # [N, Q]
    n_tops  = is_top.sum(axis=1)                             # [N]
    n_Ws    = is_W.sum(axis=1)                               # [N]

    valid    = jet_valid[:, np.newaxis, :].astype(bool)      # [N, 1, P] broadcasts
    pred_bin = binarise_predictions(
        pred_scores, jet_valid, target_cls, priors or {}, use_probs, threshold=threshold
    )                                                        # [N, Q, P]
    target_b        = target_masks.astype(bool)
    mismatch        = (pred_bin != target_b) & valid             # [N, Q, P]
    errors_per_slot = mismatch.sum(axis=2)                       # [N, Q]
    slot_perfect    = errors_per_slot == 0                        # [N, Q]

    # In strict mode, also require the objectness head to predict the slot as real
    pred_real = None  # set below if pred_obj is available
    if strict and pred_obj is not None:
        obj_thresh = 0.5 if use_probs else 0.0
        pred_real  = pred_obj > obj_thresh                       # [N, Q]
        slot_detected_perfect = slot_perfect & pred_real
    else:
        slot_detected_perfect = slot_perfect

    # all_tops_perfect: every real top slot is perfect (used for ttbar efficiency)
    all_tops_perfect = ((~is_top) | (~is_real) | slot_detected_perfect).all(axis=1)  # [N]
    all_Ws_perfect   = ((~is_W)   | (~is_real) | slot_detected_perfect).all(axis=1)  # [N]
    perfect_all      = (~is_real  | slot_detected_perfect).all(axis=1)               # [N]

    has_top   = n_tops >= 1
    has_W     = n_Ws >= 1
    both_tops = n_tops == 2               # events with exactly 2 tops

    # top efficiency: per-slot — fraction of individual real top objects correctly reconstructed
    n_top_slots_correct = (is_top & is_real & slot_detected_perfect).sum()
    n_top_slots_total   = (is_top & is_real).sum()

    # W efficiency:     ≥1 W present, all Ws perfectly reconstructed (event-level)
    # ttbar efficiency: exactly 2 tops present, both tops perfectly reconstructed (event-level)
    top_eff   = n_top_slots_correct / max(n_top_slots_total, 1)
    W_eff     = (all_Ws_perfect   & has_W).sum()     / max(has_W.sum(),     1)
    ttbar_eff = (all_tops_perfect & both_tops).sum() / max(both_tops.sum(), 1)
    all_eff   = perfect_all.sum()                    / len(perfect_all)

    # ── Object purity (from objectness predictions) ──
    obj_purity    = None
    ttbar_purity  = None
    n_pred_real   = None
    top_purity    = None
    W_purity      = None
    n_pred_top    = None
    n_pred_W      = None
    n_pred_both_tops = None
    pred_is_top = None
    pred_is_W   = None
    if pred_obj is not None:
        obj_thresh = 0.5 if use_probs else 0.0
        pred_real  = pred_obj > obj_thresh                        # [N, Q]
        n_pred_real = int(pred_real.sum())
        # Of predicted-real slots, fraction that are actually real
        obj_purity   = float((pred_real & is_real).sum()
                             / max(n_pred_real, 1))
        # Per-type purity using predicted type (positive logit/prob>0.5 → top, else W)
        if pred_type is not None:
            type_thresh  = 0.5 if use_probs else 0.0
            pred_is_top  = pred_type > type_thresh                # [N, Q]
            pred_is_W    = ~pred_is_top                           # [N, Q]
        else:
            # Fall back to ground-truth type if type predictions unavailable
            pred_is_top = is_top
            pred_is_W   = is_W
        # Denominator: predicted-real slots of predicted type
        n_pred_top = int((pred_real & pred_is_top).sum())
        n_pred_W   = int((pred_real & pred_is_W).sum())
        # Numerator: predicted-real, predicted correct type, actually correct type, correct mask
        top_purity = float((pred_real & pred_is_top & is_top & is_real & slot_detected_perfect).sum()
                           / max(n_pred_top, 1))
        W_purity   = float((pred_real & pred_is_W & is_W & is_real & slot_detected_perfect).sum()
                           / max(n_pred_W, 1))
        # ttbar purity: of events where model predicts >=2 real predicted-top slots, fraction with both correct
        pred_both_tops   = (pred_real & pred_is_top).sum(axis=1) >= 2    # [N]
        n_pred_both_tops = int(pred_both_tops.sum())
        both_tops_correct = (pred_real & pred_is_top & is_top & is_real & slot_detected_perfect).sum(axis=1) == 2
        ttbar_purity = float((pred_both_tops & both_tops_correct).sum()
                             / max(n_pred_both_tops, 1))

    # Per-multiplicity breakdown (by signal jet count)
    multiplicity = jet_valid.sum(axis=1).astype(int)         # [N]
    breakdown = _compute_breakdown(
        multiplicity, is_top, is_W, is_real, slot_detected_perfect,
        all_tops_perfect, all_Ws_perfect, pred_real, pred_is_top, pred_is_W,
    )

    # Per-original-multiplicity breakdown (grouped by true event jet count before signal filtering)
    breakdown_orig = {}
    if original_mult is not None:
        breakdown_orig = _compute_breakdown(
            original_mult, is_top, is_W, is_real, slot_detected_perfect,
            all_tops_perfect, all_Ws_perfect, pred_real, pred_is_top, pred_is_W,
        )

    return {
        "N": len(pred_scores),
        "Q": pred_scores.shape[1],
        "P": pred_scores.shape[2],
        "top_eff":   float(top_eff),
        "W_eff":     float(W_eff),
        "ttbar_eff": float(ttbar_eff),
        "all_eff":   float(all_eff),
        "obj_purity":       obj_purity,
        "ttbar_purity":     ttbar_purity,
        "top_purity":       top_purity,
        "W_purity":         W_purity,
        "n_pred_real":      n_pred_real,
        "n_pred_top":       n_pred_top,
        "n_pred_W":         n_pred_W,
        "n_pred_both_tops": n_pred_both_tops,
        "n_has_top":   int(n_top_slots_total),
        "n_has_W":     int(has_W.sum()),
        "n_both_tops": int(both_tops.sum()),
        "breakdown":      breakdown,
        "breakdown_orig": breakdown_orig,
        "priors":      priors or {},
        "use_probs":   use_probs,
        "threshold":   threshold if threshold is not None else (0.5 if use_probs else 0.0),
        "strict":      strict,
    }


# ---------------------------------------------------------------------------
# Text output
# ---------------------------------------------------------------------------

def print_results(run_dir: Path, results: dict):
    N  = results["N"]
    Q  = results["Q"]
    P  = results["P"]
    br = results["breakdown"]

    id_to_name = {v: k for k, v in TYPE_IDS.items()}
    prior_str  = ", ".join(
        f"{id_to_name.get(tid, tid)}={k}" for tid, k in results["priors"].items()
    ) if results["priors"] else f"threshold={results['threshold']}"
    scores_str = "probs" if results["use_probs"] else "logits"

    print(f"\n=== Evaluation: {run_dir} ===")
    print(f"Events: {N:,}  |  Query slots Q: {Q}  |  Particles P: {P}")
    print(f"Scores: {scores_str}  |  Binarisation: {prior_str}\n")

    if results.get("strict"):
        print("EFFICIENCY SUMMARY (strict: pred_real & correct mask / N_real)")
    else:
        print("EFFICIENCY SUMMARY (recall: N_correct / N_real)")
    print("─" * 70)
    print(f"  Top efficiency     (per top object):                  {results['top_eff']*100:6.2f}%   (N={results['n_has_top']:,} tops)")
    print(f"  W efficiency       (>=1 W, all correct):             {results['W_eff']*100:6.2f}%   (N={results['n_has_W']:,})")
    print(f"  ttbar efficiency   (exactly 2 tops, both correct):   {results['ttbar_eff']*100:6.2f}%   (N={results['n_both_tops']:,})")
    print("─" * 70)
    print(f"  All-object efficiency:                                {results['all_eff']*100:6.2f}%   (N={N:,})")

    if results.get("obj_purity") is not None:
        print(f"\nPURITY SUMMARY (N_correct_predicted / N_all_predicted)")
        print("─" * 70)
        print(f"  Object purity      (pred real & actual real / pred real):       {results['obj_purity']*100:5.2f}%   (N_pred_real={results['n_pred_real']:,})")
        print(f"  Top purity         (pred real top & perfect / pred real top):   {results['top_purity']*100:5.2f}%   (N_pred_top={results['n_pred_top']:,})")
        print(f"  W purity           (pred real W & perfect / pred real W):      {results['W_purity']*100:5.2f}%   (N_pred_W={results['n_pred_W']:,})")
        print(f"  ttbar purity       (both tops pred real & correct / pred 2t):  {results['ttbar_purity']*100:5.2f}%   (N_pred_2t={results['n_pred_both_tops']:,})")
        print("─" * 70)

    def _print_mult_table(label, table):
        if not table:
            return
        print(f"\n{label}")
        header = f"  {'Jets':>5}   {'Top eff':>8}   {'W eff':>8}   {'ttbar eff':>9}   {'Events':>8}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for m, row in sorted(table.items()):
            print(
                f"  {m:>5}   {row['top_eff']*100:>7.2f}%   "
                f"{row['W_eff']*100:>7.2f}%   "
                f"{row['ttbar_eff']*100:>8.2f}%   "
                f"{row['n_events']:>8,}"
            )

    _print_mult_table("EFFICIENCY BY SIGNAL JET MULTIPLICITY (jets fed to model)", br)
    _print_mult_table("EFFICIENCY BY ORIGINAL JET MULTIPLICITY (all jets in event)", results.get("breakdown_orig", {}))

    if results.get("quick_metrics"):
        qm = results["quick_metrics"]
        print("QUICK METRICS (threshold = 0.5 on sigmoid)")
        print("─" * 70)
        if "obj_f1" in qm:
            print(f"  Objectness  precision={qm['obj_precision']:.4f}  recall={qm['obj_recall']:.4f}  F1={qm['obj_f1']:.4f}"
                  f"  (TP={qm['obj_tp']}, FP={qm['obj_fp']}, FN={qm['obj_fn']})")
        if "type_accuracy" in qm:
            print(f"  Type accuracy (real slots only): {qm['type_accuracy']:.4f}  (N={qm['type_n_real']})")
        if "mask_mean_iou" in qm:
            print(f"  Mask mean IoU: {qm['mask_mean_iou']:.4f}  (N objects={qm['mask_n_objects']})")
        print("─" * 70)
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
    return {"top_eff": top_eff, "W_eff": W_eff, "ttbar_eff": ttbar_eff, "n_events": n_events}


def _agg_purity_bin(br: dict, mult_keys):
    """Aggregate purity across a set of multiplicity keys using weighted average."""
    n_pr_top  = sum(br[m]["n_pred_top"]       for m in mult_keys if m in br)
    n_pr_W    = sum(br[m]["n_pred_W"]         for m in mult_keys if m in br)
    n_pr_both = sum(br[m]["n_pred_both_tops"] for m in mult_keys if m in br)
    top   = sum(br[m]["top_purity"]   * br[m]["n_pred_top"]       for m in mult_keys if m in br) / max(n_pr_top,  1)
    W     = sum(br[m]["W_purity"]     * br[m]["n_pred_W"]         for m in mult_keys if m in br) / max(n_pr_W,    1)
    ttbar = sum(br[m]["ttbar_purity"] * br[m]["n_pred_both_tops"] for m in mult_keys if m in br) / max(n_pr_both, 1)
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

    # 1. Grouped bar chart: Top / W / ttbar efficiency in bins 6, 7, ≥8, All
    all_mults = sorted(br.keys())
    ge8_keys  = [m for m in all_mults if m >= 8]

    bins = {
        "6 jets":  _agg_bin(br, [6]),
        "7 jets":  _agg_bin(br, [7]),
        "≥8 jets": _agg_bin(br, ge8_keys),
        "All":     {
            "top_eff":   results["top_eff"],
            "W_eff":     results["W_eff"],
            "ttbar_eff": results["ttbar_eff"],
            "n_events":  results["N"],
        },
    }

    bin_labels  = list(bins.keys())
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

    # Annotate event counts below x-axis labels
    bin_counts = [f"{bins[b]['n_events']:,}" for b in bin_labels]
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lbl}\n(N={n})" for lbl, n in zip(bin_labels, bin_counts)])
    ax.set_ylabel("Efficiency (%)")
    ax.set_title("Reconstruction Efficiency by Jet Multiplicity Bin")
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
        top_effs   = [table[m]["top_eff"]   * 100 for m in mults]
        W_effs     = [table[m]["W_eff"]     * 100 for m in mults]
        ttbar_effs = [table[m]["ttbar_eff"] * 100 for m in mults]
        ax.plot(mults, top_effs,   "o-", label="Top efficiency",   color="#4c72b0")
        ax.plot(mults, W_effs,     "s-", label="W efficiency",     color="#dd8452")
        ax.plot(mults, ttbar_effs, "^-", label="ttbar efficiency", color="#55a868")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Efficiency (%)")
        ax.set_title(title)
        ax.legend()
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)

    # 2. Efficiency vs multiplicity
    br_orig = results.get("breakdown_orig", {})
    if br or br_orig:
        n_panels = (1 if br else 0) + (1 if br_orig else 0)
        fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 4), squeeze=False)
        panel = 0
        if br:
            _plot_eff_vs_mult(axes[0, panel], br,
                              "Number of signal jets (fed to model)",
                              "Reconstruction Efficiency vs Signal Jet Multiplicity")
            panel += 1
        if br_orig:
            _plot_eff_vs_mult(axes[0, panel], br_orig,
                              "Number of jets in full event (original multiplicity)",
                              "Reconstruction Efficiency vs Original Jet Multiplicity")
        fig.tight_layout()
        out = run_dir / "eval_efficiency_vs_multiplicity.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out}")

    # ── Purity plots (only when objectness predictions are available) ──
    has_purity = br and "ttbar_purity" in next(iter(br.values()))
    if has_purity:
        # 3. Grouped bar chart: Top / W / ttbar purity in bins 6, 7, ≥8, All
        purity_bins = {
            "6 jets":  _agg_purity_bin(br, [6]),
            "7 jets":  _agg_purity_bin(br, [7]),
            "≥8 jets": _agg_purity_bin(br, ge8_keys),
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
        ax.set_title("Reconstruction Purity by Jet Multiplicity Bin")
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
        ax.set_title("Reconstruction Purity vs Jet Multiplicity")
        ax.legend()
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = run_dir / "eval_purity_vs_multiplicity.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------

def sweep_threshold_efficiencies(pred_scores, target_masks, jet_valid, target_obj, target_cls,
                                  use_probs=False, t_min=None, t_max=None, n_steps=100,
                                  priors=None):
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
    is_real = target_obj > 0.5                       # [N, Q]
    is_top  = target_cls == 1                        # [N, Q]
    is_W    = target_cls == 2                        # [N, Q]
    n_tops  = is_top.sum(axis=1)
    n_Ws    = is_W.sum(axis=1)
    has_W     = n_Ws >= 1
    both_tops = n_tops == 2
    valid     = jet_valid[:, np.newaxis, :].astype(bool)   # [N, 1, P]
    target_b  = target_masks.astype(bool)

    top_effs, W_effs, ttbar_effs = [], [], []

    for t in thresholds:
        pred_bin = binarise_predictions(
            pred_scores, jet_valid, target_cls, priors or {}, use_probs, threshold=t
        )
        mismatch        = (pred_bin != target_b) & valid       # [N, Q, P]
        errors_per_slot = mismatch.sum(axis=2)                 # [N, Q]
        slot_perfect    = errors_per_slot == 0                 # [N, Q]

        all_tops_perfect = ((~is_top) | (~is_real) | slot_perfect).all(axis=1)
        all_Ws_perfect   = ((~is_W)   | (~is_real) | slot_perfect).all(axis=1)

        n_top_correct = (is_top & is_real & slot_perfect).sum()
        n_top_total   = (is_top & is_real).sum()

        top_effs.append(float(n_top_correct / max(n_top_total, 1)))
        W_effs.append(float((all_Ws_perfect & has_W).sum() / max(has_W.sum(), 1)))
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

    # Mark the peak ttbar efficiency
    best_idx = np.argmax(ttbar_effs)
    ax.axvline(thresholds[best_idx], color="#55a868", linestyle="--", alpha=0.6,
               label=f"Best ttbar @ {thresholds[best_idx]:.3f} ({ttbar_effs[best_idx]*100:.1f}%)")

    ax.set_xlabel(x_label)
    ax.set_ylabel("Efficiency (%)")
    ax.set_title("Reconstruction Efficiency vs Binarisation Threshold")
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
        description="Compute top/W/ttbar reconstruction efficiency from saved test outputs."
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
        "--use_probs", action="store_true",
        help="Use saved sigmoid probabilities instead of raw logits (default threshold 0.5)"
    )
    parser.add_argument(
        "--threshold", type=float, default=None, metavar="T",
        help=(
            "Binarisation threshold applied to scores (logits or probs). "
            "Overrides the default (0.0 for logits, 0.5 for probs). "
            "Ignored when --prior is used for a given slot type."
        ),
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
        "--strict", action="store_true",
        help="Require objectness prediction to also mark the slot as real for efficiency metrics"
    )
    parser.add_argument(
        "--save_summary", action="store_true",
        help="Save a plain-text evaluation_summary.txt to --run_dir"
    )
    parser.add_argument(
        "--prior", nargs="+", metavar="TYPE=K", default=[],
        help=(
            "Per-type top-k binarisation prior, e.g. --prior top=3 W=2. "
            f"Valid types: {list(TYPE_IDS.keys())}. "
            "For each named slot type, select the K highest-scoring valid particles "
            "instead of thresholding."
        ),
    )
    args = parser.parse_args()

    # Parse --prior top=3 W=2 → {1: 3, 2: 2}
    priors = {}
    for item in args.prior:
        if "=" not in item:
            sys.exit(f"ERROR: --prior entries must be TYPE=K, got '{item}'")
        name, k_str = item.split("=", 1)
        if name not in TYPE_IDS:
            sys.exit(f"ERROR: unknown type '{name}' in --prior. Valid: {list(TYPE_IDS.keys())}")
        try:
            k = int(k_str)
        except ValueError:
            sys.exit(f"ERROR: K must be an integer in --prior, got '{k_str}'")
        priors[TYPE_IDS[name]] = k

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        sys.exit(f"ERROR: --run_dir does not exist or is not a directory: {run_dir}")

    pred_scores, target_masks, jet_valid, target_obj, target_cls, pred_obj, pred_type, original_mult = load_run_data(
        run_dir, args.data_file, use_probs=args.use_probs
    )

    # Collect quick per-task metrics
    quick_metrics = {}
    obj_path  = run_dir / "test_outputs_objectness.h5"
    type_path = run_dir / "test_outputs_object_type.h5"
    mask_path = run_dir / "test_outputs_mask.h5"
    if obj_path.exists():
        quick_metrics.update(compute_objectness_metrics(obj_path))
    if type_path.exists():
        quick_metrics.update(compute_type_accuracy(type_path))
    if mask_path.exists():
        quick_metrics.update(compute_mask_iou(mask_path))

    results = compute_efficiencies(
        pred_scores, target_masks, jet_valid, target_obj, target_cls,
        pred_obj=pred_obj, pred_type=pred_type, priors=priors,
        use_probs=args.use_probs, threshold=args.threshold, strict=args.strict,
        original_mult=original_mult,
    )
    results["quick_metrics"] = quick_metrics
    print_results(run_dir, results)

    if args.save_summary:
        out = run_dir / "evaluation_summary.txt"
        lines = [
            f"top_eff: {results['top_eff']:.4f}",
            f"W_eff: {results['W_eff']:.4f}",
            f"ttbar_eff: {results['ttbar_eff']:.4f}",
            f"all_eff: {results['all_eff']:.4f}",
        ]
        if results.get("obj_purity") is not None:
            lines += [
                f"obj_purity: {results['obj_purity']:.4f}",
                f"top_purity: {results['top_purity']:.4f}",
                f"W_purity: {results['W_purity']:.4f}",
                f"ttbar_purity: {results['ttbar_purity']:.4f}",
            ]
        for k, v in quick_metrics.items():
            lines.append(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
        with open(out, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"  Saved summary: {out}")

    if args.plot:
        make_plots(run_dir, results)

    if args.threshold_sweep:
        t_min, t_max = args.sweep_range if args.sweep_range else (None, None)
        print(f"\nRunning threshold sweep ({args.sweep_steps} steps)...")
        thresholds, top_effs, W_effs, ttbar_effs = sweep_threshold_efficiencies(
            pred_scores, target_masks, jet_valid, target_obj, target_cls,
            use_probs=args.use_probs,
            t_min=t_min, t_max=t_max,
            n_steps=args.sweep_steps,
            priors=priors,
        )
        plot_threshold_sweep(run_dir, thresholds, top_effs, W_effs, ttbar_effs,
                             use_probs=args.use_probs)


if __name__ == "__main__":
    main()
