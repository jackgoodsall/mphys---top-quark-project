"""
Analysis to verify fix_matching_unique_tags.py worked correctly.

Checks:
  1. Event counts match between original and fixed files
  2. No parton branch has >1 jet index per event in the fixed file
  3. The jet kept for each fixed event is indeed the min-delta-R one
  4. Branches not in BRANCH_TRUTH_IDX are bit-for-bit identical
  5. Distribution of n-matched jets per parton before vs after
"""

import math
import numpy as np
import uproot
import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ORIG  = Path("data/topquarkreconstruction/root_data/ttbarLO_inclusive_mostof20M.root")
FIXED = Path("data/topquarkreconstruction/root_data/ttbarLO_inclusive_unique_tags.root")
PLOT_DIR = Path("src/data/analysis_plots")
SAMPLE = 500_000   # events to read for spot-checks

BRANCH_TRUTH_IDX = {
    "b_from_Wplus_jet_indices":  0,
    "Wplus_decay1_jet_indices":  2,
    "Wplus_decay2_jet_indices":  3,
    "b_from_Wminus_jet_indices": 1,
    "Wminus_decay1_jet_indices": 4,
    "Wminus_decay2_jet_indices": 5,
}

PLOT_DIR.mkdir(parents=True, exist_ok=True)


def open_sample(path, n=SAMPLE):
    f = uproot.open(path)
    reco     = f["reco;1"].arrays(["jet_eta", "jet_phi"], entry_stop=n, library="ak")
    matching = f["matching;1"].arrays(list(BRANCH_TRUTH_IDX.keys()), entry_stop=n, library="ak")
    truth    = f["truth;1"].arrays(["truth_decay_eta", "truth_decay_phi"], entry_stop=n, library="ak")
    return reco, matching, truth


def delta_r(eta1, phi1, eta2, phi2):
    dphi = (phi1 - phi2 + math.pi) % (2 * math.pi) - math.pi
    return math.sqrt((eta1 - eta2) ** 2 + dphi ** 2)


print("=" * 60)
print("1. Event count check")
print("=" * 60)
fo = uproot.open(ORIG)
ff = uproot.open(FIXED)
n_orig  = fo["reco;1"].num_entries
n_fixed = ff["reco;1"].num_entries
print(f"  Original : {n_orig:,}")
print(f"  Fixed    : {n_fixed:,}")
assert n_orig == n_fixed, "Event count mismatch!"
print("  PASS")

print()
print("=" * 60)
print(f"2. Multiplicity check (first {SAMPLE:,} events)")
print("=" * 60)
reco_o, match_o, truth_o = open_sample(ORIG)
reco_f, match_f, truth_f = open_sample(FIXED)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

all_pass = True
for ax, (branch, tidx) in zip(axes, BRANCH_TRUTH_IDX.items()):
    orig_counts  = ak.to_numpy(ak.num(match_o[branch]))
    fixed_counts = ak.to_numpy(ak.num(match_f[branch]))

    multi_orig  = int((orig_counts  > 1).sum())
    multi_fixed = int((fixed_counts > 1).sum())

    ok = multi_fixed == 0
    if not ok:
        all_pass = False
    status = "PASS" if ok else "FAIL"
    print(f"  {branch}")
    print(f"    orig  multi-matches: {multi_orig:,}")
    print(f"    fixed multi-matches: {multi_fixed:,}  [{status}]")

    bins = np.arange(0, orig_counts.max() + 2) - 0.5
    ax.hist(orig_counts,  bins=bins, alpha=0.6, label="original", color="steelblue")
    ax.hist(fixed_counts, bins=bins, alpha=0.6, label="fixed",    color="tomato")
    ax.set_title(branch.replace("_jet_indices", ""), fontsize=9)
    ax.set_xlabel("# jets matched to parton")
    ax.set_ylabel("events")
    ax.legend(fontsize=8)
    ax.set_yscale("log")

plt.suptitle("Jets matched per parton: original vs fixed", fontsize=12)
plt.tight_layout()
plt.savefig(PLOT_DIR / "multiplicity_before_after.png", dpi=120)
print(f"\n  Plot saved: {PLOT_DIR}/multiplicity_before_after.png")
print(f"  Overall multiplicity check: {'PASS' if all_pass else 'FAIL'}")

print()
print("=" * 60)
print("3. Min-delta-R correctness spot-check")
print("=" * 60)
n_checked = 0
n_wrong   = 0
wrong_examples = []

jet_eta_o = reco_o["jet_eta"].tolist()
jet_phi_o = reco_o["jet_phi"].tolist()
t_eta = truth_o["truth_decay_eta"].tolist()
t_phi = truth_o["truth_decay_phi"].tolist()

for branch, tidx in BRANCH_TRUTH_IDX.items():
    orig_idxs  = match_o[branch].tolist()
    fixed_idxs = match_f[branch].tolist()

    for ev, (oi, fi) in enumerate(zip(orig_idxs, fixed_idxs)):
        if len(oi) <= 1:
            continue   # no ambiguity — nothing to check

        n_checked += 1
        p_eta = t_eta[ev][tidx]
        p_phi = t_phi[ev][tidx]

        # Compute delta R for all original candidates
        drs = {}
        for ji in oi:
            if ji < 0 or ji >= len(jet_eta_o[ev]):
                continue
            drs[ji] = delta_r(jet_eta_o[ev][ji], jet_phi_o[ev][ji], p_eta, p_phi)

        if not drs:
            continue

        expected_best = min(drs, key=drs.get)
        got = fi[0] if fi else None

        if got != expected_best:
            n_wrong += 1
            if len(wrong_examples) < 3:
                wrong_examples.append(
                    f"    event={ev} branch={branch}: expected jet {expected_best} "
                    f"(dR={drs[expected_best]:.3f}), got jet {got}"
                )

print(f"  Multi-match events checked: {n_checked:,}")
print(f"  Wrong assignments         : {n_wrong}")
for ex in wrong_examples:
    print(ex)
print(f"  Min-dR correctness: {'PASS' if n_wrong == 0 else 'FAIL'}")

print()
print("=" * 60)
print("4. Unchanged-branch identity check (reco jet_eta first 1000 events)")
print("=" * 60)
reco_o2 = fo["reco;1"].arrays(["jet_eta"], entry_stop=1000, library="ak")
reco_f2 = ff["reco;1"].arrays(["jet_eta"], entry_stop=1000, library="ak")
flat_o = ak.to_numpy(ak.flatten(reco_o2["jet_eta"]))
flat_f = ak.to_numpy(ak.flatten(reco_f2["jet_eta"]))
identical = np.allclose(flat_o, flat_f, atol=0, rtol=0)
print(f"  reco.jet_eta identical: {'PASS' if identical else 'FAIL'}")

print()
print("=" * 60)
print("5. Fraction of events with ≥1 parton fully matched (original vs fixed)")
print("=" * 60)
for label, m in [("original", match_o), ("fixed", match_f)]:
    # event has ≥1 fully-matched parton if at least one branch has ≥1 entry
    any_match = np.zeros(SAMPLE, dtype=bool)
    all_match = np.ones(SAMPLE, dtype=bool)
    for branch in BRANCH_TRUTH_IDX:
        has = ak.to_numpy(ak.num(m[branch])) >= 1
        any_match |= has
        all_match &= has
    print(f"  {label}: any-parton matched {any_match.sum():,}  all-6 matched {all_match.sum():,}"
          f"  ({100.*all_match.sum()/SAMPLE:.2f}%)")

print()
print("Analysis complete.")
