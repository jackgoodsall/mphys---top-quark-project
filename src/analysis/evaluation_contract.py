"""Target-free, exact-S2 evaluation primitives for two-chain reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np


@dataclass(frozen=True)
class S2Score:
    permutation: np.ndarray
    top_exact: np.ndarray
    w_exact: np.ndarray
    chain_exact: np.ndarray
    identifiable: np.ndarray
    fully_matchable: np.ndarray
    event_exact: np.ndarray


def _validate_masks(pred_top, pred_w, truth_top, truth_w, jet_valid, top_valid, w_valid):
    arrays = [np.asarray(value) for value in (pred_top, pred_w, truth_top, truth_w)]
    if any(value.ndim != 3 for value in arrays) or any(value.shape != arrays[0].shape for value in arrays):
        raise ValueError("prediction/truth masks must share shape [events, 2, particles]")
    if arrays[0].shape[1] != 2:
        raise ValueError("exact S2 scoring requires two chain slots")
    batch, _, particles = arrays[0].shape
    jet_valid = np.asarray(jet_valid, dtype=bool)
    top_valid = np.asarray(top_valid, dtype=bool)
    w_valid = np.asarray(w_valid, dtype=bool)
    if jet_valid.shape != (batch, particles) or top_valid.shape != (batch, 2) or w_valid.shape != (batch, 2):
        raise ValueError("invalid jet or chain validity shape")
    return [value.astype(bool) for value in arrays], jet_valid, top_valid, w_valid


def score_s2(pred_top, pred_w, truth_top, truth_w, jet_valid, top_valid, w_valid):
    """Score raw learned-query predictions against truth under both S2 elements."""
    (pred_top, pred_w, truth_top, truth_w), jets, top_valid, w_valid = _validate_masks(
        pred_top, pred_w, truth_top, truth_w, jet_valid, top_valid, w_valid
    )
    permutations = np.array([[0, 1], [1, 0]])
    candidate_cost = []
    candidate_top = []
    candidate_w = []
    candidate_chain = []
    for permutation in permutations:
        aligned_top = pred_top[:, permutation]
        aligned_w = pred_w[:, permutation]
        top_mismatch = ((aligned_top != truth_top) & jets[:, None, :]).any(axis=-1)
        w_mismatch = ((aligned_w != truth_w) & jets[:, None, :]).any(axis=-1)
        top_exact = ~top_mismatch
        w_exact = ~w_mismatch
        # Missing components are censored. The state head is evaluated separately.
        cost = ((~top_exact) & top_valid).sum(axis=1) + ((~w_exact) & w_valid).sum(axis=1)
        candidate_cost.append(cost)
        candidate_top.append(top_exact)
        candidate_w.append(w_exact)
        candidate_chain.append(
            (~top_valid | top_exact) & (~w_valid | w_exact) & (top_valid | w_valid)
        )
    chain_candidates = np.stack(candidate_chain, axis=1)
    identifiable = top_valid | w_valid
    fully_matchable = (top_valid & w_valid).all(axis=1)
    event_candidates = fully_matchable[:, None] & chain_candidates.all(axis=2)
    partial_event_candidates = np.all(
        ~identifiable[:, None, :] | chain_candidates, axis=2
    )
    component_exact = -np.stack(candidate_cost, axis=1)
    # Optimize the declared event/chain metrics before component-level tie
    # breaking. This prevents an equal component cost from undercounting M3/M4.
    rank = (
        event_candidates.astype(np.int64) * 1_000_000
        + partial_event_candidates.astype(np.int64) * 100_000
        + chain_candidates.sum(axis=2) * 1_000
        + component_exact
    )
    best = np.argmax(rank, axis=1)
    rows = np.arange(len(best))
    top_exact = np.stack(candidate_top, axis=1)[rows, best]
    w_exact = np.stack(candidate_w, axis=1)[rows, best]
    chain_exact = (~top_valid | top_exact) & (~w_valid | w_exact) & identifiable
    event_exact = fully_matchable & chain_exact.all(axis=1)
    return S2Score(best, top_exact, w_exact, chain_exact, identifiable, fully_matchable, event_exact)


def wilson_interval(numerator, denominator, z=1.959963984540054):
    if denominator < 0 or numerator < 0 or numerator > denominator:
        raise ValueError("invalid binomial counts")
    if denominator == 0:
        return np.nan, np.nan
    p = numerator / denominator
    scale = 1 + z * z / denominator
    centre = (p + z * z / (2 * denominator)) / scale
    half = z * sqrt((p * (1 - p) + z * z / (4 * denominator)) / denominator) / scale
    return centre - half, centre + half


def metric_row(name, selected, successes, event_ids):
    selected = np.asarray(selected, dtype=bool)
    successes = np.asarray(successes, dtype=bool)
    event_ids = np.asarray(event_ids, dtype=np.uint64)
    if selected.shape != successes.shape or selected.shape != event_ids.shape:
        raise ValueError("metric masks and event IDs must align")
    denominator_ids = event_ids[selected]
    numerator_ids = event_ids[selected & successes]
    low, high = wilson_interval(len(numerator_ids), len(denominator_ids))
    return {
        "metric": name,
        "numerator": len(numerator_ids),
        "denominator": len(denominator_ids),
        "efficiency": len(numerator_ids) / len(denominator_ids) if len(denominator_ids) else np.nan,
        "wilson_low": low,
        "wilson_high": high,
        "numerator_event_ids": numerator_ids,
        "denominator_event_ids": denominator_ids,
    }


def minimal_scorecard(score: S2Score, event_ids, selected=None):
    event_ids = np.asarray(event_ids, dtype=np.uint64)
    selected = np.ones(len(event_ids), dtype=bool) if selected is None else np.asarray(selected, dtype=bool)
    at_least_one = score.identifiable.any(axis=1)
    all_identifiable_exact = np.all(~score.identifiable | score.chain_exact, axis=1)
    rows = [
        metric_row("M2", score.fully_matchable, score.event_exact, event_ids),
        metric_row("M3", at_least_one, all_identifiable_exact, event_ids),
        metric_row("M5", selected, score.event_exact, event_ids),
    ]
    chain_ids = np.repeat(event_ids, 2) * np.uint64(2) + np.tile(np.arange(2, dtype=np.uint64), len(event_ids))
    rows.append(metric_row("M4", score.identifiable.ravel(), score.chain_exact.ravel(), chain_ids))
    return rows


def stratified_scorecard(score: S2Score, event_ids, njets, selected=None):
    """Frozen M2--M5 scorecard for 6, 7, >=8, and inclusive populations."""
    event_ids = np.asarray(event_ids, dtype=np.uint64)
    njets = np.asarray(njets)
    selected = np.ones(len(event_ids), dtype=bool) if selected is None else np.asarray(selected, bool)
    if njets.shape != event_ids.shape or selected.shape != event_ids.shape:
        raise ValueError("multiplicity/selection arrays must align with event IDs")
    bins = {
        "6": njets == 6,
        "7": njets == 7,
        "ge8": njets >= 8,
        "inclusive": np.ones(len(event_ids), dtype=bool),
    }
    rows = []
    for label, in_bin in bins.items():
        at_least_one = score.identifiable.any(axis=1)
        all_identifiable_exact = np.all(~score.identifiable | score.chain_exact, axis=1)
        bin_rows = [
            metric_row("M2", in_bin & score.fully_matchable, score.event_exact, event_ids),
            metric_row("M3", in_bin & at_least_one, all_identifiable_exact, event_ids),
            metric_row("M5", in_bin & selected, score.event_exact, event_ids),
        ]
        chain_ids = np.repeat(event_ids, 2) * np.uint64(2) + np.tile(
            np.arange(2, dtype=np.uint64), len(event_ids)
        )
        bin_rows.append(metric_row(
            "M4", np.repeat(in_bin, 2) & score.identifiable.ravel(),
            score.chain_exact.ravel(), chain_ids,
        ))
        for row in bin_rows:
            row["njets_bin"] = label
        rows.extend(bin_rows)
    return rows


def assert_aligned_event_ids(**artifacts):
    reference_name = None
    reference = None
    for name, values in artifacts.items():
        values = np.asarray(values, dtype=np.uint64)
        if values.ndim != 1 or len(np.unique(values)) != len(values):
            raise ValueError(f"{name} event IDs are not one-dimensional and unique")
        if reference is None:
            reference_name, reference = name, values
        elif not np.array_equal(reference, values):
            raise ValueError(f"event ID mismatch between {reference_name} and {name}")
    return reference
