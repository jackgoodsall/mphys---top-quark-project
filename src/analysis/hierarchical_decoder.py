"""Exact target-free decoder for absent, W-only, and full-top chains."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ABSENT, W_ONLY, FULL_TOP = 0, 1, 2


@dataclass(frozen=True)
class HierarchicalDecode:
    state: np.ndarray
    top_mask: np.ndarray
    w_mask: np.ndarray
    score: np.ndarray
    margin: np.ndarray


def _chain_candidates(state_score, w_pair_score, b_extension_score, valid):
    valid_jets = np.flatnonzero(valid)
    candidates = [(float(state_score[ABSENT]), ABSENT, (), -1, np.uint64(0))]
    for a, left in enumerate(valid_jets):
        for right in valid_jets[a + 1:]:
            w_score = float(w_pair_score[left, right])
            w_bits = (np.uint64(1) << np.uint64(left)) | (np.uint64(1) << np.uint64(right))
            candidates.append((float(state_score[W_ONLY]) + w_score, W_ONLY,
                               (int(left), int(right)), -1, w_bits))
            for b in valid_jets:
                if b == left or b == right:
                    continue
                bits = w_bits | (np.uint64(1) << np.uint64(b))
                candidates.append((float(state_score[FULL_TOP]) + w_score
                                   + float(b_extension_score[b, left, right]),
                                   FULL_TOP, (int(left), int(right)), int(b), bits))
    return candidates


def decode_hierarchical(state_scores, w_pair_scores, b_extension_scores, jet_valid):
    """Exactly maximize the additive two-query energy over the declared support.

    The signature deliberately contains no targets or truth multiplicity.  W
    daughters are unordered; full-top candidates add one distinct b jet; the
    two decoded chains are jet-disjoint.
    """
    state_scores = np.asarray(state_scores, dtype=np.float64)
    w_pair_scores = np.asarray(w_pair_scores, dtype=np.float64)
    b_extension_scores = np.asarray(b_extension_scores, dtype=np.float64)
    jet_valid = np.asarray(jet_valid, dtype=bool)
    if state_scores.ndim != 3 or state_scores.shape[1:] != (2, 3):
        raise ValueError("state_scores must have shape [events, 2, 3]")
    batch = state_scores.shape[0]
    particles = jet_valid.shape[1] if jet_valid.ndim == 2 else -1
    if particles > 63:
        raise ValueError("exact bitset decoder supports at most 63 particles")
    if w_pair_scores.shape != (batch, 2, particles, particles):
        raise ValueError("w_pair_scores must have shape [events, 2, particles, particles]")
    if b_extension_scores.shape != (batch, 2, particles, particles, particles):
        raise ValueError("b_extension_scores has the wrong shape")
    if not all(np.isfinite(value).all() for value in (state_scores, w_pair_scores, b_extension_scores)):
        raise ValueError("decoder scores must be finite calibrated log scores")

    states = np.zeros((batch, 2), dtype=np.uint8)
    top = np.zeros((batch, 2, particles), dtype=bool)
    w = np.zeros_like(top)
    scores = np.empty(batch, dtype=np.float64)
    margins = np.empty(batch, dtype=np.float64)
    for event in range(batch):
        chains = [
            _chain_candidates(
                state_scores[event, query], w_pair_scores[event, query],
                b_extension_scores[event, query], jet_valid[event],
            )
            for query in range(2)
        ]
        best = [(-np.inf, None), (-np.inf, None)]
        right_scores = np.asarray([candidate[0] for candidate in chains[1]])
        right_bits = np.asarray([candidate[4] for candidate in chains[1]], dtype=np.uint64)
        for left in chains[0]:
            legal_indices = np.flatnonzero((right_bits & left[4]) == 0)
            if legal_indices.size == 0:
                continue
            legal_scores = right_scores[legal_indices]
            take = min(2, legal_indices.size)
            local_best = np.argpartition(legal_scores, -take)[-take:]
            local_best = local_best[np.argsort(legal_scores[local_best])[::-1]]
            for local_index in local_best:
                right = chains[1][int(legal_indices[local_index])]
                candidate = (left[0] + right[0], (left, right))
                if candidate[0] > best[0][0]:
                    best[1], best[0] = best[0], candidate
                elif candidate[0] > best[1][0]:
                    best[1] = candidate
        scores[event] = best[0][0]
        margins[event] = best[0][0] - best[1][0]
        for query, candidate in enumerate(best[0][1]):
            _, state, pair, b, _ = candidate
            states[event, query] = state
            if state >= W_ONLY:
                w[event, query, list(pair)] = True
                top[event, query, list(pair)] = True
            if state == FULL_TOP:
                top[event, query, b] = True
    return HierarchicalDecode(states, top, w, scores, margins)
