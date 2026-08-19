"""Exact legal decoder for two hierarchical chain queries."""

from dataclasses import dataclass
from itertools import combinations
from typing import Optional, Tuple

import torch


ABSENT, W_ONLY, FULL_TOP = 0, 1, 2


@dataclass(frozen=True)
class Hypothesis:
    state: int
    w_pair: Optional[Tuple[int, int]]
    b_jet: Optional[int]
    score: float

    @property
    def jets(self) -> frozenset[int]:
        if self.w_pair is None:
            return frozenset()
        return frozenset((*self.w_pair,)) if self.b_jet is None else frozenset((*self.w_pair, self.b_jet))


@dataclass(frozen=True)
class DecodedEvent:
    chains: tuple[Hypothesis, Hypothesis]
    score: float


def _hypotheses_for_query(
    state_logits: torch.Tensor,
    w_pair_logits: torch.Tensor,
    b_extension_logits: torch.Tensor,
    valid_particles: torch.Tensor,
    state_temperature: float,
    candidate_temperature: float,
) -> list[Hypothesis]:
    valid = [index for index, is_valid in enumerate(valid_particles.tolist()) if is_valid]
    state_scores = torch.log_softmax(state_logits / state_temperature, dim=-1).tolist()
    result = [Hypothesis(ABSENT, None, None, state_scores[ABSENT])]
    for i, j in combinations(valid, 2):
        w_score = state_scores[W_ONLY] + float(w_pair_logits[i, j]) / candidate_temperature
        result.append(Hypothesis(W_ONLY, (i, j), None, w_score))
        for b in valid:
            if b in (i, j):
                continue
            score = w_score - state_scores[W_ONLY] + state_scores[FULL_TOP]
            score += float(b_extension_logits[b, i, j]) / candidate_temperature
            result.append(Hypothesis(FULL_TOP, (i, j), b, score))
    return result


def decode_event(
    state_logits: torch.Tensor,
    w_pair_logits: torch.Tensor,
    b_extension_logits: torch.Tensor,
    valid_particles: torch.Tensor,
    state_temperature: float = 1.0,
    candidate_temperature: float = 1.0,
) -> DecodedEvent:
    """Decode two queries while enforcing disjoint legal chains."""
    if state_logits.shape[0] != 2:
        raise ValueError("G2 scaffold currently expects exactly two chain queries")
    if valid_particles.ndim != 1 or valid_particles.shape[0] != w_pair_logits.shape[-1]:
        raise ValueError("valid_particles must be [N] and match candidate logits")
    if state_temperature <= 0 or candidate_temperature <= 0:
        raise ValueError("temperatures must be positive")

    candidates = [
        _hypotheses_for_query(
            state_logits[q],
            w_pair_logits[q],
            b_extension_logits[q],
            valid_particles,
            state_temperature,
            candidate_temperature,
        )
        for q in range(2)
    ]
    # For each left hypothesis, the first compatible right hypothesis in score
    # order is optimal. This keeps the exact global decoder from doing an
    # O(number_of_candidates^2) scan for every event.
    right_by_score = sorted(candidates[1], key=lambda item: item.score, reverse=True)
    best = None
    for left in candidates[0]:
        for right in right_by_score:
            if left.jets & right.jets:
                continue
            score = left.score + right.score
            if best is None or score > best.score:
                best = DecodedEvent((left, right), score)
            break
    if best is None:  # absent/absent always makes this unreachable
        raise RuntimeError("no legal pair of chain hypotheses")
    return best


def decode_batch(
    outputs: dict[str, torch.Tensor],
    valid_particles: torch.Tensor,
    state_temperature: float = 1.0,
    candidate_temperature: float = 1.0,
) -> list[DecodedEvent]:
    """Decode a batch of target-free scorer outputs."""
    return [
        decode_event(
            outputs["state_logits"][batch],
            outputs["w_pair_logits"][batch],
            outputs["b_extension_logits"][batch],
            valid_particles[batch],
            state_temperature,
            candidate_temperature,
        )
        for batch in range(valid_particles.shape[0])
    ]
