"""Small event-level metrics for legal G2 decoded hypotheses."""

import torch

from .decoder import DecodedEvent, FULL_TOP


METRIC_KEYS = ("state_correct", "w_exact", "full_top_exact", "event_exact", "events", "w_events", "full_events")


def _same_pair(predicted, target: torch.Tensor) -> bool:
    if predicted is None or target.numel() != 2:
        return False
    return tuple(predicted) == tuple(sorted(int(value) for value in target.tolist()))


def score_event(decoded: DecodedEvent, targets: dict[str, torch.Tensor], row: int) -> dict[str, int]:
    """Score one event under the better of the two chain permutations."""
    state = targets["state_targets"][row]
    w = targets["w_targets"][row]
    b = targets["b_targets"][row]
    best = None
    for permutation in ((0, 1), (1, 0)):
        state_correct = 0
        w_exact = 0
        full_top_exact = 0
        chain_exact = 0
        w_events = 0
        full_events = 0
        for query, target_chain in enumerate(permutation):
            hypothesis = decoded.chains[query]
            target_state = int(state[target_chain])
            state_hit = hypothesis.state == target_state
            state_correct += int(state_hit)
            has_w = target_state != 0
            has_full = target_state == FULL_TOP
            w_hit = has_w and _same_pair(hypothesis.w_pair, w[target_chain])
            full_hit = (
                has_full
                and hypothesis.state == FULL_TOP
                and w_hit
                and hypothesis.b_jet == int(b[target_chain])
            )
            w_events += int(has_w)
            full_events += int(has_full)
            w_exact += int(w_hit)
            full_top_exact += int(full_hit)
            chain_exact += int(state_hit and (not has_w or w_hit) and (not has_full or full_hit))
        candidate = {
            "state_correct": state_correct,
            "w_exact": w_exact,
            "full_top_exact": full_top_exact,
            "event_exact": int(chain_exact == 2),
            "events": 1,
            "w_events": w_events,
            "full_events": full_events,
        }
        if best is None or (candidate["event_exact"], candidate["full_top_exact"], candidate["w_exact"]) > (
            best["event_exact"], best["full_top_exact"], best["w_exact"]
        ):
            best = candidate
    return best


def score_batch(decoded: list[DecodedEvent], targets: dict[str, torch.Tensor]) -> dict[str, int]:
    totals = {key: 0 for key in METRIC_KEYS}
    for row, event in enumerate(decoded):
        scores = score_event(event, targets, row)
        for key, value in scores.items():
            totals[key] += value
    return totals


def merge_counts(total: dict[str, int], update: dict[str, int]) -> None:
    for key in METRIC_KEYS:
        total[key] += int(update[key])


def rates(counts: dict[str, int]) -> dict[str, float]:
    events = max(1, counts["events"])
    w_events = max(1, counts["w_events"])
    full_events = max(1, counts["full_events"])
    return {
        "state_accuracy": counts["state_correct"] / (2 * events),
        "w_exact": counts["w_exact"] / w_events,
        "full_top_exact": counts["full_top_exact"] / full_events,
        "event_exact": counts["event_exact"] / events,
        "legal_selection": 1.0,
        "overlap_rate": 0.0,
        "events": counts["events"],
        "w_events": counts["w_events"],
        "full_events": counts["full_events"],
    }
