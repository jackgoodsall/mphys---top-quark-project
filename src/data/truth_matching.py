"""Truth-parton to reconstructed-jet matching utilities."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Matchability:
    """Exclusive matchability summary for a declared set of partons."""

    per_parton: np.ndarray
    matched_count: int
    required_count: int
    any_matchable: bool
    fully_matchable: bool


@dataclass(frozen=True)
class StageMatchability:
    """Matchability before and after an acceptance or truncation stage."""

    pre: Matchability
    post: Matchability
    lost_partons: np.ndarray


@dataclass(frozen=True)
class TruthMatchResult:
    """Complete matching record for one event.

    ``incidence`` contains every geometrically admissible pair.  When
    ``jet_mask`` is supplied, ``stage_incidence`` additionally applies it.
    Second-closest-local fields describe the second-closest stage-eligible jet
    for each parton, independent of the globally selected assignment.
    """

    radius: float
    distances: np.ndarray
    incidence: np.ndarray
    stage_incidence: np.ndarray
    eligible_jets: np.ndarray
    selected_jet: np.ndarray
    selected_distance: np.ndarray
    second_closest_local_jet: np.ndarray
    second_closest_local_distance: np.ndarray
    parton_ambiguous: np.ndarray
    parton_collision: np.ndarray
    jet_collision: np.ndarray

    @property
    def has_ambiguity(self) -> bool:
        return bool(self.parton_ambiguous.any())

    @property
    def has_collision(self) -> bool:
        return bool(self.jet_collision.any())

    def matchability(self, required_partons=None) -> Matchability:
        return summarize_matchability(self.selected_jet, required_partons)


def _coordinates(eta, phi, name):
    eta = np.asarray(eta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    if eta.ndim != 1 or phi.ndim != 1 or eta.shape != phi.shape:
        raise ValueError(f"{name} eta and phi must be equal-length 1D arrays")
    if not np.isfinite(eta).all() or not np.isfinite(phi).all():
        raise ValueError(f"{name} eta and phi must be finite")
    return eta, phi


def pairwise_delta_r(parton_eta, parton_phi, jet_eta, jet_phi):
    """Return the vectorized pairwise ΔR matrix with phi wrapped to [-pi, pi)."""

    parton_eta, parton_phi = _coordinates(parton_eta, parton_phi, "parton")
    jet_eta, jet_phi = _coordinates(jet_eta, jet_phi, "jet")
    deta = parton_eta[:, None] - jet_eta[None, :]
    dphi = (parton_phi[:, None] - jet_phi[None, :] + np.pi) % (2 * np.pi) - np.pi
    return np.hypot(deta, dphi)


def _lexicographic_tie_cost(n_rows, n_cols):
    """Infinitesimal base-N digits selecting the lowest columns by row order."""

    if not n_rows or not n_cols:
        return np.empty((n_rows, n_cols), dtype=np.float64)
    base = float(n_cols + 1)
    columns = np.arange(n_cols, dtype=np.float64)
    return np.vstack([columns / base ** (row + 1) for row in range(n_rows)])


def match_truth_to_jets(
    parton_eta,
    parton_phi,
    jet_eta,
    jet_phi,
    radius=0.4,
    jet_mask=None,
):
    """Exactly match one event, maximizing cardinality then minimizing total ΔR.

    Each parton and jet is used at most once.  Pairs outside ``radius`` or
    excluded by ``jet_mask`` cannot be selected.  Equal-cost assignments are
    resolved lexicographically by parton order and then jet index.
    """

    radius = float(radius)
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("radius must be finite and non-negative")

    parton_eta, parton_phi = _coordinates(parton_eta, parton_phi, "parton")
    jet_eta, jet_phi = _coordinates(jet_eta, jet_phi, "jet")
    n_partons, n_jets = len(parton_eta), len(jet_eta)

    if jet_mask is None:
        eligible_jets = np.ones(n_jets, dtype=bool)
    else:
        eligible_jets = np.asarray(jet_mask, dtype=bool)
        if eligible_jets.shape != (n_jets,):
            raise ValueError("jet_mask must have one entry per jet")

    distances = pairwise_delta_r(parton_eta, parton_phi, jet_eta, jet_phi)
    incidence = distances <= radius
    stage_incidence = incidence & eligible_jets[None, :]

    selected_jet = np.full(n_partons, -1, dtype=np.int64)
    selected_distance = np.full(n_partons, np.inf, dtype=np.float64)
    if n_partons:
        # One private dummy column per parton permits unmatched rows.  Its cost
        # dominates every possible sum of admissible distances, so cardinality
        # is optimized before distance.
        unmatched_cost = (n_partons + 1) * max(radius, 1.0)
        blocked_cost = (n_partons + 2) * unmatched_cost
        cost = np.full((n_partons, n_jets + n_partons), blocked_cost)
        cost[:, :n_jets] = np.where(stage_incidence, distances, blocked_cost)
        cost[np.arange(n_partons), n_jets + np.arange(n_partons)] = unmatched_cost

        tie = _lexicographic_tie_cost(*cost.shape)
        # A sub-ULP perturbation rounds away. Use a declared 1e-12-scale term:
        # far below detector precision, but large enough to make exact ties
        # reproducibly lexicographic across scipy versions.
        tie_epsilon = max(1e-12, np.finfo(np.float64).eps * blocked_cost * 64)
        cost += tie_epsilon * tie
        rows, columns = linear_sum_assignment(cost)
        matched = columns < n_jets
        selected_jet[rows[matched]] = columns[matched]
        selected_distance[rows[matched]] = distances[rows[matched], columns[matched]]

    second_closest_local_jet = np.full(n_partons, -1, dtype=np.int64)
    second_closest_local_distance = np.full(n_partons, np.inf, dtype=np.float64)
    for parton in range(n_partons):
        candidates = np.flatnonzero(stage_incidence[parton])
        if candidates.size >= 2:
            order = np.lexsort((candidates, distances[parton, candidates]))
            second = candidates[order[1]]
            second_closest_local_jet[parton] = second
            second_closest_local_distance[parton] = distances[parton, second]

    candidate_counts = stage_incidence.sum(axis=1)
    jet_candidate_counts = stage_incidence.sum(axis=0)
    jet_collision = jet_candidate_counts > 1
    parton_collision = (
        (stage_incidence & jet_collision[None, :]).any(axis=1)
        if n_jets
        else np.zeros(n_partons, dtype=bool)
    )

    return TruthMatchResult(
        radius=radius,
        distances=distances,
        incidence=incidence,
        stage_incidence=stage_incidence,
        eligible_jets=eligible_jets,
        selected_jet=selected_jet,
        selected_distance=selected_distance,
        second_closest_local_jet=second_closest_local_jet,
        second_closest_local_distance=second_closest_local_distance,
        parton_ambiguous=candidate_counts > 1,
        parton_collision=parton_collision,
        jet_collision=jet_collision,
    )


def summarize_matchability(selected_jet, required_partons=None):
    """Summarize exclusive matchability, optionally for required parton slots."""

    selected_jet = np.asarray(selected_jet)
    if selected_jet.ndim != 1:
        raise ValueError("selected_jet must be a 1D array")
    matched = selected_jet >= 0
    if required_partons is None:
        required = np.ones(selected_jet.shape, dtype=bool)
    else:
        required = np.asarray(required_partons, dtype=bool)
        if required.shape != selected_jet.shape:
            raise ValueError("required_partons must match selected_jet")
    per_parton = matched & required
    required_count = int(required.sum())
    matched_count = int(per_parton.sum())
    return Matchability(
        per_parton=per_parton,
        matched_count=matched_count,
        required_count=required_count,
        any_matchable=bool(matched_count),
        fully_matchable=matched_count == required_count,
    )


def compare_stage_matchability(pre, post, required_partons=None):
    """Compare exclusive parton matchability before and after a pipeline stage."""

    pre_summary = pre.matchability(required_partons)
    post_summary = post.matchability(required_partons)
    if pre_summary.per_parton.shape != post_summary.per_parton.shape:
        raise ValueError("pre and post results must contain the same partons")
    return StageMatchability(
        pre=pre_summary,
        post=post_summary,
        lost_partons=pre_summary.per_parton & ~post_summary.per_parton,
    )
