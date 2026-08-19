"""Minimal G2 losses with exact two-chain permutation handling."""

from typing import Optional

import torch
import torch.nn.functional as F


def _pair_scores(logits: torch.Tensor) -> torch.Tensor:
    """Return unordered pair scores and their (i, j) ordering."""
    n = logits.shape[-1]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if not pairs:
        raise ValueError("at least two particles are required")
    return logits[..., [i for i, _ in pairs], [j for _, j in pairs]], pairs


def _query_loss(
    state_logits: torch.Tensor,
    w_pair_logits: torch.Tensor,
    b_extension_logits: torch.Tensor,
    state_target: torch.Tensor,
    w_target: torch.Tensor,
    b_target: torch.Tensor,
    valid_particles: torch.Tensor,
) -> torch.Tensor:
    """Per-event loss for one query; missing components are censored."""
    state_loss = F.cross_entropy(state_logits, state_target, reduction="none")
    total = state_loss

    w_scores, pairs = _pair_scores(w_pair_logits)
    pair_valid = torch.stack(
        [valid_particles[:, i] & valid_particles[:, j] for i, j in pairs], dim=-1
    )
    w_scores = w_scores.masked_fill(~pair_valid, -torch.inf)
    has_w = (state_target != 0) & (w_target >= 0).all(dim=-1)
    if has_w.any():
        target_pair = w_target[has_w]
        pair_to_index = {pair: index for index, pair in enumerate(pairs)}
        pair_indices = []
        for pair in target_pair.tolist():
            i, j = sorted((int(pair[0]), int(pair[1])))
            if (i, j) not in pair_to_index:
                raise ValueError("W target must contain two distinct particle indices")
            pair_indices.append(pair_to_index[(i, j)])
        w_losses = torch.zeros_like(total)
        w_losses[has_w] = F.cross_entropy(
            w_scores[has_w], torch.as_tensor(pair_indices, device=w_scores.device), reduction="none"
        )
        total = total + w_losses

    has_b = (state_target == 2) & (b_target >= 0) & (w_target >= 0).all(dim=-1)
    if has_b.any():
        b_losses = torch.zeros_like(total)
        for row in torch.where(has_b)[0].tolist():
            i, j = sorted((int(w_target[row, 0]), int(w_target[row, 1])))
            b_scores = b_extension_logits[row, :, i, j].clone()
            b_scores[~valid_particles[row]] = -torch.inf
            b_scores[[i, j]] = -torch.inf
            b_losses[row] = F.cross_entropy(
                b_scores.unsqueeze(0), b_target[row].reshape(1), reduction="none"
            )[0]
        total = total + b_losses

    return total


def hierarchical_loss(
    outputs: dict[str, torch.Tensor],
    state_targets: torch.Tensor,
    w_targets: torch.Tensor,
    b_targets: torch.Tensor,
    mode: str = "hard_min",
    temperature: float = 1.0,
    valid_particles: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute a two-chain permutation-invariant G2 loss.

    State labels are 0=absent, 1=W-only, 2=full-top.  A W target is shaped
    ``[B, 2, 2]`` and a b target ``[B, 2]``; censored components use ``-1``.
    ``hard_min`` selects the better chain assignment, while ``marginal`` is a
    normalized soft minimum over the two assignments.
    """
    if mode not in {"hard_min", "marginal"}:
        raise ValueError("mode must be 'hard_min' or 'marginal'")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    state_logits = outputs["state_logits"]
    w_pair_logits = outputs["w_pair_logits"]
    b_extension_logits = outputs["b_extension_logits"]
    if state_logits.shape[1] != 2:
        raise ValueError("G2 scaffold currently expects exactly two chain queries")
    if valid_particles is None:
        valid_particles = torch.ones(
            w_pair_logits.shape[0], w_pair_logits.shape[-1],
            dtype=torch.bool, device=w_pair_logits.device
        )
    if valid_particles.shape != (w_pair_logits.shape[0], w_pair_logits.shape[-1]):
        raise ValueError("valid_particles must be [B,N]")

    losses = []
    for permutation in ((0, 1), (1, 0)):
        query_losses = []
        for query, target_chain in enumerate(permutation):
            query_losses.append(
                _query_loss(
                    state_logits[:, query],
                    w_pair_logits[:, query],
                    b_extension_logits[:, query],
                    state_targets[:, target_chain],
                    w_targets[:, target_chain],
                    b_targets[:, target_chain],
                    valid_particles,
                )
            )
        losses.append(torch.stack(query_losses, dim=0).sum(dim=0))

    assignment_losses = torch.stack(losses, dim=0)
    if mode == "hard_min":
        return assignment_losses.min(dim=0).values.mean()
    return (
        -temperature * torch.logsumexp(-assignment_losses / temperature, dim=0)
        + temperature * torch.log(torch.tensor(2.0, device=assignment_losses.device))
    ).mean()
