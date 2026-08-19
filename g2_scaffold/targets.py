"""Convert the existing top/W masks into G2 hierarchical targets."""

import torch

from .decoder import ABSENT, FULL_TOP, W_ONLY


CLASS_TOP = 1
CLASS_W = 2


def targets_to_g2(targets: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Build censored three-state targets from the existing collated targets.

    The current data contract stores objects as ``[top0, top1, W0, W1]``.
    ``classes`` is used when present to verify/find those slots; the fixed
    layout remains the fallback for the synthetic smoke path.
    """
    masks = targets["jet_mask_true"]
    valid_particles = targets["jet_valid_mask"].bool()
    object_valid = targets.get(
        "target_valid_mask",
        torch.ones(masks.shape[:2], dtype=torch.bool, device=masks.device),
    ).bool()
    classes = targets.get("classes")
    B, T, P = masks.shape

    if classes is None:
        top_slots, w_slots = list(range(min(2, T))), list(range(2, min(4, T)))
    else:
        top_slots = torch.where(classes[0] == CLASS_TOP)[0].tolist()
        w_slots = torch.where(classes[0] == CLASS_W)[0].tolist()
    if len(top_slots) != 2 or len(w_slots) != 2:
        raise ValueError("G2 expects two top and two W target slots per event")

    top_masks = masks[:, top_slots].gt(0.5) & valid_particles[:, None, :]
    w_masks = masks[:, w_slots].gt(0.5) & valid_particles[:, None, :]
    top_valid = object_valid[:, top_slots]
    w_valid = object_valid[:, w_slots]

    w_counts = w_masks.sum(dim=-1)
    if torch.any(w_valid & (w_counts != 2)):
        raise ValueError("valid G2 W targets must contain exactly two particles")

    full = top_valid & w_valid
    b_masks = top_masks & ~w_masks
    b_counts = b_masks.sum(dim=-1)
    if torch.any(full & (b_counts != 1)):
        raise ValueError("valid full-top G2 targets must contain exactly one b extension")

    # top-k only supplies placeholders for censored states; those entries are
    # ignored by the loss because their state is absent.
    w_targets = torch.topk(w_masks.to(torch.float32), k=2, dim=-1).indices.long()
    w_targets = w_targets.sort(dim=-1).values
    b_targets = torch.topk(b_masks.to(torch.float32), k=1, dim=-1).indices[..., 0].long()
    w_targets = w_targets.masked_fill(~w_valid[..., None], -1)
    b_targets = b_targets.masked_fill(~full, -1)

    states = torch.full((B, 2), ABSENT, dtype=torch.long, device=masks.device)
    states[w_valid & ~top_valid] = W_ONLY
    states[full] = FULL_TOP
    return {
        "state_targets": states,
        "w_targets": w_targets,
        "b_targets": b_targets,
        "valid_particles": valid_particles,
    }
