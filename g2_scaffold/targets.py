"""Convert the existing top/W masks into G2 hierarchical targets."""

import torch
from typing import Optional

from .decoder import ABSENT, FULL_TOP, W_ONLY


CLASS_TOP = 1
CLASS_W = 2


def _event_description(rows, event_ids):
    if event_ids is None:
        return f"batch rows {rows}"
    values = event_ids.detach().cpu().reshape(-1).tolist()
    return "batch rows " + ", ".join(f"{row} (event_id={values[row]})" for row in rows)


def _raise_target_error(message, rows, event_ids=None):
    raise ValueError(f"{message}; {_event_description(rows, event_ids)}")


def targets_to_g2(
    targets: dict[str, torch.Tensor], event_ids: Optional[torch.Tensor] = None
) -> dict[str, torch.Tensor]:
    """Build censored three-state targets from the existing collated targets.

    The current data contract stores objects as ``[top0, top1, W0, W1]``.
    ``classes`` is used when present to verify/find those slots; the fixed
    layout remains the fallback for the synthetic smoke path.
    """
    missing = {"jet_mask_true", "jet_valid_mask"} - set(targets)
    if missing:
        raise ValueError(f"G2 target conversion missing keys: {sorted(missing)}")
    masks = targets["jet_mask_true"]
    valid_particles = targets["jet_valid_mask"].bool()
    if masks.ndim != 3 or valid_particles.ndim != 2 or masks.shape[0] != valid_particles.shape[0] or masks.shape[2] != valid_particles.shape[1]:
        raise ValueError("G2 targets have incompatible mask/particle shapes; expected masks [B,T,P] and valid particles [B,P]")
    object_valid = targets.get(
        "target_valid_mask",
        torch.ones(masks.shape[:2], dtype=torch.bool, device=masks.device),
    ).bool()
    classes = targets.get("classes")
    B, T, P = masks.shape

    if classes is None:
        if T != 4:
            _raise_target_error("G2 requires exactly four target slots when classes are absent", list(range(B)), event_ids)
        top_slots, w_slots = [0, 1], [2, 3]
    else:
        if classes.shape != (B, T):
            raise ValueError("G2 classes must have shape [B,T] matching jet masks")
        top_counts = classes.eq(CLASS_TOP).sum(dim=1)
        w_counts = classes.eq(CLASS_W).sum(dim=1)
        bad = torch.where((top_counts != 2) | (w_counts != 2))[0].tolist()
        if bad:
            _raise_target_error("G2 expects exactly two top and two W target slots per event", bad, event_ids)
        if not torch.all(classes == classes[0]):
            _raise_target_error("G2 target class layout changes between events", list(range(B)), event_ids)
        top_slots = torch.where(classes[0] == CLASS_TOP)[0].tolist()
        w_slots = torch.where(classes[0] == CLASS_W)[0].tolist()

    top_masks = masks[:, top_slots].gt(0.5) & valid_particles[:, None, :]
    w_masks = masks[:, w_slots].gt(0.5) & valid_particles[:, None, :]
    top_valid = object_valid[:, top_slots]
    w_valid = object_valid[:, w_slots]

    off_particle = masks.gt(0.5) & ~valid_particles[:, None, :]
    bad = torch.where(off_particle.any(dim=(1, 2)))[0].tolist()
    if bad:
        _raise_target_error("G2 target mask marks a padded/invalid particle; fix the eligibility audit or source targets", bad, event_ids)

    w_counts = w_masks.sum(dim=-1)
    bad = torch.where(w_valid & (w_counts != 2))[0].tolist()
    if bad:
        _raise_target_error("valid G2 W targets must contain exactly two particles (masks are never truncated)", bad, event_ids)

    full = top_valid & w_valid
    bad = torch.where(full & (w_masks & ~top_masks).any(dim=-1))[0].tolist()
    if bad:
        _raise_target_error("valid full-top G2 targets must contain the complete W pair inside the top mask", bad, event_ids)
    b_masks = top_masks & ~w_masks
    b_counts = b_masks.sum(dim=-1)
    bad = torch.where(full & (b_counts != 1))[0].tolist()
    if bad:
        _raise_target_error("valid full-top G2 targets must contain exactly one b extension", bad, event_ids)

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
