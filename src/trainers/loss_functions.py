import torch.nn as nn
import torch

import torch
import vector

def set_invariant_loss(output: torch.Tensor, targets: torch.Tensor, reduction: str = "mean"):
    """
    Event-wise set/permutation-invariant MSE loss along axis=1 (particles).
    For each event (batch element), choose the lower loss between
    the original targets and the targets flipped along axis 1.

    Args:
        output:  shape (B, P, ...)
        targets: shape (B, P, ...) matching output
        reduction: "mean" | "sum" | "none"
    Returns:
        scalar if reduction != "none", else per-event losses of shape (B,)
    """
    # Per-element squared errors
    l_orig = (output - targets)**2
    l_flip = (output - targets.flip(1).contiguous())**2

    # Reduce over all non-batch dims to get per-event losses
    reduce_dims = tuple(range(1, output.ndim))
    per_event_orig = l_orig.mean(dim=reduce_dims)
    per_event_flip = l_flip.mean(dim=reduce_dims)

    # Event-wise minimum
    per_event = torch.minimum(per_event_orig, per_event_flip)

    if reduction == "mean":
        return per_event.mean()
    elif reduction == "sum":
        return per_event.sum()
    elif reduction == "none":
        return per_event
    else:
        raise ValueError(f"Unknown reduction: {reduction}")
    

def W_boson_loss_function(
    outputs: dict,
    targets: dict,
    top_weight: float = 0.6,
    W_weight: float = 0.4,
    reduction: str = "mean",
):
    """
    Combined permutation-invariant loss for tops and W bosons.

    Expects:
        outputs["top"], targets["top"]: tensors of shape (B, P_top, ...)
        outputs["W"],   targets["W"]:   tensors of shape (B, P_W, ...)

    Returns:
        scalar if reduction != "none", else per-event losses of shape (B,)
    """

    # Per-event losses for each species (no reduction yet)
    top_loss_per_event = set_invariant_loss(
        outputs["top"], targets["top"], reduction = "none"
    )
    W_loss_per_event = set_invariant_loss(
        outputs["W"], targets["W"], reduction = "none"
    )

    # Weighted combination per event
    combined_per_event =  top_loss_per_event * top_weight + W_loss_per_event * W_weight
    # Final reduction
    if reduction == "mean":
        return combined_per_event.mean()
    elif reduction == "sum":
        return combined_per_event.sum()
    elif reduction == "none":
        return combined_per_event
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

def invariant_mass_loss_function(
        output_kinematics,
        target_invariant_mass,
        reverse_output_transformers,
        invariant_mass_scalers, 
):
    output_kinematics = output_kinematics.reshape(-1, 5)
    output_kinematics = reverse_output_transformers.inverse_transform(output_kinematics).reshape(-1, 2, 4)
    output_kinematics = vector.array(
        {
            "pt": output_kinematics[..., 0],
            "eta": output_kinematics[..., 1],
            "phi": output_kinematics[..., 2],
            "energy": output_kinematics[..., 3],
        }
    )
    invariant_mass = (output_kinematics[:, 0], output_kinematics[:, 1]).__module__

    scaled_invariant_mass = (invariant_mass - invariant_mass_scalers[0]) / invariant_mass_scalers[1]

    return nn.MSELoss()(scaled_invariant_mass, target_invariant_mass)