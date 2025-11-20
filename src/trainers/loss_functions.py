import torch.nn as nn
import torch

import torch
import vector


def logminmax_inverse_torch(x_scaled, logminmax, device):
    # logminmax is the fitted sklearn LogMinMaxScaler
    mean = torch.tensor(logminmax.scalar.mean_,  device=device, dtype=torch.float32)
    std  = torch.tensor(logminmax.scalar.scale_, device=device, dtype=torch.float32)
    scale = torch.tensor(logminmax.scale_,       device=device, dtype=torch.float32)
    min_offset = torch.tensor(logminmax.min_offset_, device=device, dtype=torch.float32)

    # undo StandardScaler
    x_mm = x_scaled * std + mean            # X_mm
    # undo min–max
    x_log = (x_mm - min_offset) / scale     # X_log
    # undo log1p (and clamp because original did clip to >=0 before log1p)
    x = torch.expm1(x_log)
    x = torch.clamp(x, min=0.0)

    return x

def standard_inverse_torch(x_scaled, mean, std):
    return x_scaled * std + mean

def phi_inverse_torch(sin_phi, cos_phi):
    return torch.atan2(sin_phi, cos_phi)

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


def invariant_mass_loss(
    output,
    target_scaled_mass,
    pt_scaler,      # LogMinMaxScaler for pT
    eta_scaler,     # StandardScaler for eta
    E_scaler,       # LogMinMaxScaler for E
    mass_mean, mass_std,   # scalers for invariant mass itself
):
    device = output.device

    pt_scaled  = output[..., 0]
    eta_scaled = output[..., 1]
    sin_phi    = output[..., 2]
    cos_phi    = output[..., 3]
    E_scaled   = output[..., 4]

    # invert scalers in torch
    pt = logminmax_inverse_torch(pt_scaled, pt_scaler, device)
    eta = standard_inverse_torch(
        eta_scaled,
        torch.tensor(eta_scaler.mean_,  dtype = torch.float32,device=device),
        torch.tensor(eta_scaler.scale_,dtype = torch.float32 ,device=device),
    )
    phi = phi_inverse_torch(sin_phi, cos_phi)
    E   = logminmax_inverse_torch(E_scaled, E_scaler, device)

    # build 4-vectors in torch
    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)

    px_tot = px[:, 0] + px[:, 1]
    py_tot = py[:, 0] + py[:, 1]
    pz_tot = pz[:, 0] + pz[:, 1]
    E_tot  = E[:, 0]  + E[:, 1]

    m2 = E_tot**2 - (px_tot**2 + py_tot**2 + pz_tot**2)
    inv_mass = torch.sqrt(torch.clamp(m2, 0))

    # scale invariant mass same way you did for the target
    mass_mean = torch.as_tensor(mass_mean, device=device, dtype=torch.float32)
    mass_std  = torch.as_tensor(mass_std,  device=device, dtype=torch.float32)
    inv_mass_scaled = (torch.log1p(inv_mass) - mass_mean) / mass_std

    return torch.nn.functional.mse_loss(inv_mass_scaled.unsqueeze(-1), target_scaled_mass)