import torch.nn as nn
import torch

import torch
import vector


from scipy.optimize import linear_sum_assignment
import numpy as np

import torch.nn.functional as F


def logminmax_forward_torch(x, scaler, device, eps=0.0):
    """
    Torch version of LogMinMax.transform:
        X_scaled = (log1p(clip(X, 0, inf)) - log_min) * scale + min_offset
    but using the already-computed scaler.scale_ and scaler.min_offset_.

    x: tensor of physical values (unscaled)
    scaler: fitted LogMinMax instance
    """
    x = x.to(device).float()

    # NaN mask preserved
    mask = torch.isnan(x)

    # clip x >= 0 as in _log_transform
    x_pos = torch.clamp(x, min=0.0)

    # log1p
    x_log = torch.log1p(x_pos)

    # bring scaler parameters to torch
    scale = torch.as_tensor(scaler.scale_,      dtype=torch.float32, device=device)
    min_offset = torch.as_tensor(scaler.min_offset_, dtype=torch.float32, device=device)

    # broadcast if needed
    while scale.dim() < x_log.dim():
        scale = scale.unsqueeze(0)
        min_offset = min_offset.unsqueeze(0)

    x_scaled = x_log * scale + min_offset

    mean = torch.as_tensor(scaler.scalar.mean_, dtype= torch.float32, device = device)
    std = torch.as_tensor(scaler.scalar.scale_, dtype= torch.float32, device = device)

    x_scaled = (x_scaled - mean) / std


    # optional clipping to feature_range
    if getattr(scaler, "clip", False):
        low, high = scaler.feature_range
        low_t  = torch.tensor(low,  dtype=torch.float32, device=device)
        high_t = torch.tensor(high, dtype=torch.float32, device=device)
        x_scaled = torch.clamp(x_scaled, low_t, high_t)

    # restore NaNs
    x_scaled[mask] = torch.nan
    return x_scaled


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
    inv_mass_transform   # scalers for invariant mass itself
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

    inv_mass_scaled = logminmax_forward_torch(inv_mass, inv_mass_transform, device)

    return torch.nn.functional.mse_loss(inv_mass_scaled.unsqueeze(-1), target_scaled_mass)

import torch
import itertools 

def hungarian_match_top_W(outputs, targets, reduction="mean"):
    """
    Batched permutation-based matching between outputs and targets (tops + Ws),
    using all 4! permutations, fully vectorised.

    Args
    ----
    outputs : torch.Tensor, shape (N, 4, F)
        Network outputs per event.
    targets : dict of torch.Tensor
        {
            "top": (N, 2, F),
            "W":   (N, 2, F),
        }
    reduction : "mean" or "sum"
        How to compute per-event MSE over the 4 objects.

    Returns
    -------
    matched : dict
        {
            "top":  (N, 2, F),
            "W":    (N, 2, F),
            "loss": (N,),  # per-event optimal MSE
        }
    """

    assert isinstance(outputs, torch.Tensor), "outputs must be a torch.Tensor"
    assert isinstance(targets["top"], torch.Tensor)
    assert isinstance(targets["W"], torch.Tensor)

    device = outputs.device
    dtype  = outputs.dtype

    top = targets["top"].to(device=device, dtype=dtype)
    W   = targets["W"].to(device=device, dtype=dtype)

    N, P, F = outputs.shape
    n_top = top.shape[1]
    n_W   = W.shape[1]
    assert P == n_top + n_W == 4, "This implementation assumes exactly 4 objects per event."

    # Concatenate truth in fixed order: [top0, top1, W0, W1]
    targets_cat = torch.cat([top, W], dim=1)           # (N, 4, F)

    # All permutations of the 4 indices → (24, 4)
    perms = torch.tensor(
        list(itertools.permutations(range(P))),
        device=device,
        dtype=torch.long,
    )   # (num_perm=24, 4)
    num_perm = perms.shape[0]

    # outputs: (N, 4, F)
    # We want permuted outputs for each permutation and event:
    # shape → (N, num_perm, 4, F)

    # Expand outputs to (N, 1, 4, F)
    outputs_exp = outputs.unsqueeze(1)                 # (N, 1, 4, F)
    # Index for gather over dimension 2
    perm_idx = perms.unsqueeze(0).unsqueeze(-1)        # (1, num_perm, 4, 1)
    perm_idx = perm_idx.expand(N, num_perm, 4, F)      # (N, num_perm, 4, F)

    # Broadcast outputs_exp along num_perm and gather
    outputs_exp = outputs_exp.expand(N, num_perm, 4, F)        # (N, num_perm, 4, F)
    outputs_perm = torch.gather(outputs_exp, 2, perm_idx)      # (N, num_perm, 4, F)

    # Expand targets for broadcasting
    targets_exp = targets_cat.unsqueeze(1)                     # (N, 1, 4, F)
    targets_exp = targets_exp.expand(N, num_perm, 4, F)        # (N, num_perm, 4, F)

    # Squared errors per object-feature-permutation
    diff = outputs_perm - targets_exp                          # (N, num_perm, 4, F)
    se   = diff**2                                             # (N, num_perm, 4, F)

    if reduction == "mean":
        # MSE per event per permutation
        perm_loss = se.mean(dim=(2, 3))                        # (N, num_perm)
    elif reduction == "sum":
        perm_loss = se.sum(dim=(2, 3))                         # (N, num_perm)
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")

    # Best permutation index per event
    best_perm_idx = perm_loss.argmin(dim=1)                    # (N,)

    # Gather best permutation indices per event: shape (N, 4)
    best_perm = perms[best_perm_idx]                           # (N, 4)

    # Use best_perm to reorder outputs into target order for each event
    # Make index tensor for gather over dim=1
    gather_idx = best_perm.unsqueeze(-1).expand(N, P, F)       # (N, 4, F)
    matched_all = torch.gather(outputs, 1, gather_idx)         # (N, 4, F)

    # Split into tops and Ws according to target order
    matched_top = matched_all[:, :n_top, :]                    # (N, 2, F)
    matched_W   = matched_all[:, n_top:, :]                    # (N, 2, F)

    # Per-event optimal loss
    loss_per_event = perm_loss[torch.arange(N, device=device), best_perm_idx].mean()  # (N,)

    return {
        "top":  matched_top,
        "W":    matched_W,
        "loss": loss_per_event,
    }
