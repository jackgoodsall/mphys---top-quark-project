import torch


def logminmax_forward_torch(x, scaler, device, eps=0.0):
    """
    Torch version of LogMinMax.transform:
        X_scaled = (log1p(clip(X, 0, inf)) - log_min) * scale + min_offset
    but using the already-computed scaler.scale_ and scaler.min_offset_.

    x: tensor of physical values (unscaled)
    scaler: fitted LogMinMax instance
    """
    x = x.to(device).float()

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

    # optional clipping to feature_range
    if getattr(scaler, "clip", False):
        low, high = scaler.feature_range
        low_t  = torch.tensor(low,  dtype=torch.float32, device=device)
        high_t = torch.tensor(high, dtype=torch.float32, device=device)
        x_scaled = torch.clamp(x_scaled, low_t, high_t)

    return x_scaled
