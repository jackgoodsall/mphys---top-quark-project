#!/usr/bin/env python
"""
CPU build + forward/loss/backward check for a training config, before submitting
a GPU job. Uses a tiny synthetic batch. Reports non-finite losses and any
requires_grad parameter that receives no gradient (a DDP-unused-parameter proxy —
should be 0 for a config meant to run under strategy: ddp).

    .transformer_env/bin/python scripts/validate_config.py config/exp_bundle_v1.yaml
"""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

import torch  # noqa: E402
from utils.utils import load_any_config  # noqa: E402
import scripts.smoke_forward as sf  # noqa: E402


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config/exp_bundle_v1.yaml"
    cfg = load_any_config(cfg_path)
    # Force a CPU-friendly matcher for the check (GPU brute-force falls back anyway).
    cfg["model_parameters"]["transformer"]["matching_solver"] = "scipy"

    model, tr = sf.build_model(cfg)
    model.train()

    # Synthetic hadronic batch; include jet_p4_raw for the invariant-mass task.
    samples, targets = sf.build_batch(with_p4=True)
    inp = dict(samples)
    inp["targets"] = targets
    out = model(inp)
    loss = sf.compute_loss(out, tr)

    ok = torch.isfinite(loss).item()
    loss.backward()
    missing = [n for n, p in model.named_parameters()
               if p.requires_grad and p.grad is None]

    print(f"config      : {cfg_path}")
    print(f"tasks       : {list(tr.tasks.keys())}")
    print(f"loss finite : {ok}   (loss={loss.item():+.4f})")
    print(f"no-grad params (DDP-unused proxy): {len(missing)}")
    for n in missing:
        print(f"   - {n}")
    if not ok or missing:
        print("RESULT: FAIL")
        sys.exit(1)
    print("RESULT: OK — safe to submit under strategy: ddp")


if __name__ == "__main__":
    main()
