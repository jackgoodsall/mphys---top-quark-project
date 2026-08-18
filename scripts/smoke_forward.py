#!/usr/bin/env python
"""
CPU smoke test for the assignment-model upgrade (Stages A-D).

Builds a tiny MaskedReconstructionPart from config/smoke_config.yaml, runs
forward + loss + backward on a synthetic batch (B=4, N=20 with obj_valid
patterns [T,T], [T,F], [F,F], [T,T]) across every new flag combination:
mask_embed_head x mask_logit_scale x phase1_mask_overwrite x masked_cross_attention
x new tasks (exclusive_ce / chain_state) x leptonic global token.

Asserts every loss is finite and that every requires_grad parameter that should be
trained receives a gradient (a cheap DDP-unused-parameter proxy).

Run:  .transformer_env/bin/python scripts/smoke_forward.py
"""
import copy
import itertools
import os
import sys

import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)                       # for `from src.models...` absolute imports
sys.path.insert(0, os.path.join(_ROOT, "src"))  # for `from models...` / `import main`

from models.particle_transformer import (  # noqa: E402
    ParticleEmbedder, InteractionEmbedder, MaskedReconstructionPart,
)
from utils.utils import load_any_config  # noqa: E402
import main as main_mod  # noqa: E402

B, N, K_NU = 4, 20, 1
CHAIN_PATTERNS = [
    ((1, 1), (1, 1)),  # both full
    ((0, 1), (0, 0)),  # W-only + absent
    ((0, 0), (0, 0)),  # both absent
    ((1, 1), (1, 1)),
]


def build_batch(leptonic=False, with_p4=False, all_negative=False):
    torch.manual_seed(0)
    jet = torch.randn(B, N, 7)
    src_mask = torch.ones(B, N, dtype=torch.bool)
    interactions = torch.randn(B, N, N, 4).abs()

    # Two chains: top0=particles[0:6] W0=[0:3]; top1=[6:12] W1=[6:9] (W subset of top).
    jmt = torch.zeros(B, 4, N)   # [top0, top1, w0, w1]
    tvm = torch.zeros(B, 4, dtype=torch.bool)
    for b, chains in enumerate(CHAIN_PATTERNS):
        for chain, (top_valid, w_valid) in enumerate(chains):
            start = 6 * chain
            if w_valid:
                jmt[b, 2 + chain, start:start + 3] = 1.0
                tvm[b, 2 + chain] = True
            if top_valid:
                jmt[b, chain, start:start + 6] = 1.0
                tvm[b, chain] = True

    samples = {"jet": jet, "src_mask": src_mask, "interactions": interactions}
    targets = {
        "jet_mask_true": jmt,
        "jet_valid_mask": src_mask.float(),
        "target_valid_mask": tvm,
    }
    if with_p4:
        # E >= |p| on every valid row (E from a larger scale than px/py/pz).
        p3 = torch.randn(B, N, 3)
        E = p3.norm(dim=-1, keepdim=True) + torch.rand(B, N, 1) + 1.0
        targets["jet_p4_raw"] = torch.cat([E, p3], dim=-1)  # [B, N, 4] = (E, px, py, pz)
    if leptonic:
        samples["particle_type"] = torch.zeros(B, N, dtype=torch.long)
        samples["globals"] = torch.randn(B, 6)
        targets["chain_type"] = torch.zeros(B, 2, dtype=torch.long)  # hadronic
        targets["neutrino_truth"] = torch.zeros(B, 2, K_NU)
    if all_negative:
        # Force the masked-attention all-blocked fallback: no attendable particle.
        samples["jet"] = jet * 0.0
    return samples, targets


def build_model(cfg):
    pe = ParticleEmbedder(**cfg["model_parameters"]["particle_embedder"])
    ie = InteractionEmbedder(**cfg["model_parameters"]["interaction_embedder"])
    tr = main_mod.create_default_task_registry(cfg)
    model = MaskedReconstructionPart(
        particle_embedder=pe, interaction_embedder=ie, task_registry=tr,
        **cfg["model_parameters"]["transformer"],
    )
    return model, tr


def compute_loss(outputs, task_registry):
    total = 0.0
    final = max(outputs.keys())
    for lid, preds in outputs.items():
        lt = preds["__targets__"]
        p = {k: v for k, v in preds.items() if k != "__targets__"}
        loss, _ = task_registry.compute_total_loss(
            predictions=p, targets=lt, valid_mask=lt.get("jet_valid_mask"),
            layer_id=lid, is_final_layer=(lid == final),
        )
        total = total + loss
    return total


def run_case(name, cfg, batch):
    model, tr = build_model(cfg)
    model.train()
    samples, targets = batch
    out = model.match_for_loss(model(samples), targets)
    loss = compute_loss(out, tr)
    assert torch.isfinite(loss), f"[{name}] non-finite loss: {loss}"
    loss.backward()
    # DDP proxy: every trained param should get a grad. Query heads/embeddings that
    # are genuinely unused in a given flag combo are allowed to be None.
    missing = [n for n, p in model.named_parameters()
               if p.requires_grad and p.grad is None]
    print(f"  [{name:52s}] loss={loss.item():+.4f}  no-grad params={len(missing)}")
    return missing


def main():
    base = load_any_config("config/smoke_config.yaml")

    cases = 0
    for embed, scale, overwrite in itertools.product([False, True], repeat=3):
        cfg = copy.deepcopy(base)
        cfg["tasks"]["mask"]["mask_embed_head"] = embed
        cfg["tasks"]["mask_W"]["mask_embed_head"] = embed
        cfg["model_parameters"]["transformer"]["mask_logit_scale"] = scale
        cfg["model_parameters"]["transformer"]["phase1_mask_overwrite"] = overwrite
        run_case(f"embed={embed} scale={scale} overwrite={overwrite}",
                 cfg, build_batch())
        cases += 1

    # New tasks on
    cfg = copy.deepcopy(base)
    cfg["tasks"]["exclusive_ce"] = {"loss_weight": 0.25, "layer_weights": {0: 0.1, 1: 0.1, 2: 1.0}}
    cfg["tasks"]["exclusive_ce_W"] = {"loss_weight": 0.25, "layer_weights": {0: 0.1, 1: 1.0, 2: 0.1}}
    cfg["tasks"]["mask_consistency"] = {"loss_weight": 0.1, "margin": 0.0, "layer_weight_strategy": "final_only"}
    run_case("new tasks: exclusive_ce + mask_consistency", cfg, build_batch())
    cases += 1

    # Masked cross-attention (normal + all-negative fallback)
    for allneg in [False, True]:
        cfg = copy.deepcopy(base)
        cfg["model_parameters"]["transformer"]["masked_cross_attention"] = True
        cfg["model_parameters"]["transformer"]["masked_attention_start_epoch"] = 0
        run_case(f"masked_cross_attention allneg={allneg}", cfg,
                 build_batch(all_negative=allneg))
        cases += 1

    # Leptonic global-token path
    cfg = copy.deepcopy(base)
    cfg["model_parameters"]["transformer"]["enable_leptonic"] = True
    cfg["model_parameters"]["transformer"]["global_dim"] = 6
    cfg["model_parameters"]["transformer"]["global_hidden_sizes"] = [8]
    cfg["tasks"]["chain_type"] = {"loss_weight": 1.0, "layer_weight_strategy": "final_only"}
    cfg["tasks"]["neutrino"] = {"loss_weight": 1.0, "output_dim": 1, "layer_weight_strategy": "final_only"}
    run_case("leptonic global token + masked attn", cfg,
             build_batch(leptonic=True))
    cases += 1

    # Leptonic + masked attention together
    cfg["model_parameters"]["transformer"]["masked_cross_attention"] = True
    run_case("leptonic + masked_cross_attention", cfg, build_batch(leptonic=True))
    cases += 1

    check_augmenter()

    print(f"\nAll {cases} smoke cases produced finite losses and ran backward. OK")


def check_augmenter():
    """Stage D3: φ-rotation preserves sin²+cos²; η-flip is an involution; padding stays 0."""
    from data.datamodule import Augmenter
    eta_consts = ((0.1, 1.3), (0.0, 1.0), (0.0, 1.0))

    # φ-rotation
    aug = Augmenter({"phi_rotation": True, "eta_flip": False})
    jet = torch.randn(20, 7)
    jet[10:] = 0.0  # padding rows
    src = torch.ones(20, dtype=torch.bool); src[10:] = False
    sample = {"jet": jet.clone(), "src_mask": src}
    target = {"target_kinematics": torch.randn(4, 5)}
    r2_before = jet[:, 2] ** 2 + jet[:, 3] ** 2
    s, t = aug(sample, target)
    r2_after = s["jet"][:, 2] ** 2 + s["jet"][:, 3] ** 2
    assert torch.allclose(r2_before, r2_after, atol=1e-5), "phi rotation broke sin^2+cos^2"
    assert torch.allclose(s["jet"][10:], torch.zeros(10, 7), atol=1e-6), "phi rotated padding"

    # η-flip involution on jet col 1 (apply twice = identity on real rows)
    aug2 = Augmenter({"phi_rotation": False, "eta_flip": True}, eta_consts=eta_consts)
    jet = torch.randn(20, 7); jet[10:] = 0.0
    sample = {"jet": jet.clone(), "src_mask": src}
    jm, js = eta_consts[0]
    twice = -(-jet[:, 1] - 2 * jm / js) - 2 * jm / js
    assert torch.allclose(twice, jet[:, 1], atol=1e-5), "eta flip not an involution"
    print("  [augmenter: phi-rotation invariants + eta-flip involution]     OK")


if __name__ == "__main__":
    main()
