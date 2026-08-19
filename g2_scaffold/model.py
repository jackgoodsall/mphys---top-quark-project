"""Small, standalone G2 candidate scorer.

This module is deliberately not imported by the live G1 model.  It produces
three-state chain logits, unordered W-pair logits, and conditional b-extension
logits for a later G2 training integration.
"""

import torch
from torch import nn


class G2CandidateScorer(nn.Module):
    """Score W pairs and b extensions for each chain query."""

    def __init__(self, embedding_size: int, hidden_size: int = 32):
        super().__init__()
        self.query_w = nn.Linear(embedding_size, hidden_size)
        self.particle_w = nn.Linear(embedding_size, hidden_size)
        self.query_b = nn.Linear(embedding_size, hidden_size)
        self.particle_b = nn.Linear(embedding_size, hidden_size)
        self.pair_b = nn.Linear(hidden_size, hidden_size)
        self.w_bias = nn.Linear(hidden_size, 1)
        self.state = nn.Linear(embedding_size, 3)
        self.scale = hidden_size**-0.5

    def forward(self, queries: torch.Tensor, particles: torch.Tensor) -> dict[str, torch.Tensor]:
        if queries.ndim != 3 or particles.ndim != 3:
            raise ValueError("queries must be [B,Q,D] and particles must be [B,N,D]")
        if queries.shape[0] != particles.shape[0] or queries.shape[-1] != particles.shape[-1]:
            raise ValueError("queries and particles must share batch and embedding dimensions")

        particle_w = torch.tanh(self.particle_w(particles))
        pair = torch.tanh(particle_w[:, :, None, :] + particle_w[:, None, :, :])
        w_pair = torch.einsum("bqh,bjkh->bqjk", self.query_w(queries), pair) * self.scale
        w_pair = w_pair + self.w_bias(pair).squeeze(-1)[:, None]

        b_context = torch.tanh(
            self.query_b(queries)[:, :, None, :] + self.particle_b(particles)[:, None, :, :]
        )
        pair_context = torch.tanh(self.pair_b(pair))
        b_extension = torch.einsum("bqih,bjkh->bqijk", b_context, pair_context) * self.scale

        return {
            "state_logits": self.state(queries),
            "w_pair_logits": w_pair,
            "b_extension_logits": b_extension,
        }
