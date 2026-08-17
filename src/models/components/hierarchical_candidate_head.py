"""Memory-efficient W-pair and conditional b-extension scorer."""

import torch
import torch.nn as nn


class HierarchicalCandidateHead(nn.Module):
    """Score unordered W pairs and b extensions for each chain query."""

    def __init__(self, embedding_size: int, hidden_size: int = 32):
        super().__init__()
        self.query_w = nn.Linear(embedding_size, hidden_size)
        self.particle_w = nn.Linear(embedding_size, hidden_size)
        self.query_b = nn.Linear(embedding_size, hidden_size)
        self.particle_b = nn.Linear(embedding_size, hidden_size)
        self.pair_b = nn.Linear(hidden_size, hidden_size)
        self.w_bias = nn.Linear(hidden_size, 1)
        self.scale = hidden_size ** -0.5

    def forward(self, queries, particles):
        particle_w = torch.tanh(self.particle_w(particles))
        pair = torch.tanh(particle_w[:, :, None, :] + particle_w[:, None, :, :])
        w_scores = torch.einsum("bqh,bjkh->bqjk", self.query_w(queries), pair) * self.scale
        w_scores = w_scores + self.w_bias(pair).squeeze(-1)[:, None]
        b_context = torch.tanh(
            self.query_b(queries)[:, :, None, :] + self.particle_b(particles)[:, None, :, :]
        )
        pair_context = torch.tanh(self.pair_b(pair))
        b_extension = torch.einsum("bqih,bjkh->bqijk", b_context, pair_context) * self.scale
        return w_scores, b_extension
