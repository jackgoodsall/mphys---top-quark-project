import torch.nn as nn
import torch
import torch.nn.functional as F

class DecoderAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward, 
                 p_dropout, activation):
        super().__init__()
        # --- Attention blocks ---
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=p_dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=p_dropout, batch_first=True
        )

        # --- Feedforward network ---
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # --- Normalization and dropout ---
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p_dropout)
        self.dropout2 = nn.Dropout(p_dropout)
        self.dropout3 = nn.Dropout(p_dropout)

        # --- Activation function ---
        if isinstance(activation, str):
            if activation.lower() == "relu":
                self.activation = F.relu
            elif activation.lower() == "gelu":
                self.activation = F.gelu
            else:
                raise ValueError(f"Unsupported activation {activation}")
        else:
            self.activation = activation  # custom callable

    def forward(self, x, memory, 
                tgt_mask = None, memory_mask = None,
                tgt_key_padding_mask = None, memory_key_padding_mask = None,
                tgt_is_causal = None, memory_is_causal = None):
        # --- Self-attention (causal) ---
        residual = x
        x_sa, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
        )
        x = self.norm1(residual + self.dropout1(x_sa))

        residual = x
        x_ca, _ = self.cross_attn(
            query=x,
            key=memory,
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        x = self.norm2(residual + self.dropout2(x_ca))

        # --- Feed-forward network ---
        residual = x
        x_ff = self.linear2(self.dropout3(self.activation(self.linear1(x))))
        x = self.norm3(residual + x_ff)
        return x


class ParticleAttentionBlock(nn.Module):
    """
    Implements the Particle Attention block decribed in the paper https://arxiv.org/abs/2202.03772 

    """
    def __init__(self, embedded_dim, feed_for_dim, 
                 n_heads, p_dropout ,pair_wise_dim, interaction_mode = "sum", attn_dropout = 0):
        super().__init__()

        self.interaction_mode = interaction_mode
        self.embedded_dim = embedded_dim
        self.feed_for_dim = feed_for_dim
        self.n_heads = n_heads

        assert embedded_dim % n_heads  == 0 ; "nheads needs to be a factor of embedded dimension"
        self.head_dim  = embedded_dim // n_heads 

        self.attn_dropout = attn_dropout
        self.p_dropout = p_dropout
        self.pair_wise_dim = pair_wise_dim

        ## Attention Layers
        self.ln1 = nn.LayerNorm(embedded_dim)
        self.qkv = nn.Linear(embedded_dim, embedded_dim * 3) # wq, wk, wv
        self.out_proj = nn.Linear(embedded_dim, embedded_dim)

        self.ln2 = nn.LayerNorm(embedded_dim)
        ## Feedforward residual area
        self.feed_forward_block = nn.Sequential(
        nn.LayerNorm(self.embedded_dim),
        nn.Linear(self.embedded_dim, self.feed_for_dim),
        nn.GELU(),
        nn.LayerNorm(self.feed_for_dim),
        nn.Linear(self.feed_for_dim, self.embedded_dim),
        )

    def forward(self, x, u, src_mask = None):
        B, T = x.shape[0], x.shape[1]
        H = self.n_heads
        head_dim = self.head_dim
        x1 = self.ln1(x) # [B, N_part, embedded]
        qkv = self.qkv(x1) # [B, N_part, embedded * 3]
        qkv = qkv.view(B, T, H, head_dim, 3).transpose(1, 2)  # [B, N_head, N_particle, N_head_dim, 3]
        q, k , v = qkv[..., 0],qkv[..., 1],qkv[..., 2] 

        ## Assumes by construction that u -> the interaction is padded to -inf where
        ## either particle is a padded particle in the original tensor.

        attention_scores = F.scaled_dot_product_attention( 
            q, k, v, 
            attn_mask = u,
            dropout_p = self.attn_dropout,
            is_causal = False)
        
        attn = attention_scores.transpose(1, 2).contiguous().view(B, T, self.embedded_dim)
        attn = self.out_proj(attn)


        post_attention_scores = self.ln2(attn) + x
        residual = self.feed_forward_block(post_attention_scores) + post_attention_scores
        return residual


class ParticleGatingModule(nn.Module):
    """
    Learns a per-particle relevance score in [0, 1] via a small cross-attention
    stack over encoder memory.  Gated memory = memory * relevance, suppressing
    background particles before the decoder cross-attends to them.
    """

    def __init__(self, d_model, n_gating_layers=2, n_gate_queries=1, nhead=8, dim_feedforward=256):
        super().__init__()
        self.gate_queries = nn.Parameter(0.01 * torch.randn(n_gate_queries, d_model))
        self.gating_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
            for _ in range(n_gating_layers)
        ])

    def forward(self, memory, src_key_padding_mask=None):
        # memory: [B, N, D],  src_key_padding_mask: [B, N] True=PADDING
        B = memory.size(0)
        gate_q = self.gate_queries.unsqueeze(0).expand(B, -1, -1)  # [B, G, D]
        for layer in self.gating_layers:
            gate_q = layer(gate_q, memory, memory_key_padding_mask=src_key_padding_mask)
        scores = torch.einsum('bgd,bnd->bgn', gate_q, memory) / (memory.size(-1) ** 0.5)  # [B, G, N]
        if src_key_padding_mask is not None:
            scores = scores.masked_fill(src_key_padding_mask.unsqueeze(1), float('-inf'))
        relevance = torch.sigmoid(scores.max(dim=1).values)  # [B, N]
        gated_memory = memory * relevance.unsqueeze(-1)
        return relevance, gated_memory


class MIParticleAttentionBlock(nn.Module):
    """
    MI-Particle Attention Block from MIParT (arXiv:2407.08682).

    Replaces P-MHA with More-Interaction Attention (MIA):

        MIA(U, V) = Softmax(U) · V

    U [B, D_mia, N, N] is the high-dimensional interaction tensor (channels-first,
    -inf at padding positions).  V = W_v(x) is a value projection of particle
    features.  Each of the D_mia channels acts as an independent attention head,
    giving the model D_mia separate soft-selection patterns over particles.

    Structure mirrors ParticleAttentionBlock (NormFormer-style residuals).
    """

    def __init__(self, embedded_dim: int, feed_for_dim: int,
                 p_dropout: float, mia_dim: int):
        super().__init__()
        self.embedded_dim = embedded_dim
        self.mia_dim = mia_dim

        # MIA sub-block
        self.ln1 = nn.LayerNorm(embedded_dim)
        self.v_proj = nn.Linear(embedded_dim, mia_dim)
        self.mia_out_proj = nn.Linear(mia_dim, embedded_dim)
        self.mia_dropout = nn.Dropout(p_dropout)

        # Feed-forward block — same structure as ParticleAttentionBlock
        self.ln2 = nn.LayerNorm(embedded_dim)
        self.feed_forward_block = nn.Sequential(
            nn.LayerNorm(embedded_dim),
            nn.Linear(embedded_dim, feed_for_dim),
            nn.GELU(),
            nn.LayerNorm(feed_for_dim),
            nn.Linear(feed_for_dim, embedded_dim),
        )

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, N, D]        particle features
            u : [B, D_mia, N, N] high-dim interaction tensor, channels-first.
                                  Padding pairs must already be -inf so Softmax
                                  assigns zero weight to them.
        Returns:
            [B, N, D] updated particle features
        """
        # channels-last view needed for the einsum: [B, N, N, D_mia]
        u_t = u.permute(0, 2, 3, 1)

        x1 = self.ln1(x)                                         # [B, N, D]
        V  = self.v_proj(x1)                                     # [B, N, D_mia]
        weights    = torch.nan_to_num(F.softmax(u_t, dim=2), nan=0.0)  # [B, N, N, D_mia]; nan_to_num handles fully-masked padding rows
        mia_result = torch.einsum('bijn,bjn->bin', weights, V)   # [B, N, D_mia]
        mia_result = self.mia_dropout(self.mia_out_proj(mia_result))  # [B, N, D]

        post_mia = self.ln2(mia_result) + x
        residual = self.feed_forward_block(post_mia) + post_mia
        return residual


class InteractionDimReducer(nn.Module):
    """
    Pointwise 1-D convolution that compresses the interaction tensor from
    D_mia (used by MIParticleAttentionBlocks) down to n_heads (used by the
    subsequent ParticleAttentionBlocks), exactly as in MIParT Fig. 1.

    Input/output shape: [B, D_in, N, N] → [B, D_out, N, N].
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.conv = nn.Conv1d(d_in, d_out, kernel_size=1)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        B, D, N, M = u.shape
        u_flat = u.view(B, D, N * M)           # flatten pairs to length dim
        return self.conv(u_flat).view(B, -1, N, M)


class ClassAttentionBlock(nn.Module):
    """
    Implements the class attention block desribed in the paper  https://arxiv.org/abs/2202.03772 
    """
    def __init__(self, embedded_dim,n_heads, feed_for_dim, 
                  p_dropout):
        super().__init__()

        self.embedded_dim = embedded_dim
        self.feed_for_dim = feed_for_dim
        self.n_heads = n_heads
        assert embedded_dim % n_heads  == 0 ; "nheads needs to be a factor of embedded dimension"
        self.head_dim  = embedded_dim // n_heads 
        self.p_dropout = p_dropout

        ## Attention

        self.ln1 = nn.LayerNorm(embedded_dim)


        self.query_linear = nn.Linear(embedded_dim, embedded_dim)
        self.attention = nn.MultiheadAttention(embedded_dim, n_heads, p_dropout, batch_first= True)
        self.ln2 = nn.LayerNorm(embedded_dim)
        ## Feedforward block
        self.feed_forward_block = nn.Sequential(
        nn.LayerNorm(self.embedded_dim),
        nn.Linear(self.embedded_dim, self.feed_for_dim),
        nn.GELU(),
        nn.LayerNorm(self.feed_for_dim),
        nn.Linear(self.feed_for_dim, self.embedded_dim),
        )
    
    def forward(self, x, cls_tkn, src_mask = None):
        B, N, _ = x.shape
        inputs = torch.concat((x, cls_tkn), axis = 1)
        if src_mask is not None:
            src_mask = torch.concat((src_mask, torch.zeros(B, cls_tkn.shape[1]).to(x.device)), axis = 1)
        attention_input = self.ln1(inputs)
        
        attention_scores, _ = self.attention(
            cls_tkn,
            attention_input,
            attention_input,
            key_padding_mask = src_mask
        )

        
        attn = attention_scores + cls_tkn


        post_attention_scores = self.ln2(attn) 

        residual = self.feed_forward_block(post_attention_scores) + post_attention_scores
        return residual