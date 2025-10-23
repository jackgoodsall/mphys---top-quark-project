import torch.nn as nn
import torch
import torch.nn.functional as F

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
        assert embedded_dim // n_heads  == 0 ; "nheads needs to be a factor of embedded dimension"
        self.head_dim  = embedded_dim // n_heads 

        self.attn_dropout = attn_dropout
        self.p_dropout = p_dropout
        self.pair_wise_dim = pair_wise_dim

        ## Attention Layers
        self.ln1 = nn.LayerNorm(embedded_dim)
        self.qkv = nn.Linear(embedded_dim, embedded_dim * 3) # wq, wk, wv
        
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
        qkv = qkv.view(B, T, H, head_dim, 3).tranponse(1, 2)  # [B, N_head, N_particle, N_head_dim, 3]
        q, k , v = qkv[..., 0],qkv[..., 1],qkv[..., 2] 

        ## Assumes by construction that u -> the interaction is padded to -inf where
        ## either particle is a padded particle in the original tensor.

        attention_scores = F.scaled_dot_product_attention( 
            q, k, v, 
            attn_mask = u,
            dropout_p = self.attn_dropout,
            is_causal = False)
        
        post_attention_scores = self.ln2(attention_scores) + x
        residual = self.feed_forward_block(post_attention_scores) + post_attention_scores
        return residual



class ClassAttentionBlock(nn.module):
    """
    Implements the class attention block desribed in the paper  https://arxiv.org/abs/2202.03772 
    """
    def __init__(self, embedded_dim, feed_for_dim, 
                 n_heads, p_dropout):
        super().__init__()

        self.embedded_dim = embedded_dim
        self.feed_for_dim = feed_for_dim
        self.n_heads = n_heads
        assert embedded_dim // n_heads  == 0 ; "nheads needs to be a factor of embedded dimension"
        self.head_dim  = embedded_dim // n_heads 
        self.p_dropout = p_dropout

        ## Attention

        self.ln1 = nn.LayerNorm(embedded_dim)
