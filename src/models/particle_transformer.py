import torch.nn as nn
import torch.nn.functional as F
import torch

from typing import List

class ParticleBinaryClassificaitionHead(nn.Module):
    def __init__(self, input_size: int,
                 hidden_sizes: List[int],
                 p_dropout : float,
                 activation_function: F = F.relu,
                 return_logits: bool = True,
                 n_classes: int = 1):
        super().__init__()
        self.layer_sizes = [input_size] + hidden_sizes + [n_classes]
        self.linear_layers = nn.ModuleList([
            nn.Linear(in_size, out_size) for in_size, out_size in zip(self.layer_sizes[: -1 ], self.layer_sizes[1: ])
        ])
        self.activation_function = activation_function
        self.dropout = nn.Dropout(p_dropout)
        self.return_logits = return_logits

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        for layer in self.linear_layers[: -1]:
            x = layer(x)
            x = self.activation_function(x)
            x = self.dropout(x)
        if self.return_logits:
            return self.linear_layers[-1](x)
        return F.sigmoid(self.linear_layers[-1](x))

class ParticleTransformer(nn.Module):
    def __init__(self, n_input_features: int, 
                 embedding_size: int,
                 classifcation_head: nn.Module,
                 activation_function = F.gelu,
                 nhead = 8,
                 dim_feedforward = 2048,
                 with_cls_tkn: bool = True,
                 num_layers = 6,
                 dropout_p = 0.1,
                 n_global_features: int = 0,
                 ):
        super().__init__()
        self.with_cls_tkn = with_cls_tkn

        self.particle_linear_embedding = nn.Linear(n_input_features, embedding_size)
        if n_global_features != 0:
            self.use_global = True
            self.global_linear_embedding = nn.Linear(n_global_features, embedding_size)
        else:
            self.use_global = False
        if self.with_cls_tkn:
            self.cls_tkn = nn.Parameter(torch.rand(1, 1, embedding_size))
        self.transformer_layer = nn.TransformerEncoderLayer(
            embedding_size,
            nhead = nhead,
            dim_feedforward = dim_feedforward,
            activation = activation_function,
            batch_first = True,
            dropout = dropout_p
        )
        self.transformer_block = nn.TransformerEncoder(self.transformer_layer,
                                                       num_layers = num_layers)
        
        self.classificaiton_head = classifcation_head

    def forward(self, x) -> torch.Tensor:

        particle_features = x["particle_features"]          
        B = particle_features.size(0)

        part_tok = self.particle_linear_embedding(particle_features)  

        cls_tok = None
        if self.with_cls_tkn:
            cls_tok = self.cls_token.expand(B, -1, -1)              

        glob_tok = None
        if self.use_global:
            global_features = x["global_features"]                    
            if global_features.dim() == 2:
                global_features = global_features.unsqueeze(1)      
            glob_tok = self.global_linear_embedding(global_features)  

        # >>> The easy concat: build list, drop None, cat on seq dim (dim=1)
        tokens = [t for t in (cls_tok, part_tok, glob_tok) if t is not None]
        src = torch.cat(tokens, dim=1)                                # [B, T, E]

        src_key_padding_mask = None
        if "particle_mask" in x:
            pmask = x["particle_mask"].bool()                         # [B, Np]
            masks = []
            if self.with_cls_tkn:
                masks.append(torch.zeros(B, 1, dtype=torch.boole))
            masks.append(pmask)
            if self.use_global:
                if "global_mask" in x:
                    gmask = x["global_mask"].bool()
                    if gmask.dim() == 1:
                        gmask = gmask.unsqueeze(1)
                else:
                    # assume globals are real tokens by default
                    gmask = torch.zeros(B, glob_tok.size(1), dtype=torch.bool, device=pmask.device)
                masks.append(gmask)
            src_key_padding_mask = torch.cat(masks, dim=1)            # [B, T]

        h = self.transformer_block(src, src_key_padding_mask=src_key_padding_mask)  
        ## Uses cls token if avaliable or mean pool across particles 
        pooled = h[:, 0] if self.with_cls_tkn else h.mean(dim=1)     

        return self.classificaiton_head(pooled)
        

            