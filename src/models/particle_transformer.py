import torch.nn as nn
import torch.nn.functional as F
import torch
import sys
from typing import List

from .components.attention_layers import ParticleAttentionBlock, ClassAttentionBlock

class ParticleBinaryClassificaitionHead(nn.Module):
    def __init__(self, input_size: int,
                 hidden_sizes: List[int],
                 p_dropout : float,
                 activation_function: str,
                 return_logits: bool,
                 n_classes: int,
                 *args,
                 **kwargs):
        super().__init__()
        self.layer_sizes = [input_size] + hidden_sizes + [n_classes]
        self.linear_layers = nn.ModuleList([
            nn.Linear(in_size, out_size) for in_size, out_size in zip(self.layer_sizes[: -1 ], self.layer_sizes[1: ])
        ])
        if activation_function == "relu":
            self.activation_function = F.relu
        self.dropout = nn.Dropout(p_dropout)
        self.return_logits = return_logits

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        for layer in self.linear_layers[: -1]:
            x = layer(x)
            x = self.activation_function(x)
            x = self.dropout(x)
        #if self.return_logits:
        return self.linear_layers[-1](x)
        #return F.sigmoid(self.linear_layers[-1](x))

class ParticleTransformer(nn.Module):
    def __init__(self, n_input_features: int, 
                 embedding_size: int,
                 classifcation_head: nn.Module,
                 activation_function: str,
                 nhead: int,
                 dim_feedforward: int,
                 with_cls_tkn: bool,
                 num_layers: int,
                 p_dropout: float,
                 n_global_features: int,
                 *args,
                 **kwargs
                 ):
        super().__init__()
        self.with_cls_tkn = with_cls_tkn
        print(type(activation_function))
        self.particle_linear_embedding = nn.Sequential(
            nn.Linear(n_input_features, embedding_size//2),
            nn.GELU(),
            nn.Dropout(p_dropout),
            nn.LayerNorm(embedding_size//2),
            nn.Linear(embedding_size//2, embedding_size),
        )
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
            dropout = p_dropout
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
            cls_tok = self.cls_tkn.expand(B, -1, -1)              

        glob_tok = None
        if self.use_global:
            global_features = x["global_features"]                    
            if global_features.dim() == 2:
                global_features = global_features.unsqueeze(1)      
            glob_tok = self.global_linear_embedding(global_features)  

        tokens = [t for t in (cls_tok, part_tok, glob_tok) if t is not None]
        src = torch.cat(tokens, dim=1)                               

        src_key_padding_mask = None
        if "particle_mask" in x:
            pmask = x["particle_mask"].bool()    
            B = pmask.size(0)                     
            masks = []
            if self.with_cls_tkn:
                masks.append(pmask.new_zeros((B, 1)))      
            masks.append(pmask)
            if self.use_global:
                g_len = glob_tok.size(1)                                   
                masks.append(pmask.new_zeros((B, g_len)))   
            src_key_padding_mask = torch.cat(masks, dim=1)            
        else:
            src_key_padding_mask = None
        h = self.transformer_block(src, src_key_padding_mask=src_key_padding_mask)  
        ## Uses cls token if avaliable or mean pool across particles 
        pooled = h[:, 0] if self.with_cls_tkn else h.mean(dim=1)     

        return self.classificaiton_head(pooled)
        

class ParTInteractionFormer(nn.Module):
    def __init__(
            self,
            particle_embedder,
            classifcation_head,
            n_particle_blocks,
            n_class_blocks,
            embedded_dim,
            p_dropout,
    ):
        self.particle_embedder = particle_embedder
        self.classification_head = classifcation_head

        self.n_particle_blocks = n_particle_blocks
        self.n_class_blocks = n_class_blocks

        self.cls_tkn = nn.Parameter(torch.rand(1, 1, embedded_dim))
        
        

    def forward(self, x):
        part_features = x["particle_features"]
        global_features = x["global_features"]
        interaction_features = x["interaction_features"]
        src_mask = x["src_mask"]

        (input_sequence,
        interaction_features)  = self.particle_embedder(part_features, 
                                                        global_features, 
                                                        interaction_features)

        for particle_block in self.particle_blocks:
            input_sequence = particle_block(input_sequence,interaction_features)
        output = input_sequence

        for class_block in self.class_blocks:
            cls_tkn  = class_block()