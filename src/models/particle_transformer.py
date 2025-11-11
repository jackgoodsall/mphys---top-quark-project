import torch.nn as nn
import torch.nn.functional as F
import torch
import sys
from typing import List
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.components.attention_layers import ParticleAttentionBlock, ClassAttentionBlock, DecoderAttentionBlock

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
        super().__init__()
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


class ParticleEmbedder(nn.Module):
    def __init__(self,
                 n_input,
                 hidden_sizes,
                 embedding_size,
                 p_dropout):
        super().__init__()

        self.layer_sizes = [n_input] + hidden_sizes 
        self.layers = nn.ModuleList()
        for size_1, size_2 in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            self.layers.extend([ nn.LayerNorm(size_1), nn.Linear(size_1, size_2), nn.GELU(), nn.Dropout(p_dropout)])
        self.layers.append(nn.Linear(size_2, embedding_size))
        
    def forward(self, X):

        for layers in self.layers:
            X = layers(X)
        return X

class ReverseEmbedder(nn.Module):
    def __init__(self,
                 n_input,
                 hidden_sizes,
                 output_size,
                 p_dropout):
        super().__init__()

        self.layer_sizes = [n_input] + hidden_sizes 
        self.layers = nn.ModuleList()
        for size_1, size_2 in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            self.layers.extend([nn.LayerNorm(size_1), nn.Linear(size_1, size_2), nn.GELU(), nn.Dropout(p_dropout)])
        self.layers.append(nn.Linear(size_2, output_size))
        
    def forward(self, X):

        for layers in self.layers:
            X = layers(X)
        return X



class ReconstructionTransformer(nn.Module):
    def __init__(self,
                 particle_embedder,
                 reverse_embedder,
                 embedding_size,
                 n_encoder_layers,
                 n_decoder_layers,
                 n_heads,
                 out_dimensions,
                 dim_ff,
                 p_dropout,
                 activation_function = "gelu",
                 ):
        super().__init__()

        self.particle_embedder = particle_embedder

        self.encoder_layer = nn.TransformerEncoderLayer(
            embedding_size,
            nhead = n_heads,
            dim_feedforward = dim_ff,
            activation = activation_function,
            batch_first = True,
            dropout = p_dropout
        )

        """ #self.decoder_layer = DecoderAttentionBlock(embedding_size,
                                                   n_heads,
                                                   dim_ff,
                                                   p_dropout,
                                                   activation_function)
         """
        self.decoder_layer = nn.TransformerDecoderLayer(
            embedding_size, n_heads, dim_ff, p_dropout, activation = activation_function,
            batch_first= True
        )
            


        self.encoder_stack = nn.TransformerEncoder(self.encoder_layer, n_encoder_layers)
        self.decoder_stack = nn.TransformerDecoder(self.decoder_layer, n_decoder_layers)

        self.reverse_embedder = reverse_embedder

        self.tgt_tokens = nn.Parameter(torch.rand((1, 2, embedding_size)) *0.02)



    def forward(self, X):
        
        jet = X["jet"]
        src_mask = X["src_mask"]
        
        jet_embedded = self.particle_embedder(jet)

        B, N, F = jet_embedded.shape

        tgt_tokens = self.tgt_tokens.expand(B, 2,F  )
        tgt_tokens = torch.zeros((B, 2,
                                  F)).to(jet.device)

        jet_memory = self.encoder_stack(jet_embedded, src_key_padding_mask = src_mask)

        decoded_tokens = self.decoder_stack(tgt_tokens, 
                                            jet_memory,
                                            memory_key_padding_mask =src_mask)
        return self.reverse_embedder(decoded_tokens)



class ReconstructionEncoderClassTokens(nn.Module):
    def __init__(self,
                 particle_embedder,
                 reverse_embedder,
                 embedding_size,
                 n_encoder_layers,
                 n_decoder_layers,
                 n_heads,
                 out_dimensions,
                 dim_ff,
                 p_dropout,
                 activation_function = "gelu",
                 ):
        super().__init__()

        self.particle_embedder = particle_embedder

        self.encoder_layer = nn.TransformerEncoderLayer(
            embedding_size,
            nhead = n_heads,
            dim_feedforward = dim_ff,
            activation = activation_function,
            batch_first = True,
            dropout = p_dropout
        )
 
        self.class_tokens = nn.Parameter(torch.rand(1, 2, embedding_size) * 0.01)
            


        self.encoder_stack = nn.TransformerEncoder(self.encoder_layer, n_encoder_layers)

        self.reverse_embedder = reverse_embedder


    def forward(self, X):
        
        jet = X["jet"]
        src_mask = X["src_mask"]
        
        jet_embedded = self.particle_embedder(jet)
        B, N, F = jet_embedded.shape

        cls_tkns = self.class_tokens.expand(B, 2, F).to(jet.device)

        empty_mask  = torch.zeros((B, 2)).to(jet.device)
        src_mask = torch.concat((empty_mask, src_mask), axis = 1)


        encoder_input = torch.concat((cls_tkns, jet_embedded), axis = 1)
        encoder_output = self.encoder_stack(encoder_input, src_key_padding_mask = src_mask)
        # Take first 2 tokens
        decoded_tokens = encoder_output[:, :2 , :] 

        return self.reverse_embedder(decoded_tokens)
