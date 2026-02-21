import torch.nn as nn
import torch.nn.functional as F
import torch
import sys
from typing import List
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import copy
from src.models.components.attention_layers import ParticleAttentionBlock, ClassAttentionBlock, DecoderAttentionBlock
from src.models.components.masked_former_tasks import *
from src.models.components.matcher import *
from typing import Dict, Optional, Union


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
        
    def forward(self, X, src_mask = None):

        for layers in self.layers:
            X = layers(X)
        
        if src_mask is not None:
            # mask -> [B, N, 1] → broadcasts over features
            mask = src_mask.unsqueeze(-1).bool()
            X = X.masked_fill(mask, 0.0)

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


class InteractionEmbedder(nn.Module):
    def __init__(self,
        input_features,
        hidden_layers,
        output_size,
        p_dropout):
        super().__init__()
        hidden_sizes = [input_features] + hidden_layers 
        self.layers =  nn.ModuleList([])
        for size_1, size_2 in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            self.layers.extend([ nn.BatchNorm1d(size_1), nn.Conv1d(size_1, size_2, kernel_size=1), nn.GELU(), nn.Dropout(p_dropout)])
        self.layers.append(nn.Conv1d(size_2, output_size, kernel_size=1))

    def forward(self, x, src_mask = None):
        """
        x: [B, N, N, input_features]
        returns: [B, N, N, output_size]
        """
        B, N, M, F = x.shape  # F = input_features

        # Flatten pair (i, j) and arrange for Conv1d: [B*N*M, C_in, L]
        x = x.view(B * N * M, F)       # [B*N*M, F]
        x = x.unsqueeze(-1)    
        x /= 3        # [B*N*M, F, 1]

        for layer in self.layers:
            if isinstance(layer, nn.LayerNorm):
                # LayerNorm expects last dim = normalized_shape (channels)
                # Current x: [B*N*M, C, 1] -> make channels last: [B*N*M, 1, C]
                x = x.transpose(1, 2)  # [B*N*M, 1, C]
                x = layer(x)
                x = x.transpose(1, 2)  # back to [B*N*M, C, 1]
            else:
                x = layer(x)

        # Remove length dim and reshape back to [B, N, N, output_size]
        x = x.squeeze(-1)              # [B*N*M, output_size]
        x = x.view(B, N, M, -1)        # [B, N, N, output_size]

        x=  x.permute(0, 3, 1, 2)
        out = x
        if src_mask is not None:
            mask_expanded_T = src_mask[:, None, None, :] # [B,1,1,N]
            out = out.masked_fill(mask_expanded_T.bool(), float("-inf"))
        return out


class ReconstructionInteractionTransformer(nn.Module):
    def __init__(self,
                 particle_embedder,
                 interaction_embedder,
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
        self.interaction_embedder = interaction_embedder
        self.reverse_embedder = reverse_embedder

        self.encoder_stack = nn.ModuleList(
            [ParticleAttentionBlock(embedding_size, dim_ff,
                                    n_heads, p_dropout,pair_wise_dim=8) for _ in range(n_encoder_layers)]
        )
        self.decoder_layer = nn.TransformerDecoderLayer(
            embedding_size, n_heads, dim_ff, p_dropout, activation = activation_function,
            batch_first= True
        )
        self.decoder_stack = nn.TransformerDecoder(self.decoder_layer, n_decoder_layers)

        self.tgt_tokens = nn.Parameter(torch.zeros(( 2,
                                  embedding_size)))

    def forward(self, X):
        # Unpack
        jet = X["jet"]
        interactions = X["interactions"]
        src_mask = X["src_mask"]

        # Embed
        jet = self.particle_embedder(jet)
        interactions = self.interaction_embedder(interactions)
        
        B, N, F = jet.shape
    
        tgt_tokens = self.tgt_tokens.expand(B, 2, F)



        # Encode
        for layer in self.encoder_stack:
            memory = layer(jet, interactions)
    
        # Decorder  
        decoded_tokens = self.decoder_stack(tgt_tokens, 
                                            memory,
                                            memory_key_padding_mask =src_mask)

        # Regression
        outputs = self.reverse_embedder(decoded_tokens)
        return outputs
        


class ReconstructionPart(nn.Module):
    def __init__(self,
                 particle_embedder,
                 interaction_embedder,
                 reverse_embedder,
                 embedding_size,
                 n_encoder_layers,
                 n_decoder_layers,
                 n_heads,
                 out_dimensions,
                 dim_ff,
                 p_dropout,
                 number_class_tokens,
                 activation_function = "gelu",
                 reconstruct_Ws = False,
                 use_hungarian_matching = False,
                 ):
        super().__init__()

        self.reconstruct_Ws = reconstruct_Ws
        self.particle_embedder = particle_embedder
        self.interaction_embedder = interaction_embedder
        self.reverse_embedder = reverse_embedder
        self.w_boson = copy.deepcopy(reverse_embedder)
        self.number_class_tokens = number_class_tokens
        self.use_hungarian_matching = use_hungarian_matching

        self.encoder_stack = nn.ModuleList(
            [ParticleAttentionBlock(embedding_size, dim_ff,
                                    n_heads, p_dropout,pair_wise_dim=8) for _ in range(n_encoder_layers)]
        )
        self.decoder_stack = nn.ModuleList([
           ClassAttentionBlock(
            embedding_size, n_heads, dim_ff, p_dropout
        ) for _ in range(n_decoder_layers)])
        self.particle_tokens = nn.Parameter(torch.rand(number_class_tokens, embedding_size) * 0.01)

    def forward(self, X):
        # Unpack
        jet = X["jet"]
        interactions = X["interactions"]
        src_mask = X["src_mask"]

        # Embed
        jet = self.particle_embedder(jet, src_mask = src_mask)
        interactions = self.interaction_embedder(interactions, src_mask = src_mask)
        
        B, N, F = jet.shape

        cls_tkns = self.particle_tokens.expand(B, self.number_class_tokens, self.particle_tokens.shape[-1])

        # Encode
        for layer in self.encoder_stack:
            memory = layer(jet, interactions)
    
        # Decorder  
        outputs = []
        for layer in self.decoder_stack:

            cls_tkns = layer(memory, 
                                            cls_tkns,
                                            src_mask =src_mask)
            outputs.append(cls_tkns)
            

        # Regression
        tops = self.reverse_embedder(outputs[-1])
        if self.reconstruct_Ws:
            if self.use_hungarian_matching:
                return tops
            W_bosons = self.w_boson(outputs[-2])
            return {
                "top":tops, 
                "W":W_bosons
            }
        return tops
    




class MaskedReconstructionPart(nn.Module):
    def __init__(self,
                 particle_embedder,
                 interaction_embedder,
                 embedding_size,
                 n_encoder_layers,
                 n_decoder_layers,
                 n_heads,
                 dim_ff,
                 p_dropout,
                 number_class_tokens,
                 task_registry: TaskRegistry,
                 activation_function="gelu",
                 use_hungarian_matching=True,
                 matching_solver: str = "gpu_bruteforce",
                 max_targets: int = 5,
                 *args,
                 **kwargs
                 ):
        super().__init__()

        # Model parameters
        self.particle_embedder = particle_embedder
        self.interaction_embedder = interaction_embedder
        self.number_class_tokens = number_class_tokens
        self.use_hungarian_matching = use_hungarian_matching
        self.task_registry = task_registry

        # Encoder
        self.encoder_stack = nn.ModuleList(
            [ParticleAttentionBlock(embedding_size, dim_ff,
                                    n_heads, p_dropout, pair_wise_dim=8)
             for _ in range(n_encoder_layers)]
        )

        # Target tokens (learnable query embeddings)
        self.target_tokens = nn.Parameter(
            torch.rand((self.number_class_tokens, embedding_size)) * 0.01
        )

        # Decoder
        self.decoder_stack = nn.ModuleList(
            [nn.TransformerDecoderLayer(
                embedding_size, n_heads, dim_ff, p_dropout,
                activation=activation_function, batch_first=True
            ) for _ in range(n_decoder_layers)]
        )

        # Build prediction heads from task registry
        self.prediction_heads = self._build_prediction_heads(embedding_size)

        # Matching setup
        if self.use_hungarian_matching:
            self.matcher = create_matcher(
                matching_solver=matching_solver,
                num_queries=number_class_tokens,
                max_targets=max_targets,
            )
    
    def _build_prediction_heads(self, embedding_size: int) -> nn.ModuleDict:
        """
        Build prediction heads from task registry.
        Each task defines what outputs it needs.
        """
        heads = nn.ModuleDict()
        
        # Collect all unique output names from all tasks
        output_specs = {}  # {output_name: output_dim}
        
        for task in self.task_registry.tasks.values():
            for output_name in task.config.output_names:
                if output_name not in output_specs:
                    # Get dimension from task config
                    output_dim = task.config.output_dims.get(output_name)
                    output_specs[output_name] = output_dim
        
        # Build heads for each output type
        for output_name, output_dim in output_specs.items():
            if output_name == 'mask_predictions':
                # Special case: computed via einsum with memory
                heads[output_name] = nn.Identity()
            elif output_dim is not None:
                # Standard prediction head
                heads[output_name] = nn.Sequential(
                    nn.Linear(embedding_size, embedding_size),
                    nn.GELU(),
                    nn.Linear(embedding_size, output_dim)
                )
            else:
                raise ValueError(
                    f"Task requires output '{output_name}' but didn't specify "
                    f"dimension in output_dims. Add it to TaskConfig."
                )
        
        return heads
    
    def forward(self, X, last_output_only=False):
        # Unpack
        jet = X["jet"]
        interactions = X["interactions"]
        src_mask = X["src_mask"]
        targets = X.get("targets", None)
        
        # Embed
        jet = self.particle_embedder(jet, src_mask=~src_mask)
        interactions = self.interaction_embedder(interactions, src_mask=~src_mask)
        
        B, N, F = jet.shape
        
        # Encode
        memory = jet
        for layer in self.encoder_stack:
            memory = layer(memory, interactions)
        
        # Initialize queries
        tgt = self.target_tokens.expand(B, -1, -1)
        layer_outputs = {}
        
        # Decode
        for i, layer in enumerate(self.decoder_stack):
            tgt = layer(tgt, memory, memory_key_padding_mask=~src_mask)
            
            # Generate outputs dynamically from task registry
            layer_outputs[i] = self._compute_layer_outputs(tgt, memory)
        
        # Apply matching using task registry
        if self.use_hungarian_matching and targets is not None:
            layer_outputs = self._match_and_permute_outputs(layer_outputs, targets)
        
        if last_output_only:
            final_layer = max(layer_outputs.keys())
            return {0: layer_outputs[final_layer]}
        
        return layer_outputs
    
    def _compute_layer_outputs(
        self, 
        queries: torch.Tensor,  # [B, num_queries, embedding_size]
        memory: torch.Tensor    # [B, N_particles, embedding_size]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute outputs for all tasks.
        Output names come from task registry, not hardcoded.
        """
        outputs = {}
        
        # Generate each output type that tasks need
        for output_name, head in self.prediction_heads.items():
            if output_name == 'mask_predictions':
                # Special case: cross-attention with memory
                outputs[output_name] = torch.einsum("bnd,bmd->bnm", queries, memory)
            else:
                # Apply prediction head to queries
                outputs[output_name] = head(queries)
        
        return outputs
    def _collate_targets(
        self,
        targets_list: list,
    ) -> tuple:
        """
        Collate List[Dict] targets into batched tensors with padding to T_max.

        Returns:
            targets_batched: Dict[str, Tensor] with all targets padded to T_max
            target_valid_mask: [B, T_max] bool indicating which targets are real
        """
        B = len(targets_list)
        device = targets_list[0]['jet_mask_true'].device

        # Find T per event and T_max
        T_per_event = [t['jet_mask_true'].shape[0] for t in targets_list]
        T_max = max(T_per_event)

        # Pad and stack jet_mask_true: [T_i, P] -> [B, T_max, P]
        jmt_list = []
        for t in targets_list:
            T_i = t['jet_mask_true'].shape[0]
            pad_size = T_max - T_i
            jmt = t['jet_mask_true']
            if jmt.ndim == 1:
                jmt = jmt.unsqueeze(0)
            if pad_size > 0:
                jmt = F.pad(jmt, (0, 0, 0, pad_size), value=0.0)
            jmt_list.append(jmt)

        batched = {
            'jet_mask_true': torch.stack(jmt_list),  # [B, T_max, P]
            'jet_valid_mask': torch.stack(
                [t['jet_valid_mask'] for t in targets_list]
            ),  # [B, P]
        }

        # Pad and stack target_kinematics if present
        if 'target_kinematics' in targets_list[0]:
            tk_list = []
            for t in targets_list:
                tk = t['target_kinematics']
                if tk.ndim == 1:
                    tk = tk.unsqueeze(0)
                T_i = tk.shape[0]
                pad_size = T_max - T_i
                if pad_size > 0:
                    tk = F.pad(tk, (0, 0, 0, pad_size), value=0.0)
                tk_list.append(tk)
            batched['target_kinematics'] = torch.stack(tk_list)
            batched['kinematics'] = batched['target_kinematics']  # alias

        # Pad and stack classes if present
        if 'classes' in targets_list[0]:
            cls_list = []
            for t in targets_list:
                cls = t['classes']
                T_i = cls.shape[0]
                pad_size = T_max - T_i
                if pad_size > 0:
                    cls = F.pad(cls, (0, pad_size), value=CLASS_NULL)
                cls_list.append(cls)
            batched['classes'] = torch.stack(cls_list)

        # Build target_valid_mask [B, T_max]
        target_valid_mask = torch.zeros(B, T_max, dtype=torch.bool, device=device)
        for i, T_i in enumerate(T_per_event):
            target_valid_mask[i, :T_i] = True

        return batched, target_valid_mask

    def _match_and_permute_outputs(
        self,
        decoder_outputs: Dict[int, Dict[str, torch.Tensor]],
        targets: Union[Dict[str, torch.Tensor], list],
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Returns a NEW permuted decoder_outputs dict with __targets__ injected.

        Handles both:
        - Dict targets (backward compat, fixed T)
        - List[Dict] targets (new path, variable T per event)

        After matching, pads targets from T to Q, builds obj_valid_mask,
        and injects padded_targets into each layer dict under "__targets__".
        """
        # ---- 1. Collate targets if needed ----
        if isinstance(targets, list):
            targets_batched, target_valid_mask = self._collate_targets(targets)
        else:
            targets_batched = targets
            jmt = targets_batched['jet_mask_true']
            if jmt.ndim == 2:
                jmt = jmt.unsqueeze(1)
            B, T = jmt.shape[0], jmt.shape[1]
            target_valid_mask = torch.ones(
                B, T, dtype=torch.bool, device=jmt.device
            )
            # Ensure 3D for consistency
            if targets_batched['jet_mask_true'].ndim == 2:
                targets_batched['jet_mask_true'] = targets_batched[
                    'jet_mask_true'
                ].unsqueeze(1)

        # ---- 2. Compute matching indices (no gradients) ----
        with torch.no_grad():
            final_layer = max(decoder_outputs.keys())
            final_output = decoder_outputs[final_layer]

            cost_matrix = self.task_registry.compute_total_cost(
                predictions=final_output,
                targets=targets_batched
            )  # [B, Q, T_max]

            query_valid = targets_batched.get('query_mask')

            pred_idxs = self.matcher(
                costs=cost_matrix,
                object_valid_mask=target_valid_mask,
                query_valid_mask=query_valid
            )

        pred_idxs = pred_idxs.to(cost_matrix.device)
        B, Q = pred_idxs.shape

        # ---- 3. Permute outputs (preserves gradients) ----
        batch_idxs = torch.arange(B, device=pred_idxs.device).unsqueeze(1)

        permuted_outputs = {}
        for layer_id, layer_dict in decoder_outputs.items():
            permuted_outputs[layer_id] = {}
            for output_name, output_tensor in layer_dict.items():
                permuted_outputs[layer_id][output_name] = \
                    output_tensor[batch_idxs, pred_idxs]

        # ---- 4. Pad targets from T_max to Q ----
        T_max = target_valid_mask.shape[1]
        pad_q = Q - T_max  # may be 0 if Q == T_max

        # Per-event T from target_valid_mask
        per_event_T = target_valid_mask.sum(dim=1)  # [B]

        # obj_valid_mask: [B, Q] — True for matched real objects
        obj_valid_mask = (
            torch.arange(Q, device=pred_idxs.device).unsqueeze(0)
            < per_event_T.unsqueeze(1)
        )

        padded_targets = {
            'jet_valid_mask': targets_batched['jet_valid_mask'],  # [B, P]
            'obj_valid_mask': obj_valid_mask,  # [B, Q]
        }

        # Pad jet_mask_true [B, T_max, P] -> [B, Q, P]
        jmt = targets_batched['jet_mask_true']  # already 3D
        if pad_q > 0:
            padded_targets['jet_mask_true'] = F.pad(
                jmt, (0, 0, 0, pad_q), value=0.0
            )
        else:
            padded_targets['jet_mask_true'] = jmt

        # Pad target_kinematics if present
        if 'target_kinematics' in targets_batched:
            tk = targets_batched['target_kinematics']
            if tk.ndim == 2:
                tk = tk.unsqueeze(1)
            if pad_q > 0:
                padded_targets['target_kinematics'] = F.pad(
                    tk, (0, 0, 0, pad_q), value=0.0
                )
            else:
                padded_targets['target_kinematics'] = tk
            padded_targets['kinematics'] = padded_targets['target_kinematics']

        # Pad classes if present
        if 'classes' in targets_batched:
            cls = targets_batched['classes']
            if pad_q > 0:
                padded_targets['classes'] = F.pad(
                    cls, (0, pad_q), value=CLASS_NULL
                )
            else:
                padded_targets['classes'] = cls

        # ---- 5. Inject __targets__ into each layer dict ----
        for layer_id in permuted_outputs:
            permuted_outputs[layer_id]['__targets__'] = padded_targets

        return permuted_outputs
