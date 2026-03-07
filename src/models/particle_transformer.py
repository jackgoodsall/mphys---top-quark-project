import torch.nn as nn
import torch.nn.functional as F
import torch
import sys
from typing import List
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.components.attention_layers import ParticleAttentionBlock, MIParticleAttentionBlock, InteractionDimReducer, ParticleGatingModule
from src.models.components.masked_former_tasks import *
from src.models.components.matcher import *
from typing import Dict, Optional, Union


class ParticleBinaryClassificaitionHead(nn.Module):
    def __init__(self, input_size: int,
                 hidden_sizes: List[int],
                 p_dropout : float,
                 activation_function: str,
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

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        for layer in self.linear_layers[: -1]:
            x = layer(x)
            x = self.activation_function(x)
            x = self.dropout(x)
        return self.linear_layers[-1](x)


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
            # Add eps=1e-6 to LayerNorm for numerical stability
            self.layers.extend([ nn.LayerNorm(size_1, eps=1e-6), nn.Linear(size_1, size_2), nn.GELU(), nn.Dropout(p_dropout)])
        self.layers.append(nn.Linear(size_2, embedding_size))
        
    def forward(self, X, src_mask = None):
        # Clip input to prevent extreme values from causing NaNs in Linear layers
        X = torch.clamp(X, min=-100.0, max=100.0)
        
        # Check for NaNs in INPUT to embedder
        if torch.isnan(X).any():
            nan_count = torch.isnan(X).sum().item()
            raise RuntimeError(f"NaN in ParticleEmbedder INPUT (after clipping)! {nan_count} NaNs out of {X.numel()}")
        
        for i, layer in enumerate(self.layers):
            X = layer(X)
            # Debug: check for NaN after each layer
            if torch.isnan(X).any():
                nan_count = torch.isnan(X).sum().item()
                layer_name = layer.__class__.__name__
                raise RuntimeError(f"NaN after ParticleEmbedder layer {i} ({layer_name})! {nan_count} NaNs out of {X.numel()}")
        
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





class InteractionEmbedder(nn.Module):
    def __init__(self,
        input_features,
        hidden_layers,
        output_size,
        p_dropout):
        super().__init__()
        hidden_sizes = [input_features] + hidden_layers 
        self.layers =  nn.ModuleList([])
        for i, (size_1, size_2) in enumerate(zip(hidden_sizes[:-1], hidden_sizes[1:])):
            # First BatchNorm: slower momentum + more eps to handle extreme interaction values
            # Later BatchNorms: standard settings
            if i == 0:
                bn = nn.BatchNorm1d(size_1, eps=1e-4, momentum=0.01, affine=True)  
            else:
                bn = nn.BatchNorm1d(size_1, eps=1e-5, momentum=0.1, affine=True)
            self.layers.extend([bn, nn.Conv1d(size_1, size_2, kernel_size=1), nn.GELU(), nn.Dropout(p_dropout)])
        self.layers.append(nn.Conv1d(size_2, output_size, kernel_size=1))

    def forward(self, x, src_mask = None):
        """
        x: [B, N, N, input_features]
        returns: [B, N, N, output_size]
        """
        B, N, M, F = x.shape  # F = input_features

        # Clip extreme values (m² can be huge, kT too) to prevent BatchNorm corruption
        # Features: [delta_R, kT, z, m²] - keep reasonable range for batch norm
        x = torch.clamp(x, min=-100.0, max=100.0)

        # Flatten pair (i, j) and arrange for Conv1d: [B*N*M, C_in, L]
        x = x.view(B * N * M, F)       # [B*N*M, F]
        x = x.unsqueeze(-1)            # [B*N*M, F, 1]

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
                 use_mia_encoder: bool = False,
                 n_mia_layers: int = 5,
                 mia_interaction_dim: int = 64,
                 use_particle_gating: bool = False,
                 n_gating_layers: int = 2,
                 n_gate_queries: int = 1,
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
        self.use_mia_encoder = use_mia_encoder

        if use_mia_encoder:
            # MIParT-style encoder:
            #   K x MIParticleAttentionBlock  (high-dim interaction D1=mia_interaction_dim)
            #   InteractionDimReducer         (D1 -> n_heads = D2)
            #   L x ParticleAttentionBlock    (low-dim interaction D2=n_heads)
            # where K + L = n_encoder_layers, K = n_mia_layers
            n_part_layers = max(0, n_encoder_layers - n_mia_layers)
            self.mia_encoder_stack = nn.ModuleList(
                [MIParticleAttentionBlock(embedding_size, dim_ff,
                                         p_dropout, mia_dim=mia_interaction_dim)
                 for _ in range(n_mia_layers)]
            )
            self.interaction_reducer = InteractionDimReducer(mia_interaction_dim, n_heads)
            self.encoder_stack = nn.ModuleList(
                [ParticleAttentionBlock(embedding_size, dim_ff,
                                        n_heads, p_dropout, pair_wise_dim=n_heads)
                 for _ in range(n_part_layers)]
            )
        else:
            # Original ParT-style encoder
            self.encoder_stack = nn.ModuleList(
                [ParticleAttentionBlock(embedding_size, dim_ff,
                                        n_heads, p_dropout, pair_wise_dim=8)
                 for _ in range(n_encoder_layers)]
            )

        # Target tokens (learnable query embeddings)
        # randn (zero-centered) rather than rand (positive-biased) so queries
        # start distinguishable and the matcher can form meaningful assignments
        self.target_tokens = nn.Parameter(
            torch.randn((self.number_class_tokens, embedding_size)) * 0.02
        )

        # Type-conditioned queries: first n_top_queries are top-designated,
        # remainder are W-designated.  Type embeddings are added to queries
        # before the decoder so the model can specialise each query group.
        n_top_queries = kwargs.get('n_top_queries', number_class_tokens // 2)
        self.type_embeddings = nn.Embedding(2, embedding_size)  # 0=top, 1=W
        self.register_buffer('query_type_ids',
            torch.cat([torch.zeros(n_top_queries, dtype=torch.long),
                       torch.ones(number_class_tokens - n_top_queries, dtype=torch.long)]))

        # Store query split for hierarchical decoding
        self.n_top_queries = n_top_queries
        self.n_w_queries = number_class_tokens - n_top_queries
        self.hierarchical_decoding = kwargs.get('hierarchical_decoding', False)

        # Decoder
        if self.hierarchical_decoding:
            n_phase_layers = n_decoder_layers // 2
            self.w_decoder_stack = nn.ModuleList(
                [nn.TransformerDecoderLayer(
                    embedding_size, n_heads, dim_ff, p_dropout,
                    activation=activation_function, batch_first=True
                ) for _ in range(n_phase_layers)]
            )
            self.top_decoder_stack = nn.ModuleList(
                [nn.TransformerDecoderLayer(
                    embedding_size, n_heads, dim_ff, p_dropout,
                    activation=activation_function, batch_first=True
                ) for _ in range(n_phase_layers)]
            )
        else:
            self.decoder_stack = nn.ModuleList(
                [nn.TransformerDecoderLayer(
                    embedding_size, n_heads, dim_ff, p_dropout,
                    activation=activation_function, batch_first=True
                ) for _ in range(n_decoder_layers)]
            )

        # Particle gating (optional): scale encoder memory by learned per-particle relevance
        self.particle_gating = (
            ParticleGatingModule(embedding_size, n_gating_layers, n_gate_queries, n_heads, dim_ff)
            if use_particle_gating else None
        )
        self._final_layer_id = n_decoder_layers - 1

        # Build prediction heads from task registry
        self.prediction_heads = self._build_prediction_heads(embedding_size)

        # Pre-compute which output names are needed per decoder layer.
        # Layers where a task's layer_weight == 0 skip that task's heads,
        # avoiding unnecessary forward passes.
        self._layer_output_map = self._build_layer_output_map(n_decoder_layers)

        # Matching setup
        if self.use_hungarian_matching:
            self.matcher = create_matcher(
                matching_solver=matching_solver,
                num_queries=number_class_tokens,
                max_targets=max_targets,
            )
    
    def _build_layer_output_map(self, n_decoder_layers: int) -> Dict[int, set]:
        """
        Pre-compute which output names are needed at each decoder layer.

        An output is needed at layer i if any task that produces it has a
        non-zero layer_weight for i.  The final layer always includes all
        outputs (needed for the cost matrix and test saving).
        """
        final_layer = n_decoder_layers - 1
        output_map: Dict[int, set] = {}
        for i in range(n_decoder_layers):
            needed: set = set()
            for task in self.task_registry.tasks.values():
                lw = task.config.get_layer_weight(i)
                if lw != 0 or i == final_layer:
                    needed.update(task.config.output_names)
            output_map[i] = needed
        return output_map

    def _build_prediction_heads(self, embedding_size: int) -> nn.ModuleDict:
        """
        Build prediction heads from task registry.
        Each task defines what outputs it needs.
        """
        heads = nn.ModuleDict()
        
        # Collect all unique output names from all tasks
        output_specs = {}  # {output_name: (output_dim, head_norm)}

        for task in self.task_registry.tasks.values():
            for output_name in task.config.output_names:
                if output_name not in output_specs:
                    output_dim = task.config.output_dims.get(output_name)
                    output_specs[output_name] = (output_dim, task.config.head_norm)

        # Build heads for each output type
        for output_name, (output_dim, head_norm) in output_specs.items():
            if output_name == 'mask_predictions':
                # Special case: computed via einsum with memory
                heads[output_name] = nn.Identity()
            elif output_dim is not None:
                layers = []
                if head_norm:
                    layers.append(nn.LayerNorm(embedding_size))
                layers.extend([
                    nn.Linear(embedding_size, embedding_size),
                    nn.GELU(),
                    nn.Linear(embedding_size, output_dim),
                ])
                heads[output_name] = nn.Sequential(*layers)
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
        
        # Input validation: check for NaNs in raw inputs (exclude infs which might be intentional padding)
        if torch.isnan(jet).any():
            nan_count = torch.isnan(jet).sum().item()
            nan_pct = 100.0 * nan_count / jet.numel()
            raise RuntimeError(f"NaN in input jet features! {nan_count} NaNs ({nan_pct:.2f}% of {jet.numel()} total values)")
        if torch.isnan(interactions).any():
            nan_count = torch.isnan(interactions).sum().item()
            nan_pct = 100.0 * nan_count / interactions.numel()
            raise RuntimeError(f"NaN in input interactions! {nan_count} NaNs ({nan_pct:.2f}% of {interactions.numel()} total values)")
        
        # Check for extreme values that might cause overflow
        jet_max = jet.abs().max().item()
        if jet_max > 1e6:
            raise RuntimeError(f" Extreme value in jet features: max={jet_max:.2e}, this will cause NaN in embedder")
        
        # Embed
        jet = self.particle_embedder(jet, src_mask=~src_mask)
        interactions = self.interaction_embedder(interactions, src_mask=~src_mask)
        
        # NaN detection after embeddings (-inf is allowed for attention masking)
        if torch.isnan(jet).any():
            raise RuntimeError(f"NaN after particle embedder!")
        if torch.isnan(interactions).any():
            # Count actual NaNs vs -inf (which is expected for masking)
            nan_mask = torch.isnan(interactions)
            raise RuntimeError(f"NaN after interaction embedder! Found {nan_mask.sum()} NaN values")
        
        B, N, F = jet.shape
        
        # Encode
        memory = jet
        if self.use_mia_encoder:
            # Phase 1: MIA blocks with high-dim interactions
            for layer in self.mia_encoder_stack:
                memory = layer(memory, interactions)
            # Compress interactions from D_mia -> n_heads for P-MHA blocks.
            # Conv1d with mixed-sign weights turns -inf (padding marker) into NaN
            # (-inf * w_pos + -inf * w_neg = -inf + +inf = NaN).  Zero the padding
            # positions before the linear reduction then restore -inf afterwards.
            padding_col_mask = ~src_mask[:, None, None, :]  # [B,1,1,N] True=padding col
            interactions_finite = interactions.masked_fill(padding_col_mask, 0.0)
            interactions = self.interaction_reducer(interactions_finite)
            interactions = interactions.masked_fill(padding_col_mask, float('-inf'))
        # Phase 2 (or full encoder for non-MIA): P-MHA blocks
        for layer in self.encoder_stack:
            memory = layer(memory, interactions)
        
        # Particle gating: scale encoder memory by learned per-particle relevance
        gate_relevance = None
        if self.particle_gating is not None:
            gate_relevance, memory = self.particle_gating(memory, src_key_padding_mask=~src_mask)

        # Initialize queries with type embeddings
        type_emb = self.type_embeddings(self.query_type_ids)  # [Q, D]
        tgt = (self.target_tokens + type_emb).unsqueeze(0).expand(B, -1, -1)
        layer_outputs = {}
        
        # Decode
        if self.hierarchical_decoding:
            # Split initial queries by type
            w_tgt = tgt[:, self.n_top_queries:, :]   # [B, Q_W, D]
            top_tgt = tgt[:, :self.n_top_queries, :]  # [B, Q_top, D]
            n_w_layers = len(self.w_decoder_stack)

            # --- Phase 1: W decoding ---
            for i, w_layer in enumerate(self.w_decoder_stack):
                w_tgt = w_layer(w_tgt, memory, memory_key_padding_mask=~src_mask)
                combined = torch.cat([top_tgt, w_tgt], dim=1)  # top_tgt still at init
                layer_outputs[i] = self._compute_layer_outputs(combined, memory, layer_id=i, gate_relevance=gate_relevance)

            # Build extended memory once from final W states
            w_valid = src_mask.new_ones(B, self.n_w_queries)
            extended_src_mask = torch.cat([src_mask, w_valid], dim=1)
            extended_memory = torch.cat([memory, w_tgt], dim=1)  # [B, N+Q_W, D]

            # --- Phase 2: Top decoding ---
            for j, top_layer in enumerate(self.top_decoder_stack):
                layer_id = n_w_layers + j
                top_tgt = top_layer(top_tgt, extended_memory,
                                    memory_key_padding_mask=~extended_src_mask)
                combined = torch.cat([top_tgt, w_tgt], dim=1)  # w_tgt frozen
                layer_outputs[layer_id] = self._compute_layer_outputs(combined, memory, layer_id=layer_id, gate_relevance=gate_relevance)
        else:
            for i, layer in enumerate(self.decoder_stack):
                tgt = layer(tgt, memory, memory_key_padding_mask=~src_mask)

                # Only compute heads needed at this layer (zero-weight layers are skipped)
                layer_outputs[i] = self._compute_layer_outputs(tgt, memory, layer_id=i, gate_relevance=gate_relevance)
        
        # Apply matching using task registry
        if self.use_hungarian_matching and targets is not None:
            layer_outputs = self._match_and_permute_outputs(layer_outputs, targets)
        
        if last_output_only:
            final_layer = max(layer_outputs.keys())
            return {final_layer: layer_outputs[final_layer]}
        
        return layer_outputs
    
    def _compute_layer_outputs(
        self,
        queries: torch.Tensor,  # [B, num_queries, embedding_size]
        memory: torch.Tensor,   # [B, N_particles, embedding_size]
        layer_id: int = 0,
        gate_relevance: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute outputs for tasks needed at this layer.
        Layers with layer_weight==0 for a task skip that task's heads,
        avoiding unnecessary forward passes (e.g. kinematics at early layers).
        """
        needed = self._layer_output_map[layer_id]

        outputs = {}
        for output_name, head in self.prediction_heads.items():
            if output_name not in needed:
                continue
            if output_name == 'mask_predictions':
                outputs[output_name] = torch.einsum("bnd,bmd->bnm", queries, memory)
            else:
                outputs[output_name] = head(queries)

        if gate_relevance is not None and layer_id == self._final_layer_id:
            outputs['gate_relevance'] = gate_relevance

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

        # Edge case: every event in the batch has zero reconstructable objects.
        # Return all-False valid mask and empty-dimension target tensors so
        # the matcher (which already handles T_max==0) can proceed cleanly.
        if T_max == 0:
            P = targets_list[0]['jet_valid_mask'].shape[0]
            batched: Dict[str, torch.Tensor] = {
                'jet_mask_true': torch.zeros(B, 0, P, device=device),
                'jet_valid_mask': torch.stack(
                    [t['jet_valid_mask'] for t in targets_list]
                ),
            }
            if 'target_kinematics' in targets_list[0]:
                D = targets_list[0]['target_kinematics'].shape[-1]
                batched['target_kinematics'] = torch.zeros(B, 0, D, device=device)
                batched['kinematics'] = batched['target_kinematics']
            if 'classes' in targets_list[0]:
                batched['classes'] = torch.zeros(B, 0, dtype=torch.long, device=device)
            target_valid_mask = torch.zeros(B, 0, dtype=torch.bool, device=device)
            return batched, target_valid_mask

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
            # Old path: List[Dict] — collate on GPU (fallback)
            targets_batched, target_valid_mask = self._collate_targets(targets)
        elif 'target_valid_mask' in targets:
            # Fast path: already pre-collated by masked_former_collate_fn on CPU
            targets_batched = targets
            target_valid_mask = targets['target_valid_mask'].to(
                next(iter(targets.values())).device
            )
        else:
            # Backward compat: Dict targets without target_valid_mask (fixed T)
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

            # Type-partitioned matching: add large penalty for cross-type
            # assignments so top queries can only match top targets and
            # W queries can only match W targets.
            if hasattr(self, 'query_type_ids') and 'classes' in targets_batched:
                classes = targets_batched['classes']  # [B, T_max]
                # target type: top (CLASS_TOP=1) → 0, W (CLASS_W=2) → 1
                target_type = (classes == CLASS_W).long()             # [B, T_max]
                query_type = self.query_type_ids                      # [Q]

                # [B, Q, T_max] — True where query type != target type
                type_mismatch = (query_type[None, :, None] != target_type[:, None, :])

                cost_matrix = cost_matrix + type_mismatch.float() * 1e6

            query_valid = targets_batched.get('query_mask')

            pred_idxs = self.matcher(
                costs=cost_matrix,
                object_valid_mask=target_valid_mask,
                query_valid_mask=query_valid
            )

        pred_idxs = pred_idxs.to(cost_matrix.device)
        B, Q = pred_idxs.shape

        # ---- 3. Permute outputs with batched gather (preserves gradients) ----
        # Group layers by output name, stack [n_layers, B, Q, D], run one
        # torch.gather per output name — reduces ~32 kernel launches to ~4.
        layer_ids = sorted(decoder_outputs.keys())
        permuted_outputs = {lid: {} for lid in layer_ids}

        # Collect all output names that appear in at least one layer
        all_output_names: set = set()
        for lid in layer_ids:
            all_output_names.update(decoder_outputs[lid].keys())

        # gate_relevance is [B, N] (per-particle), not [B, Q, D] — pass through directly
        # without permutation (particle order is fixed and doesn't need reordering).
        NON_QUERY_OUTPUTS = {'gate_relevance', '__targets__'}

        for output_name in all_output_names:
            layers_with = [lid for lid in layer_ids if output_name in decoder_outputs[lid]]
            if not layers_with:
                continue

            if output_name in NON_QUERY_OUTPUTS:
                for lid in layers_with:
                    permuted_outputs[lid][output_name] = decoder_outputs[lid][output_name]
                continue

            stacked = torch.stack([decoder_outputs[lid][output_name] for lid in layers_with])
            # stacked: [L, B, Q, D]
            D = stacked.shape[-1]
            idx = (pred_idxs
                   .unsqueeze(0)          # [1, B, Q]
                   .unsqueeze(-1)         # [1, B, Q, 1]
                   .expand(len(layers_with), -1, -1, D))  # [L, B, Q, D]
            gathered = torch.gather(stacked, 2, idx)      # [L, B, Q, D]
            for j, lid in enumerate(layers_with):
                permuted_outputs[lid][output_name] = gathered[j]

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
