from dataclasses import dataclass
import math
from typing import Optional, Union
from einops import rearrange
import torch
from torch.nn import functional as F
from megatron.core import parallel_state
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb
from megatron.core.utils import divide
from linear_moe.model.common_modules.activations import ACT2FN

@dataclass
class LinearAttentionSubmodules:
    qkv_proj: Union[ModuleSpec, type] = None
    o_gate_proj: Union[ModuleSpec, type] = None
    gk_proj: Union[ModuleSpec, type] = None
    core_linear_attention: Union[ModuleSpec, type] = None
    o_proj: Union[ModuleSpec, type] = None


class LinearAttention(MegatronModule):
    def __init__(
        self,
        config,
        submodules: LinearAttentionSubmodules,
        layer_number=None,
    ):
        super().__init__(config)
        self.config = config
        self.la_module = self.config.la_module
        self.hidden_size = self.config.hidden_size
        self.query_dim = self.config.hidden_size
        self.key_dim = int(self.config.hidden_size * self.config.expand_k)
        self.value_dim = int(self.config.hidden_size * self.config.expand_v)
        self.head_dim = self.config.kv_channels
        self.num_heads = self.config.num_attention_heads
        self.la_gate_fn = self.config.la_gate_fn
        self.layer_number = layer_number

        # For normal attention without groups, num_query_groups == num_attention_heads,
        # so these two will be the same
        self.query_projection_size = self.config.kv_channels * self.config.num_attention_heads
        self.kv_projection_size = self.config.kv_channels * self.config.num_query_groups

        # Per attention head and per partition values.
        tensor_model_parallel_world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.hidden_size_per_attention_head = divide(
            self.query_projection_size, self.config.num_attention_heads
        )
        self.num_attention_heads_per_partition = divide(self.config.num_attention_heads, tensor_model_parallel_world_size)
        self.num_query_groups_per_partition = divide(self.config.num_query_groups, tensor_model_parallel_world_size)

        self.qkv_proj = build_module(
            submodules.qkv_proj,
            self.config.hidden_size,
            self.query_dim+2*self.key_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=(self.config.add_bias_linear or self.config.add_qkv_bias) if self.config.base_model=='qwen2' else self.config.add_bias_linear,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='qkv',
        )
        
        self.o_gate_proj = build_module(
            submodules.o_gate_proj,
            self.config.hidden_size,
            self.query_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=(self.config.add_bias_linear or self.config.add_qkv_bias) if self.config.base_model=='qwen2' else self.config.add_bias_linear,
            skip_bias_add=False,
            is_expert=False,
        )
        
        if self.la_module == 'gla':
            self.gk_proj = build_module(
                submodules.gk_proj,
                self.config,
            )
        
        if self.la_module == 'deltanet':
            self.beta_proj = torch.nn.Linear(self.hidden_size, self.num_heads, bias=False)
        
        if self.la_module == 'gated_deltanet':
            self.beta_proj = torch.nn.Linear(self.hidden_size, self.num_heads, bias=False)
            self.a_proj = torch.nn.Linear(self.hidden_size, self.num_heads, bias=False)
            A = torch.empty(self.num_heads, dtype=torch.float32).uniform_(0, 16)
            A_log = torch.log(A)
            self.A_log = torch.nn.Parameter(A_log)
            self.A_log._no_weight_decay = True
            self.D = torch.nn.Parameter(torch.ones(self.num_heads))
            self.D._no_weight_decay = True
            # hard coded for now
            dt_min = 0.001
            dt_max = 0.1
            dt_init_floor = 1e-4
            dt = torch.exp(
                torch.rand(self.num_heads) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            )
            dt = torch.clamp(dt, min=dt_init_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_bias = torch.nn.Parameter(inv_dt)
            # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
            # name.endswith("bias") in param_grouping.py
            self.dt_bias._no_weight_decay = True
        
        self.core_linear_attention = build_module(
            submodules.core_linear_attention,
            config=self.config,
            expand_k=self.config.expand_k,
            expand_v=self.config.expand_v,
        )
        
        self.o_proj = build_module(
            submodules.o_proj,
            self.query_dim,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear, # false
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name='proj',
        )
        
        self.la_gate_fn = ACT2FN[self.la_gate_fn]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        inference_params=None,
        rotary_pos_emb=None,
        packed_seq_params=None,
    ) -> torch.Tensor:

        qkv, _ = self.qkv_proj(hidden_states)
        o_gate, _ = self.o_gate_proj(hidden_states)
        if self.la_module == 'gla':
            gk = self.gk_proj(hidden_states)
        elif self.la_module == 'deltanet':
            beta = rearrange(self.beta_proj(hidden_states), 'b n h -> b h n').sigmoid()
        elif self.la_module == 'gated_deltanet':
            beta = self.beta_proj(hidden_states.transpose(0, 1)).sigmoid().transpose(0, 1)
            gk = -self.A_log.float().exp() * F.softplus(self.a_proj(hidden_states.transpose(0, 1)).float() + self.dt_bias).transpose(0, 1)
        else:
            beta = None
            gk = None

        q, k, v = torch.split(
            qkv.view(qkv.size()[:-1] + (self.num_attention_heads_per_partition, -1)),
            [self.head_dim, self.head_dim, self.head_dim],
            dim=3,
        )
        
        # dealing with left-padding
        # if attention_mask is not None:
        #     v = v.mul_(attention_mask.unsqueeze(-1))
        
        # For self attention we just duplicate the rotary_pos_emb if it isn't already
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        # ===================================================
        # Adjust key, value, and rotary_pos_emb for inference
        # ===================================================
        _, _, rotary_pos_emb, attn_mask_type = self._adjust_key_value_for_inference(
            inference_params, k, v, rotary_pos_emb
        )

        if packed_seq_params is not None:
            q = q.squeeze(1)
            k = k.squeeze(1)
            v = v.squeeze(1)

        rotary_pos_emb = None # for linear attention
        # ================================================
        # relative positional embedding (rotary embedding)
        # ================================================
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb

            if packed_seq_params is not None:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
            else:
                cu_seqlens_q = cu_seqlens_kv = None
            if self.config.base_model == 'mixtral':
                q = apply_rotary_pos_emb(
                    q, q_pos_emb, fused=self.config.apply_rope_fusion, cu_seqlens=cu_seqlens_q
                )
                k = apply_rotary_pos_emb(
                    k, k_pos_emb, fused=self.config.apply_rope_fusion, cu_seqlens=cu_seqlens_kv
                )
            else:
                q = apply_rotary_pos_emb(
                    q, q_pos_emb, config=self.config, cu_seqlens=cu_seqlens_q,
                )
                k = apply_rotary_pos_emb(
                    k, k_pos_emb, config=self.config, cu_seqlens=cu_seqlens_kv,
                )

            # TODO, can apply positional embedding to value_layer so it has
            # absolute positional embedding.
            # otherwise, only relative positional embedding takes effect
            # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)
        
        # expect q: n b h d
        if self.la_module == 'gla':
            o = self.core_linear_attention(
                q=q,
                k=k,
                v=v,
                gk=gk,
            )
        elif self.la_module == 'deltanet':
            o = self.core_linear_attention(
                q=q,
                k=k,
                v=v,
                beta=beta,
            )
        elif self.la_module == 'gated_deltanet':
            o = self.core_linear_attention(
                q=q,
                k=k,
                v=v,
                beta=beta,
                gk=gk,
            )
        elif self.la_module == 'mixattention':
            o = self.core_linear_attention(
                x=hidden_states,
                q=q,
                k=k,
                v=v,
            )
        else:
            o = self.core_linear_attention(
                q=q,
                k=k,
                v=v,
            )
        
        # o: n b (h d)
        o = o * self.la_gate_fn(o_gate)
        o, bias = self.o_proj(o)

        return o, bias
    
    def _allocate_memory(self, inference_max_sequence_length, batch_size, dtype):
        """Allocate memory to store kv cache during inference."""

        return torch.empty(
            inference_max_sequence_length,
            batch_size,
            self.num_query_groups_per_partition,
            self.hidden_size_per_attention_head,
            dtype=dtype,
            device=torch.cuda.current_device(),
        )

    def _adjust_key_value_for_inference(self, inference_params, key, value, rotary_pos_emb):
        """
        Saves the generated key and value tensors to the end of the buffers in inference_params.
        Returns the full size keys and values from the provided inference_params, as well as
        adjusted rotary_pos_emb.

        Returns a tuple: (key, value, rotary_pos_emb)

        """
        attn_mask_type = AttnMaskType.causal # self.attn_mask_type
        if inference_params is None:
            return key, value, rotary_pos_emb, attn_mask_type

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        is_first_step = False
        if self.layer_number not in inference_params.key_value_memory_dict:
            inf_max_seq_length = inference_params.max_sequence_length
            inf_max_batch_size = inference_params.max_batch_size
            inference_key_memory = self._allocate_memory(
                inf_max_seq_length, inf_max_batch_size, key.dtype
            )
            inference_value_memory = self._allocate_memory(
                inf_max_seq_length, inf_max_batch_size, value.dtype
            )
            inference_params.key_value_memory_dict[self.layer_number] = (
                inference_key_memory,
                inference_value_memory,
            )
            is_first_step = True
        else:
            # Get the pre-allocated buffers for this layer
            inference_key_memory, inference_value_memory = inference_params.key_value_memory_dict[
                self.layer_number
            ]
            attn_mask_type = AttnMaskType.no_mask

        batch_start = inference_params.batch_size_offset
        batch_end = batch_start + key.size(1)
        assert batch_end <= inference_key_memory.size(1)
        sequence_start = inference_params.sequence_len_offset
        sequence_end = sequence_start + key.size(0)
        assert sequence_end <= inference_key_memory.size(0)
        # Copy key and values.
        inference_key_memory[sequence_start:sequence_end, batch_start:batch_end, ...] = key
        inference_value_memory[sequence_start:sequence_end, batch_start:batch_end, ...] = value
        key = inference_key_memory[:sequence_end, batch_start:batch_end, ...]
        value = inference_value_memory[:sequence_end, batch_start:batch_end, ...]

        # adjust the key rotary positional embedding
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            # need to cross check this condition during inference
            # if not set_inference_key_value_memory:
            if not is_first_step:
                # In inference, we compute one token at a time.
                # Select the correct positional embedding
                # (only the last token in the sequence)
                q_pos_emb = q_pos_emb[sequence_end - 1 : sequence_end]
            else:
                # In the first forward pass of inference,
                # we use the entire provided prefix.
                # q_pos_emb here has the rope embeddings of the entire
                # prefix + to-be-generated output so
                # we slice to just the prefix.
                q_pos_emb = q_pos_emb[:sequence_end, :, :, :]
            k_pos_emb = k_pos_emb[:sequence_end, :, :, :]
            rotary_pos_emb = (q_pos_emb, k_pos_emb)

        return key, value, rotary_pos_emb, attn_mask_type