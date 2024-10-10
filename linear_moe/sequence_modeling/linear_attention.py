from dataclasses import dataclass
from typing import Optional, Union
from einops import rearrange
import torch
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb
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
        self.sequence_modeling_module = self.config.sequence_modeling_module
        self.hidden_size = self.config.hidden_size
        self.query_dim = self.config.hidden_size
        self.key_dim = int(self.config.hidden_size * self.config.expand_k)
        self.value_dim = int(self.config.hidden_size * self.config.expand_v)
        self.head_dim = self.config.kv_channels
        self.num_heads = self.config.num_attention_heads
        self.lsm_gate_fn = self.config.lsm_gate_fn

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
        
        if self.sequence_modeling_module == 'gla':
            self.gk_proj = build_module(
                submodules.gk_proj,
                self.config,
            )
        
        if self.sequence_modeling_module == 'deltanet':
            self.beta_proj = torch.nn.Linear(self.hidden_size, self.num_heads, bias=False)
        
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
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name='proj',
        )
        
        self.lsm_gate_fn = ACT2FN[self.lsm_gate_fn]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb=None,
        packed_seq_params=None,
    ) -> torch.Tensor:

        qkv, _ = self.qkv_proj(hidden_states)
        o_gate, _ = self.o_gate_proj(hidden_states)
        if self.sequence_modeling_module == 'gla':
            gk = self.gk_proj(hidden_states)
        else:
            gk = None
        
        if self.sequence_modeling_module == 'deltanet':
            beta = rearrange(self.beta_proj(hidden_states), 'b n h -> b h n').sigmoid()
        else:
            beta = None

        q, k, v = torch.split(
            qkv.view(qkv.size()[:-1] + (self.num_heads, -1)),
            [self.head_dim, self.head_dim, self.head_dim],
            dim=3,
        )
        
        # dealing with left-padding
        if attention_mask is not None:
            v = v.mul_(attention_mask.unsqueeze(-1))
        
        # For self attention we just duplicate the rotary_pos_emb if it isn't already
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        if packed_seq_params is not None:
            q = q.squeeze(1)
            k = k.squeeze(1)
            v = v.squeeze(1)

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
        if self.sequence_modeling_module == 'gla':
            o = self.core_linear_attention(
                q=q,
                k=k,
                v=v,
                gk=gk,
            )
        elif self.sequence_modeling_module == 'deltanet':
            o = self.core_linear_attention(
                q=q,
                k=k,
                v=v,
                beta=beta,
            )
        else:
            o = self.core_linear_attention(
                q=q,
                k=k,
                v=v,
            )
        
        # o: n b (h d)
        o = o * self.lsm_gate_fn(o_gate)
        o, bias = self.o_proj(o)

        return o, bias
