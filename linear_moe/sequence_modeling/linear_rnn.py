from dataclasses import dataclass
from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.enums import AttnMaskType
from linear_moe.model.common_modules.activations import ACT2FN, swish
from linear_moe.sequence_modeling.rwkv6 import LerpLinear


@dataclass
class LinearRNNSubmodules:
    r_proj: Union[ModuleSpec, type] = None
    w_proj: Union[ModuleSpec, type] = None
    k_proj: Union[ModuleSpec, type] = None
    v_proj: Union[ModuleSpec, type] = None
    g_proj: Union[ModuleSpec, type] = None
    q_proj: Union[ModuleSpec, type] = None
    f_proj: Union[ModuleSpec, type] = None
    i_proj: Union[ModuleSpec, type] = None
    core_linear_rnn: Union[ModuleSpec, type] = None
    o_proj: Union[ModuleSpec, type] = None


class LinearRNN(MegatronModule):
    def __init__(
        self,
        config,
        submodules: LinearRNNSubmodules,
        layer_number=None,
    ):
        super().__init__(config)
        self.config = config
        self.sequence_modeling_module = config.sequence_modeling_module
        self.num_heads = config.num_attention_heads

        self.head_dim = self.config.kv_channels
        self.key_dim = int(self.config.hidden_size * self.config.expand_k)
        self.value_dim = int(self.config.hidden_size * self.config.expand_v)
        self.hidden_size = self.config.hidden_size
        self.rwkv6_la_proj_low_rank_dim = self.config.rwkv6_la_proj_low_rank_dim
        self.head_qk_dim = self.key_dim // self.num_heads
        
        self.la_gate_fn = self.config.la_gate_fn
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        
        if config.hidden_size is not None and config.num_attention_heads is not None:
            expand_ratio = self.hidden_size // self.num_heads
        self.expand_ratio = expand_ratio
        self.forget_dim = int(self.num_heads * self.expand_ratio)
        self.input_dim = self.hidden_size
        
        if self.sequence_modeling_module == 'rwkv6':
            self.x_proj = nn.Sequential(
                LerpLinear(self.hidden_size, self.rwkv6_la_proj_low_rank_dim * 5),
                nn.Tanh(),
                nn.Linear(self.rwkv6_la_proj_low_rank_dim * 5, self.hidden_size, bias=False)
            )
            
            self.x_bias = nn.Parameter(torch.zeros(5, self.hidden_size))
            
            self.r_proj = build_module(
                submodules.r_proj,
                self.config,
                self.hidden_size,
                self.key_dim,
            )
            
            self.w_proj = build_module(
                submodules.w_proj,
                self.config,
                self.hidden_size,
                self.key_dim,
                low_rank_dim=config.rwkv6_la_gate_low_rank_dim,
            )
            
            self.k_proj = build_module(
                submodules.k_proj,
                self.config,
                self.hidden_size,
                self.key_dim,
            )
            
            self.v_proj = build_module(
                submodules.v_proj,
                self.config,
                self.hidden_size,
                self.value_dim,
            )
            
            self.g_proj = build_module(
                submodules.g_proj,
                self.config,
                self.hidden_size,
                self.value_dim,
            )
            
            self.bonus = nn.Parameter(torch.zeros(self.num_heads, self.head_qk_dim))
            
            self.core_linear_rnn = build_module(
                submodules.core_linear_rnn,
                config=self.config,
                expand_k=self.config.expand_k,
                expand_v=self.config.expand_v,
            )
            
        elif self.sequence_modeling_module == 'hgrn2':
            self.q_proj = build_module(
                submodules.q_proj,
                self.hidden_size,
                self.forget_dim,
                bias=False,
            )
            
            self.f_proj = build_module(
                submodules.f_proj,
                self.hidden_size,
                self.forget_dim,
                bias=False,
            )
            
            self.i_proj = build_module(
                submodules.i_proj,
                self.hidden_size,
                self.input_dim,
                bias=False,
            )
            
            self.core_linear_rnn = build_module(
                submodules.core_linear_rnn,
                config=self.config,
            )
        
        self.o_proj = build_module(
                submodules.o_proj,
                self.value_dim,
                self.hidden_size,
                bias=False,
            )
        self.la_gate_fn = ACT2FN[self.la_gate_fn]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb=None,
        packed_seq_params=None,
        lower_bound: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # hidden_states: n, b, (h d)
        hidden_states = rearrange(hidden_states, 'n b (h d) -> b n (h d)', h=self.num_heads)
        
        if self.sequence_modeling_module == 'rwkv6':
            batch_size, seq_len, hidden_size = hidden_states.shape
            if attention_mask is not None:
                hidden_states = hidden_states.mul_(attention_mask.unsqueeze(-1))
            shifted = self.time_shift(hidden_states)

            delta = shifted - hidden_states
            x = self.x_proj[0](hidden_states, delta).view(batch_size, seq_len, -1, self.rwkv6_la_proj_low_rank_dim)
            x = torch.einsum('b l n r, h n r-> b l n h', self.x_proj[1](x), self.x_proj[2].weight.view(hidden_size, 5, -1))

            r, w, k, v, g = x.add_(self.x_bias).unbind(-2)
            r = self.r_proj(hidden_states, r, delta)
            w = self.w_proj(hidden_states, w, delta)
            k = self.k_proj(hidden_states, k, delta)
            v = self.v_proj(hidden_states, v, delta)
            g = self.g_proj(hidden_states, g, delta)

            # dealing with left-padding
            if attention_mask is not None:
                v = v.mul_(attention_mask.unsqueeze(-1))
            
            r, w, k, v = map(lambda x: rearrange(x, 'b n (h d) -> b h n d', h=self.num_heads), (r, w, k, v))
            w = -torch.exp(w)
            u = self.bonus
            scale = 1.0
            
            o = self.core_linear_rnn(
                r,
                k,
                v,
                w,
                u,
                scale,
            )
            # expect computation in [n b (h d)]
            o = o * self.la_gate_fn(rearrange(g, 'b n (h d) -> n b (h d)', h=self.num_heads))
        
        elif self.sequence_modeling_module == 'hgrn2':
            q = self.q_proj(hidden_states)
            f = self.f_proj(hidden_states)
            i = self.i_proj(hidden_states)
            
            # dealing with left-padding
            if attention_mask is not None:
                i = i.mul_(attention_mask.unsqueeze(-1))

            q = swish(q)

            # improve precision
            f = f.float()

            # the lower bound for the first layer is zero
            if lower_bound is None:
                k, g = 1 - f.sigmoid(), F.logsigmoid(f)
            else:
                g = lower_bound + (1 - lower_bound) * f.sigmoid()
                k, g = 1 - g, g.log()

            q, k, i, g = map(lambda x: rearrange(x, 'b n (h d) -> b h n d', h=self.num_heads), (q, k.to(i), i, g))

            o = self.core_linear_rnn(
                q,
                k,
                i,
                g,
            )
        
        o = self.o_proj(o)
        return o
