from dataclasses import dataclass

import torch
from typing import Optional
from einops import rearrange
import torch.nn.functional as F
from megatron.core.transformer.module import MegatronModule
from linear_moe.model.common_modules.activations import ACT2FN

from linear_moe.model.common_modules import RMSNorm, l2_norm_fn
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

def simple_norm(x):
    return (F.normalize(x, dim=-1) * x.shape[-1] ** 0.5).to(x)


# @torch.jit.script
def elu_p1(x):
    return (F.elu(x, 1., False) + 1.).to(x)


# @torch.jit.script
def sum_norm(x):
    return (x / x.sum(-1, keepdim=True)).to(x)


# @torch.jit.script
def elu_norm(x):
    dtype = x.dtype
    x = F.elu(x, 1., False) + 1.
    return (x / x.sum(-1, keepdim=True)).to(dtype)

class GatedDeltaNet(MegatronModule):

    def __init__(
        self, 
        config,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        chunk_size: int = 64,
        qk_activation: str = 'silu',
        qk_norm: str = 'l2',
    ):
        super().__init__(config)
        
        self.la_mode = config.la_mode
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        # num_kv_heads here mains num_query_groups
        self.num_kv_heads = config.num_query_groups if config.num_query_groups is not None else config.num_attention_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.chunk_size = chunk_size

        assert self.qk_activation in ['silu', 'relu', 'elu', 'identity']
        assert self.qk_norm in ['l2', 'sum']
        self.key_dim = int(config.hidden_size * expand_k)
        self.value_dim = int(config.hidden_size * expand_v)

        assert self.la_mode in ['chunk', 'fused_chunk', 'fused_recurrent'], f"Not supported mode `{self.la_mode}`."
        assert self.key_dim % self.num_heads == 0, f"key dim must be divisible by num_heads of {self.num_heads}"
        assert self.value_dim % self.num_heads == 0, f"value dim must be divisible by num_heads of {self.num_heads}"
        
        self.head_qk_dim = self.key_dim // self.num_heads
        self.head_v_dim = self.value_dim // self.num_heads

        
        if self.la_mode == 'chunk':
            self._la_impl = chunk_gated_delta_rule
        elif self.la_mode == 'fused_recurrent':
            self._la_impl = fused_recurrent_gated_delta_rule
        else:
            raise NotImplementedError('Not supported la_mode')
        
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: torch.nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        module._is_hf_initialized = True


    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        gk: torch.Tensor,
    ) -> torch.Tensor:
        
        # torch.Size([128, 4, 16, 32])
        q, k, v, beta, gk = (x.transpose(0, 1).contiguous() for x in (q, k, v, beta, gk))

        q, k, v = torch.nn.SiLU()(q), torch.nn.SiLU()(k), torch.nn.SiLU()(v)

        if self.qk_norm is not None:
            if self.qk_norm == 'l2':
                q = l2_norm_fn(q)
                k = l2_norm_fn(k)
            elif self.qk_norm == 'sum':
                q = sum_norm(q).to(v)
                k = sum_norm(k).to(v)

        # expects q: B, H, T, K
        output, _ = self._la_impl(q, k, v, gk, beta, head_first=False)
        
        output = rearrange(output, 'b n h d -> n b (h d)').contiguous()

        return output
