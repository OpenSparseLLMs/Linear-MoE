from dataclasses import dataclass

import torch
from typing import Optional
from einops import rearrange
import torch.nn.functional as F
from megatron.core.transformer.module import MegatronModule
from linear_moe.model.common_modules.activations import ACT2FN

from linear_moe.model.common_modules import RMSNorm, l2_norm_fn
from fla.ops.delta_rule import (chunk_delta_rule, fused_chunk_delta_rule,
                                fused_recurrent_delta_rule)

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

class DeltaNet(MegatronModule):

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
        
        self.lsm_mode = config.lsm_mode
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

        assert self.lsm_mode in ['chunk', 'fused_chunk', 'fused_recurrent'], f"Not supported mode `{self.lsm_mode}`."
        assert self.key_dim % self.num_heads == 0, f"key dim must be divisible by num_heads of {self.num_heads}"
        assert self.value_dim % self.num_heads == 0, f"value dim must be divisible by num_heads of {self.num_heads}"
        
        self.head_qk_dim = self.key_dim // self.num_heads
        self.head_v_dim = self.value_dim // self.num_heads

        if config.lsm_output_norm == 'rmsnorm':
            self.lsm_output_norm = RMSNorm(hidden_size=self.head_v_dim, elementwise_affine=config.la_elementwise_affine, eps=config.la_norm_eps)
        elif config.lsm_output_norm == 'identity':
            self.lsm_output_norm = torch.nn.Identity()
        else:
            raise NotImplementedError(f"Not supported output norm `{self.lsm_output_norm}`.")
        
        if self.lsm_mode == 'chunk':
            self._la_impl = chunk_delta_rule
        elif self.lsm_mode == 'fused_chunk':
            self._la_impl = fused_chunk_delta_rule
        elif self.lsm_mode == 'fused_recurrent':
            self._la_impl = fused_recurrent_delta_rule
        
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
    ) -> torch.Tensor:
        
        # torch.Size([128, 4, 16, 32])
        q, k, v = (rearrange(x, 'n b h d -> b h n d') for x in (q, k, v))

        if self.qk_activation != 'silu':
            if self.qk_activation == 'relu':
                q, k = q.relu(), k.relu()
            elif self.qk_activation == 'elu':
                q, k = elu_p1(q), elu_p1(k)
            elif self.qk_activation == 'identity':
                pass
            else:
                raise NotImplementedError

        if self.qk_norm is not None:
            if self.qk_norm == 'l2':
                q = l2_norm_fn(q)
                k = l2_norm_fn(k)
            elif self.qk_norm == 'sum':
                q = sum_norm(q).to(v)
                k = sum_norm(k).to(v)

        # expects q: B, H, T, K
        if self.lsm_mode == 'fused_recurrent':
            output, _ = self._la_impl(q, k, v, beta)
        else:
            assert self.chunk_size in [16, 32, 64]
            output, _ = self._la_impl(q, k, v, beta, self.chunk_size)
        
        output = self.lsm_output_norm(output)
        output = rearrange(output, 'b h n d -> n b (h d)')

        return output
