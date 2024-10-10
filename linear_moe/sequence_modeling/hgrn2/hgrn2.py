from dataclasses import dataclass

import torch
from typing import Optional
from einops import rearrange
import torch.nn.functional as F
from megatron.core.transformer.module import MegatronModule
from transformers.activations import ACT2FN

from linear_moe.model.common_modules import RMSNorm
from fla.ops.gla import chunk_gla, fused_chunk_gla, fused_recurrent_gla


class HGRN2(MegatronModule):

    def __init__(
        self, 
        config,
        expand_ratio: Optional[int] = 128,
    ):
        super().__init__(config)
        
        self.lsm_mode = config.lsm_mode
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        # num_kv_heads here mains num_query_groups
        self.num_kv_heads = config.num_query_groups if config.num_query_groups is not None else config.num_attention_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        
        if config.hidden_size is not None and config.num_attention_heads is not None:
            expand_ratio = self.hidden_size // self.num_heads
        self.expand_ratio = expand_ratio
        self.forget_dim = int(self.num_heads * self.expand_ratio)
        self.input_dim = self.hidden_size
        
        assert self.lsm_mode in ['chunk', 'fused_recurrent', 'fused_chunk'], f"Not supported mode `{self.lsm_mode}`."
        assert self.forget_dim % self.num_heads == 0, f"forget key dim must be divisible by num_heads of {self.num_heads}"
        assert self.input_dim % self.num_heads == 0, f"input value dim must be divisible by num_heads of {self.num_heads}"
        
        if config.lsm_output_norm == 'rmsnorm':
            self.lsm_output_norm = RMSNorm(hidden_size=self.head_dim, elementwise_affine=config.la_elementwise_affine, eps=config.la_norm_eps)
        elif config.lsm_output_norm == 'identity':
            self.lsm_output_norm = torch.nn.Identity()
        else:
            raise NotImplementedError(f"Not supported output norm `{self.lsm_output_norm}`.")
        
        if self.lsm_mode == 'chunk':
            self._la_impl = chunk_gla
        elif self.lsm_mode == 'fused_chunk':
            self._la_impl = fused_chunk_gla
        elif self.lsm_mode == 'fused_recurrent':
            self._la_impl = fused_recurrent_gla
        
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
        i: torch.Tensor,
        g: torch.Tensor,
    ) -> torch.Tensor:
        
        # expects q: b, h, n, d
        output, _ = self._la_impl(q, k, i, g)
        # import pdb; pdb.set_trace()
        output = self.lsm_output_norm(output)
        output = rearrange(output, 'b h n d -> n b (h d)')

        return output
