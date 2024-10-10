from dataclasses import dataclass

import torch
from typing import Optional
from einops import rearrange
from megatron.core.transformer.module import MegatronModule
from linear_moe.model.common_modules.feature_map import RebasedFeatureMap
from fla.ops.linear_attn import chunk_linear_attn, fused_chunk_linear_attn
from fla.ops.rebased import parallel_rebased


class Rebased(MegatronModule):

    def __init__(
        self, 
        config,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        use_gamma: Optional[bool] = True,
        use_beta: Optional[bool] = True,
        normalize: Optional[bool] = True,
        eps: float = 1e-5,
    ):
        super().__init__(config)
        
        self.lsm_mode = config.lsm_mode
        self.hidden_size = config.hidden_size
        self.key_dim = int(config.hidden_size * expand_k)
        self.value_dim = int(config.hidden_size * expand_v)
        self.num_heads = config.num_attention_heads
        # num_kv_heads here mains num_query_groups
        self.num_kv_heads = config.num_query_groups if config.num_query_groups is not None else config.num_attention_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_qk_dim = self.key_dim // self.num_heads
        self.head_v_dim = self.value_dim // self.num_heads
        self.la_eps = eps
        self.la_feature_map_fn = RebasedFeatureMap(self.head_qk_dim, use_gamma, use_beta, normalize)
        

        assert self.lsm_mode in ['chunk', 'fused_chunk', 'parallel'], f"Not supported mode `{self.lsm_mode}`."
        assert self.key_dim % self.num_heads == 0, f"key dim must be divisible by num_heads of {self.num_heads}"
        assert self.value_dim % self.num_heads == 0, f"value dim must be divisible by num_heads of {self.num_heads}"
        
        if self.lsm_mode == 'chunk':
            self._la_impl = chunk_linear_attn
        elif self.lsm_mode == 'fused_chunk':
            self._la_impl = fused_chunk_linear_attn
        elif self.lsm_mode == 'parallel':
            self._la_impl = parallel_rebased
        
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
    ) -> torch.Tensor:
        # torch.Size([128, 4, 16, 32])
        q, k, v = (rearrange(x, 'n b h d -> b h n d') for x in (q, k, v))

        q, k = self.la_feature_map_fn(q, flatten=(self.lsm_mode != 'parallel')), self.la_feature_map_fn(k, flatten=(self.lsm_mode != 'parallel'))

        # expects q: B, H, T, K
        if self.lsm_mode in ['chunk', 'fused_chunk']:
            output, _ = self._la_impl(q, k, v, normalize=True, scale=1)
        elif self.lsm_mode == 'parallel':
            assert q.shape[-1] <= 128
            output, _ = self._la_impl(q, k, v, self.la_eps, True, True)

        output = rearrange(output, 'b h n d -> n b (h d)')
        return output
