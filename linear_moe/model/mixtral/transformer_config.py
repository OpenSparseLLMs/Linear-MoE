# Copyright (c) 2024 Alibaba PAI and Nvidia Megatron-LM Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from megatron.core.transformer import TransformerConfig


@dataclass
class MixtralTransformerConfig(TransformerConfig):

    transformer_impl: str = 'transformer_engine'

    moe_ffn_hidden_size: int = None

    shared_moe_ffn_hidden_size: int = None

    enable_shared_expert: bool = False

    num_shared_experts: int = None

    moe_layer_freq: int = None

    moe_megablocks: bool = False
    """When set to True, use Megablocks for MoE layer."""

    moe_train_capacity_factor: float = None

    moe_eval_capacity_factor: float = None

    moe_token_dropping: bool = False

    rotary_base: int = None

    rotary_scaling_factor: int = None

    max_position_embeddings: int = None

    moe_aux_loss_coeff: float = 0.0

    use_la_module: bool = False

    megatron_hybrid_mamba_method: bool = False

    la_module: str = None

    la_mode: str = None

    base_model: str = None

    la_feature_map: str = None

    la_tie_feature_map_qk:  bool = False

    la_norm_q:  bool = False

    la_norm_k:  bool = False

    la_do_feature_map_norm:  bool = False

    la_output_norm:  str = None

    la_checkpointing:  bool = False

    la_elementwise_affine: bool = True

    la_norm_eps: float = 1e-5

    gla_la_gate_logit_normalizer: int = 16

    gla_la_gate_low_rank_dim: int = 16

    gla_la_clamp_min: float = None

    rwkv6_la_proj_low_rank_dim: int = 32

    rwkv6_la_gate_low_rank_dim: int = 64

    rwkv_7_la_decay_low_rank_dim: int = 64

    rwkv_7_la_gate_low_rank_dim: int = 128

    rwkv_7_a_low_rank_dim: int = 64

    la_gate_fn: str = 'swish'

    expand_k: float = 1.0

    expand_v: float = 1.0

    layer_type_list: str = None

    num_memories: int = 4

    topk: int = 2

    capacity: float = 1.0

    shared_mem: bool = True

    def __post_init__(self):
        super().__post_init__()

        if self.moe_megablocks and self.moe_grouped_gemm:
            raise ValueError("moe_megablocks and moe_grouped_gemm cannot be both True.")
