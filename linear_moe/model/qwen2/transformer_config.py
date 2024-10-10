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
class Qwen2TransformerConfig(TransformerConfig):

    transformer_impl: str = 'transformer_engine'

    moe_ffn_hidden_size: int = None

    shared_moe_ffn_hidden_size: int = None

    enable_shared_expert: bool = False

    num_shared_experts: int = None

    moe_layer_freq: int = None

    rotary_base: int = None

    rotary_scaling_factor: int = None

    max_position_embeddings: int = None

    moe_aux_loss_coeff: float = 0.0

    sequence_modeling_type: str = None

    sequence_modeling_module: str = None

    lsm_mode: str = None

    base_model: str = None

    lsm_feature_map: str = None
    
    la_tie_feature_map_qk:  bool = False
    
    la_norm_q:  bool = False
    
    la_norm_k:  bool = False
    
    la_do_feature_map_norm:  bool = False
    
    lsm_output_norm:  str = None

    la_checkpointing:  bool = False
    
    la_elementwise_affine: bool = True
    
    la_norm_eps: float = 1e-5
    
    gla_la_gate_logit_normalizer: int = 16
    
    gla_la_gate_low_rank_dim: int = 16
    
    gla_la_clamp_min: float = None
    
    rwkv6_la_proj_low_rank_dim: int = 32
    
    rwkv6_la_gate_low_rank_dim: int = 64
    
    lsm_gate_fn: str = 'swish'
    
    expand_k: float = 1.0
    
    expand_v: float = 1.0
    
    layer_type_list: str = None
