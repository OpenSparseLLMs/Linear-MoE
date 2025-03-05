from dataclasses import dataclass
from megatron.core.transformer import TransformerConfig


@dataclass
class DeepSeekV2TransformerConfig(TransformerConfig):

    moe_ffn_hidden_size: int = None

    enable_shared_expert: bool = False

    q_lora_rank: int = None

    kv_lora_rank: int = None

    qk_nope_head_dim: int = None

    qk_rope_head_dim: int = None

    v_head_dim: int = None

    num_shared_experts: int = None

    moe_layer_freq: int = None

    rotary_base: int = None

    rotary_scaling_factor: int = None

    max_position_embeddings: int = None

    moe_aux_loss_coeff: float = 0.0

    use_la_module: bool = False

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
    
    la_gate_fn: str = 'swish'
    
    expand_k: float = 1.0
    
    expand_v: float = 1.0
    
    layer_type_list: str = None

    num_memories: int = 4

    topk: int = 2

    capacity: float = 1.0

    shared_mem: bool = True
