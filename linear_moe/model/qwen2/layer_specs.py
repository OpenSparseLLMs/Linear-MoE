import torch.nn as nn
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear

from megatron.core.transformer.custom_layers.transformer_engine import (
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TENorm,
    TERowParallelLinear,
    TEColumnParallelLinear,
)

from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec

from linear_moe.sequence_modeling.attention import DotProductAttention
from linear_moe.sequence_modeling.retention import Retention
from linear_moe.sequence_modeling.based import Based
from linear_moe.sequence_modeling.rebased import Rebased
from linear_moe.sequence_modeling.basic_linear_attention import BasicLinearAttention
from linear_moe.sequence_modeling.gla import GLA
from linear_moe.sequence_modeling.gla import GLAGate
from linear_moe.sequence_modeling.deltanet import DeltaNet
from linear_moe.sequence_modeling.gated_deltanet import GatedDeltaNet
from linear_moe.sequence_modeling.lightning_attention import LightningAttention
from linear_moe.sequence_modeling.lasp2 import LASP2
from linear_moe.sequence_modeling.rwkv6 import DDLerpLinear
from linear_moe.sequence_modeling.rwkv6 import RWKV6
from linear_moe.sequence_modeling.hgrn2 import HGRN2
from linear_moe.model.qwen2.hybrid.hybrid_transformer_block import HybridTransformerBlock, HybridTransformerBlockSubmodules
from linear_moe.sequence_modeling.ssm import MambaStack, MambaStackSubmodules
from linear_moe.sequence_modeling.mamba2.mamba_layer import MambaLayer, MambaLayerSubmodules
from linear_moe.sequence_modeling.mamba2.mamba_mixer import MambaMixer, MambaMixerSubmodules

from .transformer.mlp import MLP, MLPSubmodules
from .transformer.attention import SelfAttention, SelfAttentionSubmodules
from linear_moe.sequence_modeling.linear_attention import LinearAttention, LinearAttentionSubmodules
from linear_moe.sequence_modeling.mom_linear_attention import MomLinearAttention, MomLinearAttentionSubmodules
from linear_moe.sequence_modeling.linear_rnn import LinearRNN, LinearRNNSubmodules
from .moe.moe_layer import MoELayer
from .transformer_layer import TransformerLayer, TransformerLayerSubmodules
from .rms_norm import Qwen2RMSNorm

# Use this spec to use lower level Transformer Engine modules (required for fp8 training)
def get_gpt_layer_with_transformer_engine_spec(
    num_experts: int = None, moe_grouped_gemm: bool = False, qk_layernorm: bool = False
) -> ModuleSpec:
    mlp = _get_mlp_module_spec(
        use_te=True, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm
    )
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=TENorm if qk_layernorm else IdentityOp,
                    k_layernorm=TENorm if qk_layernorm else IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=TENorm if num_experts else IdentityOp,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        ),
    )


# Use this spec for an implementation using only modules in megatron core
def get_gpt_layer_local_spec(
    num_experts: int = None, moe_grouped_gemm: bool = False, qk_layernorm: bool = False
) -> ModuleSpec:
    mlp = _get_mlp_module_spec(
        use_te=False, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm
    )
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=Qwen2RMSNorm,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=DotProductAttention,
                    linear_proj=RowParallelLinear,
                    q_layernorm=FusedLayerNorm if qk_layernorm else IdentityOp,
                    k_layernorm=FusedLayerNorm if qk_layernorm else IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=Qwen2RMSNorm,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
            sharded_state_dict_keys_map={
                'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
                'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
            },
        ),
    )


def get_hybrid_mamba2_linear_moe_layer_local_spec(
    num_experts: int = None, moe_grouped_gemm: bool = False, qk_layernorm: bool = False
) -> ModuleSpec:
    mlp = _get_mlp_module_spec(
        use_te=False, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm
    )
    return ModuleSpec(
        module=MambaStack,
        submodules=MambaStackSubmodules(
            mamba_layer=ModuleSpec(
                module=MambaLayer,
                submodules=MambaLayerSubmodules(
                    mixer=ModuleSpec(
                        module=MambaMixer,
                        submodules=MambaMixerSubmodules(
                            in_proj=TELayerNormColumnParallelLinear,
                            out_proj=TERowParallelLinear,
                        ),
                    ),
                    mamba_bda=get_bias_dropout_add,
                ),
            ),
            # Started with spec from gpt_layer_specs.py (with MLP removed)
            # Using the TE spec because we had problems getting the non-TE spec
            # working
            attention_layer=ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    self_attention=ModuleSpec(
                        module=SelfAttention,
                        params={"attn_mask_type": AttnMaskType.causal},
                        submodules=SelfAttentionSubmodules(
                            linear_qkv=TELayerNormColumnParallelLinear,
                            core_attention=TEDotProductAttention,
                            linear_proj=TERowParallelLinear,
                        ),
                    ),
                    self_attn_bda=get_bias_dropout_add,
                ),
            ),
            # Started with spec from gpt_layer_specs.py
            # Using the TE spec because we had problems getting the non-TE spec
            # working
            # swg: uncomment this to use dense mlp_layer
            # mlp_layer=ModuleSpec(
            #     module=TransformerLayer,
            #     submodules=TransformerLayerSubmodules(
            #         mlp=ModuleSpec(
            #             module=MLP,
            #             submodules=MLPSubmodules(
            #                 linear_fc1=TELayerNormColumnParallelLinear,
            #                 linear_fc2=TERowParallelLinear,
            #             ),
            #         ),
            #         mlp_bda=get_bias_dropout_add,
            #     ),
            # ),
            # swg: uncomment this to use moe mlp_layer
            mlp_layer=ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    pre_mlp_layernorm=Qwen2RMSNorm,
                    mlp=mlp,
                    mlp_bda=get_bias_dropout_add,
                ),
            ),
        ),
    )


def get_hybrid_retention_linear_moe_layer_local_spec(
    num_experts: int = None, moe_grouped_gemm: bool = False, qk_layernorm: bool = False
) -> ModuleSpec:
    mlp = _get_mlp_module_spec(
        use_te=False, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm
    )
    return ModuleSpec(
        module=HybridTransformerBlock,
        submodules=HybridTransformerBlockSubmodules(
            linear_transformer_layer=ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    input_layernorm=Qwen2RMSNorm,
                    self_attention=ModuleSpec(
                        module=LinearAttention,
                        # params={"attn_mask_type": AttnMaskType.causal},
                        submodules=LinearAttentionSubmodules(
                            qkv_proj=ColumnParallelLinear,
                            o_gate_proj=ColumnParallelLinear,
                            core_linear_attention=Retention,
                            o_proj=RowParallelLinear,
                        ),
                    ),
                    self_attn_bda=get_bias_dropout_add,
                    pre_mlp_layernorm=Qwen2RMSNorm,
                    mlp=mlp,
                    mlp_bda=get_bias_dropout_add,
                    sharded_state_dict_keys_map={
                        'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
                        'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
                    },
                ),
            ),
            normal_transformer_layer=ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    input_layernorm=Qwen2RMSNorm,
                    self_attention=ModuleSpec(
                        module=SelfAttention,
                        params={"attn_mask_type": AttnMaskType.causal},
                        submodules=SelfAttentionSubmodules(
                            linear_qkv=ColumnParallelLinear,
                            core_attention=DotProductAttention,
                            linear_proj=RowParallelLinear,
                            q_layernorm=FusedLayerNorm if qk_layernorm else IdentityOp,
                            k_layernorm=FusedLayerNorm if qk_layernorm else IdentityOp,
                        ),
                    ),
                    self_attn_bda=get_bias_dropout_add,
                    pre_mlp_layernorm=Qwen2RMSNorm,
                    mlp=mlp,
                    mlp_bda=get_bias_dropout_add,
                    sharded_state_dict_keys_map={
                        'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
                        'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
                    },
                ),
            ),
        ),
    )


# Use this spec for an implementation using only modules in megatron core
def get_hybrid_based_linear_moe_layer_local_spec(
    num_experts: int = None, moe_grouped_gemm: bool = False, qk_layernorm: bool = False
) -> ModuleSpec:
    mlp = _get_mlp_module_spec(
        use_te=False, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm
    )
    return ModuleSpec(
        module=HybridTransformerBlock,
        submodules=HybridTransformerBlockSubmodules(
            linear_transformer_layer=ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    input_layernorm=Qwen2RMSNorm,
                    self_attention=ModuleSpec(
                        module=LinearAttention,
                        # params={"attn_mask_type": AttnMaskType.causal},
                        submodules=LinearAttentionSubmodules(
                            qkv_proj=ColumnParallelLinear,
                            o_gate_proj=ColumnParallelLinear,
                            core_linear_attention=Based,
                            o_proj=RowParallelLinear,
                        ),
                    ),
                    self_attn_bda=get_bias_dropout_add,
                    pre_mlp_layernorm=Qwen2RMSNorm,
                    mlp=mlp,
                    mlp_bda=get_bias_dropout_add,
                    sharded_state_dict_keys_map={
                        'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
                        'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
                    },
                ),
            ),
            normal_transformer_layer=ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    input_layernorm=Qwen2RMSNorm,
                    self_attention=ModuleSpec(
                        module=SelfAttention,
                        params={"attn_mask_type": AttnMaskType.causal},
                        submodules=SelfAttentionSubmodules(
                            linear_qkv=ColumnParallelLinear,
                            core_attention=DotProductAttention,
                            linear_proj=RowParallelLinear,
                            q_layernorm=FusedLayerNorm if qk_layernorm else IdentityOp,
                            k_layernorm=FusedLayerNorm if qk_layernorm else IdentityOp,
                        ),
                    ),
                    self_attn_bda=get_bias_dropout_add,
                    pre_mlp_layernorm=Qwen2RMSNorm,
                    mlp=mlp,
                    mlp_bda=get_bias_dropout_add,
                    sharded_state_dict_keys_map={
                        'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
                        'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
                    },
                ),
            ),
        ),
    )


# Use this spec for an implementation using only modules in megatron core
def get_hybrid_rebased_linear_moe_layer_local_spec(
    num_experts: int = None, moe_grouped_gemm: bool = False, qk_layernorm: bool = False
) -> ModuleSpec:
    mlp = _get_mlp_module_spec(
        use_te=False, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm
    )
    return ModuleSpec(
        module=HybridTransformerBlock,
        submodules=HybridTransformerBlockSubmodules(
            linear_transformer_layer=ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    input_layernorm=Qwen2RMSNorm,
                    self_attention=ModuleSpec(
                        module=LinearAttention,
                        # params={"attn_mask_type": AttnMaskType.causal},
                        submodules=LinearAttentionSubmodules(
                            qkv_proj=ColumnParallelLinear,
                            o_gate_proj=ColumnParallelLinear,
                            core_linear_attention=Rebased,
                            o_proj=RowParallelLinear,
                        ),
                    ),
                    self_attn_bda=get_bias_dropout_add,
                    pre_mlp_layernorm=Qwen2RMSNorm,
                    mlp=mlp,
                    mlp_bda=get_bias_dropout_add,
                    sharded_state_dict_keys_map={
                        'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
                        'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
                    },
                ),
            ),
            normal_transformer_layer=ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    input_layernorm=Qwen2RMSNorm,
                    self_attention=ModuleSpec(
                        module=SelfAttention,
                        params={"attn_mask_type": AttnMaskType.causal},
                        submodules=SelfAttentionSubmodules(
                            linear_qkv=ColumnParallelLinear,
                            core_attention=DotProductAttention,
                            linear_proj=RowParallelLinear,
                            q_layernorm=FusedLayerNorm if qk_layernorm else IdentityOp,
                            k_layernorm=FusedLayerNorm if qk_layernorm else IdentityOp,
                        ),
                    ),
                    self_attn_bda=get_bias_dropout_add,
                    pre_mlp_layernorm=Qwen2RMSNorm,
                    mlp=mlp,
                    mlp_bda=get_bias_dropout_add,
                    sharded_state_dict_keys_map={
                        'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
                        'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
                    },
                ),
            ),
        ),
    )


# Use this spec for an implementation using only modules in megatron core
def get_hybrid_gla_linear_moe_layer_local_spec(
    num_experts: int = None, moe_grouped_gemm: bool = False, qk_layernorm: bool = False
) -> ModuleSpec:
    mlp = _get_mlp_module_spec(
        use_te=False, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm
    )
    return ModuleSpec(
        module=HybridTransformerBlock,
        submodules=HybridTransformerBlockSubmodules(
            linear_transformer_layer=ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    input_layernorm=Qwen2RMSNorm,
                    self_attention=ModuleSpec(
                        module=LinearAttention,
                        # params={"attn_mask_type": AttnMaskType.causal},
                        submodules=LinearAttentionSubmodules(
                            qkv_proj=ColumnParallelLinear,
                            o_gate_proj=ColumnParallelLinear,
                            gk_proj=GLAGate,
                            core_linear_attention=GLA,
                            o_proj=RowParallelLinear,
                        ),
                    ),
                    self_attn_bda=get_bias_dropout_add,
                    pre_mlp_layernorm=Qwen2RMSNorm,
                    mlp=mlp,
                    mlp_bda=get_bias_dropout_add,
                    sharded_state_dict_keys_map={
                        'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
                        'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
                    },
                ),
            ),
            normal_transformer_layer=ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    input_layernorm=Qwen2RMSNorm,
                    self_attention=ModuleSpec(
                        module=SelfAttention,
                        params={"attn_mask_type": AttnMaskType.causal},
                        submodules=SelfAttentionSubmodules(
                            linear_qkv=ColumnParallelLinear,
                            core_attention=DotProductAttention,
                            linear_proj=RowParallelLinear,
                            q_layernorm=FusedLayerNorm if qk_layernorm else IdentityOp,
                            k_layernorm=FusedLayerNorm if qk_layernorm else IdentityOp,
                        ),
                    ),
                    self_attn_bda=get_bias_dropout_add,
                    pre_mlp_layernorm=Qwen2RMSNorm,
                    mlp=mlp,
                    mlp_bda=get_bias_dropout_add,
                    sharded_state_dict_keys_map={
                        'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
                        'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
                    },
                ),
            ),
        ),
    )

# Use this spec for an implementation using only modules in megatron core
def get_hybrid_mom_gla_linear_moe_layer_local_spec(
    num_experts: int = None, moe_grouped_gemm: bool = False, qk_layernorm: bool = False
) -> ModuleSpec:
    mlp = _get_mlp_module_spec(
        use_te=False, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm
    )
    return ModuleSpec(
        module=HybridTransformerBlock,
        submodules=HybridTransformerBlockSubmodules(
            linear_transformer_layer=ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    input_layernorm=Qwen2RMSNorm,
                    self_attention=ModuleSpec(
                        module=MomLinearAttention,
                        # params={"attn_mask_type": AttnMaskType.causal},
                        submodules=MomLinearAttentionSubmodules(
                            qkv_proj=ColumnParallelLinear,
                            o_gate_proj=ColumnParallelLinear,
                            gk_proj=GLAGate,
                            core_linear_attention=GLA,
                            o_proj=RowParallelLinear,
                        ),
                    ),
                    self_attn_bda=get_bias_dropout_add,
                    pre_mlp_layernorm=Qwen2RMSNorm,
                    mlp=mlp,
                    mlp_bda=get_bias_dropout_add,
                    sharded_state_dict_keys_map={
                        'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
                        'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
                    },
                ),
            ),
            normal_transformer_layer=ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    input_layernorm=Qwen2RMSNorm,
                    self_attention=ModuleSpec(
                        module=SelfAttention,
                        params={"attn_mask_type": AttnMaskType.causal},
                        submodules=SelfAttentionSubmodules(
                            linear_qkv=ColumnParallelLinear,
                            core_attention=DotProductAttention,
                            linear_proj=RowParallelLinear,
                            q_layernorm=FusedLayerNorm if qk_layernorm else IdentityOp,
                            k_layernorm=FusedLayerNorm if qk_layernorm else IdentityOp,
                        ),
                    ),
                    self_attn_bda=get_bias_dropout_add,
                    pre_mlp_layernorm=Qwen2RMSNorm,
                    mlp=mlp,
                    mlp_bda=get_bias_dropout_add,
                    sharded_state_dict_keys_map={
                        'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
                        'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
                    },
                ),
            ),
        ),
    )


# Use this spec for an implementation using only modules in megatron core
def get_hybrid_deltanet_linear_moe_layer_local_spec(
    num_experts: int = None, moe_grouped_gemm: bool = False, qk_layernorm: bool = False
) -> ModuleSpec:
    mlp = _get_mlp_module_spec(
        use_te=False, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm
    )
    return ModuleSpec(
        module=HybridTransformerBlock,
        submodules=HybridTransformerBlockSubmodules(
            linear_transformer_layer=ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    input_layernorm=Qwen2RMSNorm,
                    self_attention=ModuleSpec(
                        module=LinearAttention,
                        # params={"attn_mask_type": AttnMaskType.causal},
                        submodules=LinearAttentionSubmodules(
                            qkv_proj=ColumnParallelLinear,
                            o_gate_proj=ColumnParallelLinear,
                            core_linear_attention=DeltaNet,
                            o_proj=RowParallelLinear,
                        ),
                    ),
                    self_attn_bda=get_bias_dropout_add,
                    pre_mlp_layernorm=Qwen2RMSNorm,
                    mlp=mlp,
                    mlp_bda=get_bias_dropout_add,
                    sharded_state_dict_keys_map={
                        'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
                        'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
                    },
                ),
            ),
            normal_transformer_layer=ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    input_layernorm=Qwen2RMSNorm,
                    self_attention=ModuleSpec(
                        module=SelfAttention,
                        params={"attn_mask_type": AttnMaskType.causal},
                        submodules=SelfAttentionSubmodules(
                            linear_qkv=ColumnParallelLinear,
                            core_attention=DotProductAttention,
                            linear_proj=RowParallelLinear,
                            q_layernorm=FusedLayerNorm if qk_layernorm else IdentityOp,
                            k_layernorm=FusedLayerNorm if qk_layernorm else IdentityOp,
                        ),
                    ),
                    self_attn_bda=get_bias_dropout_add,
                    pre_mlp_layernorm=Qwen2RMSNorm,
                    mlp=mlp,
                    mlp_bda=get_bias_dropout_add,
                    sharded_state_dict_keys_map={
                        'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
                        'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
                    },
                ),
            ),
        ),
    )


# Use this spec for an implementation using only modules in megatron core
def get_hybrid_gated_deltanet_linear_moe_layer_local_spec(
    num_experts: int = None, moe_grouped_gemm: bool = False, qk_layernorm: bool = False
) -> ModuleSpec:
    mlp = _get_mlp_module_spec(
        use_te=False, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm
    )
    return ModuleSpec(
        module=HybridTransformerBlock,
        submodules=HybridTransformerBlockSubmodules(
            linear_transformer_layer=ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    input_layernorm=Qwen2RMSNorm,
                    self_attention=ModuleSpec(
                        module=LinearAttention,
                        # params={"attn_mask_type": AttnMaskType.causal},
                        submodules=LinearAttentionSubmodules(
                            qkv_proj=ColumnParallelLinear,
                            o_gate_proj=ColumnParallelLinear,
                            core_linear_attention=GatedDeltaNet,
                            o_proj=RowParallelLinear,
                        ),
                    ),
                    self_attn_bda=get_bias_dropout_add,
                    pre_mlp_layernorm=Qwen2RMSNorm,
                    mlp=mlp,
                    mlp_bda=get_bias_dropout_add,
                    sharded_state_dict_keys_map={
                        'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
                        'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
                    },
                ),
            ),
            normal_transformer_layer=ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    input_layernorm=Qwen2RMSNorm,
                    self_attention=ModuleSpec(
                        module=SelfAttention,
                        params={"attn_mask_type": AttnMaskType.causal},
                        submodules=SelfAttentionSubmodules(
                            linear_qkv=ColumnParallelLinear,
                            core_attention=DotProductAttention,
                            linear_proj=RowParallelLinear,
                            q_layernorm=FusedLayerNorm if qk_layernorm else IdentityOp,
                            k_layernorm=FusedLayerNorm if qk_layernorm else IdentityOp,
                        ),
                    ),
                    self_attn_bda=get_bias_dropout_add,
                    pre_mlp_layernorm=Qwen2RMSNorm,
                    mlp=mlp,
                    mlp_bda=get_bias_dropout_add,
                    sharded_state_dict_keys_map={
                        'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
                        'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
                    },
                ),
            ),
        ),
    )


# Use this spec for an implementation using only modules in megatron core
def get_hybrid_mom_gated_deltanet_linear_moe_layer_local_spec(
    num_experts: int = None, moe_grouped_gemm: bool = False, qk_layernorm: bool = False
) -> ModuleSpec:
    mlp = _get_mlp_module_spec(
        use_te=False, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm
    )
    return ModuleSpec(
        module=HybridTransformerBlock,
        submodules=HybridTransformerBlockSubmodules(
            linear_transformer_layer=ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    input_layernorm=Qwen2RMSNorm,
                    self_attention=ModuleSpec(
                        module=MomLinearAttention,
                        # params={"attn_mask_type": AttnMaskType.causal},
                        submodules=MomLinearAttentionSubmodules(
                            qkv_proj=ColumnParallelLinear,
                            o_gate_proj=ColumnParallelLinear,
                            core_linear_attention=GatedDeltaNet,
                            o_proj=RowParallelLinear,
                        ),
                    ),
                    self_attn_bda=get_bias_dropout_add,
                    pre_mlp_layernorm=Qwen2RMSNorm,
                    mlp=mlp,
                    mlp_bda=get_bias_dropout_add,
                    sharded_state_dict_keys_map={
                        'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
                        'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
                    },
                ),
            ),
            normal_transformer_layer=ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    input_layernorm=Qwen2RMSNorm,
                    self_attention=ModuleSpec(
                        module=SelfAttention,
                        params={"attn_mask_type": AttnMaskType.causal},
                        submodules=SelfAttentionSubmodules(
                            linear_qkv=ColumnParallelLinear,
                            core_attention=DotProductAttention,
                            linear_proj=RowParallelLinear,
                            q_layernorm=FusedLayerNorm if qk_layernorm else IdentityOp,
                            k_layernorm=FusedLayerNorm if qk_layernorm else IdentityOp,
                        ),
                    ),
                    self_attn_bda=get_bias_dropout_add,
                    pre_mlp_layernorm=Qwen2RMSNorm,
                    mlp=mlp,
                    mlp_bda=get_bias_dropout_add,
                    sharded_state_dict_keys_map={
                        'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
                        'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
                    },
                ),
            ),
        ),
    )


# Use this spec for an implementation using only modules in megatron core
def get_hybrid_lightning_attention_linear_moe_layer_local_spec(
    num_experts: int = None, moe_grouped_gemm: bool = False, qk_layernorm: bool = False
) -> ModuleSpec:
    mlp = _get_mlp_module_spec(
        use_te=False, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm
    )
    return ModuleSpec(
        module=HybridTransformerBlock,
        submodules=HybridTransformerBlockSubmodules(
            linear_transformer_layer=ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    input_layernorm=Qwen2RMSNorm,
                    self_attention=ModuleSpec(
                        module=LinearAttention,
                        # params={"attn_mask_type": AttnMaskType.causal},
                        submodules=LinearAttentionSubmodules(
                            qkv_proj=ColumnParallelLinear,
                            o_gate_proj=ColumnParallelLinear,
                            core_linear_attention=LightningAttention,
                            o_proj=RowParallelLinear,
                        ),
                    ),
                    self_attn_bda=get_bias_dropout_add,
                    pre_mlp_layernorm=Qwen2RMSNorm,
                    mlp=mlp,
                    mlp_bda=get_bias_dropout_add,
                    sharded_state_dict_keys_map={
                        'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
                        'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
                    },
                ),
            ),
            normal_transformer_layer=ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    input_layernorm=Qwen2RMSNorm,
                    self_attention=ModuleSpec(
                        module=SelfAttention,
                        params={"attn_mask_type": AttnMaskType.causal},
                        submodules=SelfAttentionSubmodules(
                            linear_qkv=ColumnParallelLinear,
                            core_attention=DotProductAttention,
                            linear_proj=RowParallelLinear,
                            q_layernorm=FusedLayerNorm if qk_layernorm else IdentityOp,
                            k_layernorm=FusedLayerNorm if qk_layernorm else IdentityOp,
                        ),
                    ),
                    self_attn_bda=get_bias_dropout_add,
                    pre_mlp_layernorm=Qwen2RMSNorm,
                    mlp=mlp,
                    mlp_bda=get_bias_dropout_add,
                    sharded_state_dict_keys_map={
                        'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
                        'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
                    },
                ),
            ),
        ),
    )


# Use this spec for an implementation using only modules in megatron core
def get_hybrid_lasp2_linear_moe_layer_local_spec(
    num_experts: int = None, moe_grouped_gemm: bool = False, qk_layernorm: bool = False
) -> ModuleSpec:
    mlp = _get_mlp_module_spec(
        use_te=False, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm
    )
    return ModuleSpec(
        module=HybridTransformerBlock,
        submodules=HybridTransformerBlockSubmodules(
            linear_transformer_layer=ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    input_layernorm=Qwen2RMSNorm,
                    self_attention=ModuleSpec(
                        module=LinearAttention,
                        # params={"attn_mask_type": AttnMaskType.causal},
                        submodules=LinearAttentionSubmodules(
                            qkv_proj=ColumnParallelLinear,
                            o_gate_proj=ColumnParallelLinear,
                            core_linear_attention=LASP2,
                            o_proj=RowParallelLinear,
                        ),
                    ),
                    self_attn_bda=get_bias_dropout_add,
                    pre_mlp_layernorm=Qwen2RMSNorm,
                    mlp=mlp,
                    mlp_bda=get_bias_dropout_add,
                    sharded_state_dict_keys_map={
                        'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
                        'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
                    },
                ),
            ),
            normal_transformer_layer=ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    input_layernorm=Qwen2RMSNorm,
                    self_attention=ModuleSpec(
                        module=SelfAttention,
                        params={"attn_mask_type": AttnMaskType.causal},
                        submodules=SelfAttentionSubmodules(
                            linear_qkv=ColumnParallelLinear,
                            core_attention=DotProductAttention,
                            linear_proj=RowParallelLinear,
                            q_layernorm=FusedLayerNorm if qk_layernorm else IdentityOp,
                            k_layernorm=FusedLayerNorm if qk_layernorm else IdentityOp,
                        ),
                    ),
                    self_attn_bda=get_bias_dropout_add,
                    pre_mlp_layernorm=Qwen2RMSNorm,
                    mlp=mlp,
                    mlp_bda=get_bias_dropout_add,
                    sharded_state_dict_keys_map={
                        'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
                        'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
                    },
                ),
            ),
        ),
    )


# Use this spec for an implementation using only modules in megatron core
def get_hybrid_basic_linear_attention_linear_moe_layer_local_spec(
    num_experts: int = None, moe_grouped_gemm: bool = False, qk_layernorm: bool = False
) -> ModuleSpec:
    mlp = _get_mlp_module_spec(
        use_te=False, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm
    )
    return ModuleSpec(
        module=HybridTransformerBlock,
        submodules=HybridTransformerBlockSubmodules(
            linear_transformer_layer=ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    input_layernorm=Qwen2RMSNorm,
                    self_attention=ModuleSpec(
                        module=LinearAttention,
                        # params={"attn_mask_type": AttnMaskType.causal},
                        submodules=LinearAttentionSubmodules(
                            qkv_proj=ColumnParallelLinear,
                            o_gate_proj=ColumnParallelLinear,
                            core_linear_attention=BasicLinearAttention,
                            o_proj=RowParallelLinear,
                        ),
                    ),
                    self_attn_bda=get_bias_dropout_add,
                    pre_mlp_layernorm=Qwen2RMSNorm,
                    mlp=mlp,
                    mlp_bda=get_bias_dropout_add,
                    sharded_state_dict_keys_map={
                        'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
                        'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
                    },
                ),
            ),
            normal_transformer_layer=ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    input_layernorm=Qwen2RMSNorm,
                    self_attention=ModuleSpec(
                        module=SelfAttention,
                        params={"attn_mask_type": AttnMaskType.causal},
                        submodules=SelfAttentionSubmodules(
                            linear_qkv=ColumnParallelLinear,
                            core_attention=DotProductAttention,
                            linear_proj=RowParallelLinear,
                            q_layernorm=FusedLayerNorm if qk_layernorm else IdentityOp,
                            k_layernorm=FusedLayerNorm if qk_layernorm else IdentityOp,
                        ),
                    ),
                    self_attn_bda=get_bias_dropout_add,
                    pre_mlp_layernorm=Qwen2RMSNorm,
                    mlp=mlp,
                    mlp_bda=get_bias_dropout_add,
                    sharded_state_dict_keys_map={
                        'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
                        'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
                    },
                ),
            ),
        ),
    )


# Use this spec for an implementation using only modules in megatron core
def get_hybrid_rwkv6_linear_moe_layer_local_spec(
    num_experts: int = None, moe_grouped_gemm: bool = False, qk_layernorm: bool = False
) -> ModuleSpec:
    mlp = _get_mlp_module_spec(
        use_te=False, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm
    )
    return ModuleSpec(
        module=HybridTransformerBlock,
        submodules=HybridTransformerBlockSubmodules(
            linear_transformer_layer=ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    input_layernorm=Qwen2RMSNorm,
                    self_attention=ModuleSpec(
                        module=LinearRNN,
                        # params={"attn_mask_type": AttnMaskType.causal},
                        submodules=LinearRNNSubmodules(
                            r_proj=DDLerpLinear,
                            w_proj=DDLerpLinear,
                            k_proj=DDLerpLinear,
                            v_proj=DDLerpLinear,
                            g_proj=DDLerpLinear,
                            core_linear_rnn=RWKV6,
                            o_proj=nn.Linear,
                        ),
                    ),
                    self_attn_bda=get_bias_dropout_add,
                    pre_mlp_layernorm=Qwen2RMSNorm,
                    mlp=mlp,
                    mlp_bda=get_bias_dropout_add,
                    sharded_state_dict_keys_map={
                        'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
                        'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
                    },
                ),
            ),
            normal_transformer_layer=ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    input_layernorm=Qwen2RMSNorm,
                    self_attention=ModuleSpec(
                        module=SelfAttention,
                        params={"attn_mask_type": AttnMaskType.causal},
                        submodules=SelfAttentionSubmodules(
                            linear_qkv=ColumnParallelLinear,
                            core_attention=DotProductAttention,
                            linear_proj=RowParallelLinear,
                            q_layernorm=FusedLayerNorm if qk_layernorm else IdentityOp,
                            k_layernorm=FusedLayerNorm if qk_layernorm else IdentityOp,
                        ),
                    ),
                    self_attn_bda=get_bias_dropout_add,
                    pre_mlp_layernorm=Qwen2RMSNorm,
                    mlp=mlp,
                    mlp_bda=get_bias_dropout_add,
                    sharded_state_dict_keys_map={
                        'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
                        'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
                    },
                ),
            ),
        ),
    )


# Use this spec for an implementation using only modules in megatron core
def get_hybrid_hgrn2_linear_moe_layer_local_spec(
    num_experts: int = None, moe_grouped_gemm: bool = False, qk_layernorm: bool = False
) -> ModuleSpec:
    mlp = _get_mlp_module_spec(
        use_te=False, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm
    )
    return ModuleSpec(
        module=HybridTransformerBlock,
        submodules=HybridTransformerBlockSubmodules(
            linear_transformer_layer=ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    input_layernorm=Qwen2RMSNorm,
                    self_attention=ModuleSpec(
                        module=LinearRNN,
                        # params={"attn_mask_type": AttnMaskType.causal},
                        submodules=LinearRNNSubmodules(
                            q_proj=nn.Linear,
                            f_proj=nn.Linear,
                            i_proj=nn.Linear,
                            core_linear_rnn=HGRN2,
                            o_proj=nn.Linear,
                        ),
                    ),
                    self_attn_bda=get_bias_dropout_add,
                    pre_mlp_layernorm=Qwen2RMSNorm,
                    mlp=mlp,
                    mlp_bda=get_bias_dropout_add,
                    sharded_state_dict_keys_map={
                        'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
                        'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
                    },
                ),
            ),
            normal_transformer_layer=ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    input_layernorm=Qwen2RMSNorm,
                    self_attention=ModuleSpec(
                        module=SelfAttention,
                        params={"attn_mask_type": AttnMaskType.causal},
                        submodules=SelfAttentionSubmodules(
                            linear_qkv=ColumnParallelLinear,
                            core_attention=DotProductAttention,
                            linear_proj=RowParallelLinear,
                            q_layernorm=FusedLayerNorm if qk_layernorm else IdentityOp,
                            k_layernorm=FusedLayerNorm if qk_layernorm else IdentityOp,
                        ),
                    ),
                    self_attn_bda=get_bias_dropout_add,
                    pre_mlp_layernorm=Qwen2RMSNorm,
                    mlp=mlp,
                    mlp_bda=get_bias_dropout_add,
                    sharded_state_dict_keys_map={
                        'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
                        'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
                    },
                ),
            ),
        ),
    )


# Helper function to get module spec for MLP/MoE
def _get_mlp_module_spec(
    use_te: bool = True, num_experts: int = None, moe_grouped_gemm: bool = False
) -> ModuleSpec:
    if num_experts is None:
        # Dense MLP w/ or w/o TE modules.
        return ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=TELayerNormColumnParallelLinear if use_te else ColumnParallelLinear,
                linear_fc2=TERowParallelLinear if use_te else RowParallelLinear,
            ),
        )
    else:
        # Mixture of experts with modules in megatron core.
        return ModuleSpec(
            module=MoELayer,
            submodules=MLPSubmodules(linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear,)
            if not moe_grouped_gemm
            else None,
        )
