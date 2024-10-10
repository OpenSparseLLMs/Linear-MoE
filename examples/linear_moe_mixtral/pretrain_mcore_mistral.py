# Copyright (c) 2023 Alibaba PAI and Nvidia Megatron-LM Team.
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

import os
import torch
from torch import Tensor
from functools import partial
from typing import Union

from megatron import get_args
from megatron import get_timers
from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
import megatron.model
from megatron.utils import (
    get_batch_on_this_tp_rank,
    get_batch_on_this_cp_rank,
    average_losses_across_data_parallel_group
)
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.training import pretrain
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import GPTDataset
# from megatron.arguments import core_transformer_config_from_args
from linear_moe.arguments import core_transformer_config_from_args

from linear_moe.data import build_pretrain_dataset_from_original
from linear_moe.data.utils import get_batch_on_this_tp_rank_original
from linear_moe.tokenizer import get_tokenizer, build_tokenizer
from linear_moe.arguments import get_patch_args
from linear_moe.model.mixtral.model import GPTModel
from linear_moe.sequence_modeling.mamba2.mamba_model import MambaModel
from linear_moe.model.mixtral.layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_layer_local_spec,
    get_retention_linear_moe_layer_local_spec,
    get_based_linear_moe_layer_local_spec,
    get_rebased_linear_moe_layer_local_spec,
    get_pure_mamba2_stack_linear_moe_layer_local_spec,
    get_hybrid_mamba2_stack_linear_moe_layer_local_spec,
    get_basic_linear_attention_linear_moe_layer_local_spec,
    get_gla_linear_moe_layer_local_spec,
    get_rwkv6_linear_moe_layer_local_spec,
    get_deltanet_linear_moe_layer_local_spec,
    get_hgrn2_linear_moe_layer_local_spec,
)
from linear_moe.model.mixtral.transformer_config import MixtralTransformerConfig

import huggingface_hub
huggingface_hub.login("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")


def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, MambaModel, megatron.model.GPTModel]:
    args = get_args()
    build_tokenizer(args)
    config = core_transformer_config_from_args(args, MixtralTransformerConfig)

    if args.sequence_modeling_type != "attention":
        if args.sequence_modeling_module == "pure_mamba2":
            mamba_stack_spec = get_pure_mamba2_stack_linear_moe_layer_local_spec(args.num_experts, args.moe_grouped_gemm)
        if args.sequence_modeling_module == "hybrid_mamba2":
            mamba_stack_spec = get_hybrid_mamba2_stack_linear_moe_layer_local_spec(args.num_experts, args.moe_grouped_gemm)
        elif args.sequence_modeling_module == "retention":
            transformer_layer_spec = get_retention_linear_moe_layer_local_spec(args.num_experts, args.moe_grouped_gemm)
        elif args.sequence_modeling_module == "based":
            transformer_layer_spec = get_based_linear_moe_layer_local_spec(args.num_experts, args.moe_grouped_gemm)
        elif args.sequence_modeling_module == "rebased":
            transformer_layer_spec = get_rebased_linear_moe_layer_local_spec(args.num_experts, args.moe_grouped_gemm)
        elif args.sequence_modeling_module == "linear_attention":
            transformer_layer_spec = get_basic_linear_attention_linear_moe_layer_local_spec(args.num_experts, args.moe_grouped_gemm)
        elif args.sequence_modeling_module == "gla":
            transformer_layer_spec = get_gla_linear_moe_layer_local_spec(args.num_experts, args.moe_grouped_gemm)
        elif args.sequence_modeling_module == "deltanet":
            transformer_layer_spec = get_deltanet_linear_moe_layer_local_spec(args.num_experts, args.moe_grouped_gemm)
        elif args.sequence_modeling_module == "rwkv6":
            transformer_layer_spec = get_rwkv6_linear_moe_layer_local_spec(args.num_experts, args.moe_grouped_gemm)
        elif args.sequence_modeling_module == "hgrn2":
            transformer_layer_spec = get_hgrn2_linear_moe_layer_local_spec(args.num_experts, args.moe_grouped_gemm)
    else:
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm)

    if args.sequence_modeling_module in ["pure_mamba2", "hybrid_mamba2"]:
        model = MambaModel(
            config=config,
            mamba_stack_spec=mamba_stack_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            hybrid_attention_ratio=args.hybrid_attention_ratio,
            hybrid_mlp_ratio=args.hybrid_mlp_ratio,
            hybrid_override_pattern=args.hybrid_override_pattern,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type
        )
    else:
        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
        )

    return model


def get_batch(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None

    args = get_args()

    if "-Raw" in args.dataset:
        # get batches based on the TP rank you are on
        batch = get_batch_on_this_tp_rank_original(data_iterator)
        # slice batch along sequence dimension for context parallelism
        batch = get_batch_on_this_cp_rank(batch)

    elif "-Idxmap" in args.dataset:
        # get batches based on the TP rank you are on
        batch = get_batch_on_this_tp_rank(data_iterator)
        # slice batch along sequence dimension for context parallelism
        batch = get_batch_on_this_cp_rank(batch)

    else:
        raise ValueError("please set correct --dataset ")

    return batch.values()

def loss_func(loss_mask: Tensor, output_tensor: Tensor):
    """Loss function.

    Args:
        loss_mask (Tensor): Used to mask out some portions of the loss
        output_tensor (Tensor): The tensor with the losses
    """
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    if args.context_parallel_size > 1:
        loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), loss_mask.sum().view(1)])
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())
        loss = loss[0] / loss[1]
    else:
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss.isnan(), (
            f'Rank {global_rank}: found NaN in local forward loss calculation. '
            f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
        )

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss * args.context_parallel_size, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model: Union[GPTModel, MambaModel]):
    """Forward step."""
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch-generator').stop()

    # temporarily close attention mask for mixtral
    args = get_args()
    if args.base_model == "mixtral":
        attention_mask = None
    output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def is_dataset_built_on_rank():
    return (mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()) and mpu.get_tensor_model_parallel_rank() == 0

def core_gpt_dataset_config_from_args(args):
    return GPTDatasetConfig(
        is_built_on_rank=is_dataset_built_on_rank,
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=args.data_path,
        split=args.split,
        path_to_cache=args.data_cache_path,
    )

def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()
    if "-Raw" in args.dataset:
                train_ds, valid_ds, test_ds = \
                                    build_pretrain_dataset_from_original(args.dataset)
    else:
        train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
            GPTDataset,
            train_val_test_num_samples,
            core_gpt_dataset_config_from_args(args)
        ).build()

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    train_valid_test_datasets_provider.is_distributed = True
    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             extra_args_provider=get_patch_args)
