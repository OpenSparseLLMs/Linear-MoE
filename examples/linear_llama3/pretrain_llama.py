# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT."""

import os
import torch
from functools import partial
from typing import Union
from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset
from megatron.core.datasets.utils import get_blend_from_list
import megatron.legacy.model
from megatron.training import pretrain
from megatron.core.transformer.spec_utils import import_module
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    average_losses_across_data_parallel_group
)
from megatron.training.arguments import core_transformer_config_from_args

from linear_moe.data.utils import get_batch_on_this_tp_rank_original
from linear_moe.data import build_pretrain_dataset_from_original
from linear_moe.model.llama3.layer_specs import ( 
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_layer_local_spec,
    get_hybrid_retention_linear_moe_layer_local_spec,
    get_hybrid_based_linear_moe_layer_local_spec,
    get_hybrid_rebased_linear_moe_layer_local_spec,
    get_hybrid_mamba2_linear_moe_layer_local_spec,
    get_hybrid_basic_linear_attention_linear_moe_layer_local_spec,
    get_hybrid_gla_linear_moe_layer_local_spec,
    get_hybrid_deltanet_linear_moe_layer_local_spec,
    get_hybrid_gated_deltanet_linear_moe_layer_local_spec,
    get_hybrid_rwkv6_linear_moe_layer_local_spec,
    get_hybrid_hgrn2_linear_moe_layer_local_spec,
    get_hybrid_mom_gla_linear_moe_layer_local_spec,
    get_hybrid_mom_gated_deltanet_linear_moe_layer_local_spec,
)
from linear_moe.model.llama3.model import GPTModel
from linear_moe.model.llama3.hybrid.hybrid_model import HybridGPTModel
from linear_moe.sequence_modeling.mamba2.mamba_model import MambaModel
from linear_moe.model.llama3.transformer_config import Llama3TransformerConfig
from linear_moe.arguments import get_patch_args
from linear_moe.tokenizer import get_tokenizer, build_tokenizer
import torch._dynamo
torch._dynamo.config.suppress_errors = True

def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_mcore_models to True, it will return the mcore GPT model and if not the legacy GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()
    build_tokenizer(args)
    print_rank_0('building Linear-Llama3 model ...')
    # Experimental loading arguments from yaml
    config = core_transformer_config_from_args(args, Llama3TransformerConfig)
    # import pdb; pdb.set_trace()
    if args.use_la_module:
        if args.la_module == "mamba2":
            mamba_stack_spec = get_hybrid_mamba2_linear_moe_layer_local_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)
        elif args.la_module == "retention":
            hybrid_transformer_layer_spec = get_hybrid_retention_linear_moe_layer_local_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)
        elif args.la_module == "based":
            hybrid_transformer_layer_spec = get_hybrid_based_linear_moe_layer_local_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)
        elif args.la_module == "rebased":
            hybrid_transformer_layer_spec = get_hybrid_rebased_linear_moe_layer_local_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)
        elif args.la_module == "linear_attention":
            hybrid_transformer_layer_spec = get_hybrid_basic_linear_attention_linear_moe_layer_local_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)
        elif args.la_module == "gla":
            hybrid_transformer_layer_spec = get_hybrid_gla_linear_moe_layer_local_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)
        elif args.la_module == "deltanet":
            hybrid_transformer_layer_spec = get_hybrid_deltanet_linear_moe_layer_local_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)
        elif args.la_module == "gated_deltanet":
            hybrid_transformer_layer_spec = get_hybrid_gated_deltanet_linear_moe_layer_local_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)
        elif args.la_module == "rwkv6":
            hybrid_transformer_layer_spec = get_hybrid_rwkv6_linear_moe_layer_local_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)
        elif args.la_module == "hgrn2":
            hybrid_transformer_layer_spec = get_hybrid_hgrn2_linear_moe_layer_local_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)
        elif args.la_module == "mom_gla":
            hybrid_transformer_layer_spec = get_hybrid_mom_gla_linear_moe_layer_local_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)
        elif args.la_module == "mom_gated_deltanet":
            hybrid_transformer_layer_spec = get_hybrid_mom_gated_deltanet_linear_moe_layer_local_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)
    else:
        if args.transformer_impl == "transformer_engine":
            transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)
        else:
            transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)

    if args.use_la_module:
        if args.la_module in ["mamba2"]:
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
            model = HybridGPTModel(
                config=config,
                hybrid_transformer_layer_spec=hybrid_transformer_layer_spec,
                layer_type_list=args.layer_type_list,
                vocab_size=args.padded_vocab_size,
                max_sequence_length=args.max_position_embeddings,
                pre_process=pre_process,
                post_process=post_process,
                fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
                parallel_output=True,
                share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
                position_embedding_type=args.position_embedding_type,
                rotary_percent=args.rotary_percent,
                rotary_base=args.rotary_base,
                seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor
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
            rotary_base=args.rotary_base,
            seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor
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

def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
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


def forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch-generator').stop()

    output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def is_dataset_built_on_rank():
    return (mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()) and mpu.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=get_blend_from_list(args.data_path),
        blend_per_split=[
            get_blend_from_list(args.train_data_path),
            get_blend_from_list(args.valid_data_path),
            get_blend_from_list(args.test_data_path)
        ],
        split=args.split,
        path_to_cache=args.data_cache_path,
        # mock=args.mock_data,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()
    print_rank_0("> building train, validation, and test datasets for GPT ...")

    if "-Raw" in args.dataset:
        train_ds, valid_ds, test_ds = build_pretrain_dataset_from_original(args.dataset)
    else:
        config = core_gpt_dataset_config_from_args(args)

        if config.mock:
            dataset_type = MockGPTDataset
        else:
            dataset_type = GPTDataset
        train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
            dataset_type,
            train_val_test_num_samples,
            is_dataset_built_on_rank,
            config
        ).build()

        print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    train_valid_test_datasets_provider.is_distributed = True

    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             extra_args_provider=get_patch_args)
