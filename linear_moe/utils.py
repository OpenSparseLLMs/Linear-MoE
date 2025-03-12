# -*- coding: utf-8 -*-

import functools

import torch


def contiguous(fn):
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        return fn(ctx,
                  *(i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args),
                  **{k: (v if not isinstance(v, torch.Tensor) else v.contiguous()) for k, v in kwargs.items()})
    return wrapper


def checkpoint(func):
    def wrapper(*args, **kwargs):
        return torch.utils.checkpoint.checkpoint(func, *args, **kwargs)
    return wrapper

def compute_weight_and_optimizer_memory(args, verbose=False):
    # Attention projection size.
    query_projection_size = args.kv_channels * args.num_attention_heads
    query_projection_to_hidden_size_ratio = query_projection_size / args.hidden_size
    # Group Query Attention.
    if not args.group_query_attention:
        args.num_query_groups = args.num_attention_heads
    # MoE.
    num_experts = 1 if args.num_experts is None else args.num_experts
    gated_linear_multiplier = 3 / 2 if args.swiglu else 1

    # Fine-grained Experts + Shared Experts
    if args.moe_ffn_hidden_size is not None:
        intermediate_hidden_size = args.moe_ffn_hidden_size
    else:
        intermediate_hidden_size = args.ffn_hidden_size
    
    shared_moe_ffn_hidden_size = args.shared_moe_ffn_hidden_size if args.shared_moe_ffn_hidden_size else 0

    num_parameters_in_transformer_layers = (
        2
        * args.num_layers
        * args.hidden_size
        * args.hidden_size
        * (
            # Attention.
            (
                (1 + (args.num_query_groups / args.num_attention_heads))
                * query_projection_to_hidden_size_ratio
            )
            # MLP.
            + ((intermediate_hidden_size / args.hidden_size) * num_experts * gated_linear_multiplier + (shared_moe_ffn_hidden_size / args.hidden_size) * gated_linear_multiplier)
            # Transformer layernorms.
            + (2 / args.hidden_size)
            # Final layernorm.
            + (1 / (args.num_layers * args.hidden_size))
        )
    )
    num_parameters_in_transformer_moe_act_layers = (
        2
        * args.num_layers
        * args.hidden_size
        * args.hidden_size
        * (
            # Attention.
            (
                (1 + (args.num_query_groups / args.num_attention_heads))
                * query_projection_to_hidden_size_ratio
            )
            # MLP.
            + ((intermediate_hidden_size / args.hidden_size) * args.moe_router_topk * gated_linear_multiplier + (shared_moe_ffn_hidden_size / args.hidden_size) * gated_linear_multiplier)
            # Transformer layernorms.
            + (2 / args.hidden_size)
            # Final layernorm.
            + (1 / (args.num_layers * args.hidden_size))
        )
    )

    
    embedding_size = args.hidden_size * args.padded_vocab_size
    if args.untie_embeddings_and_output_weights:
        num_parameters_in_embedding_layers = 2 * embedding_size
    else:
        num_parameters_in_embedding_layers = embedding_size
    num_act_parameters = num_parameters_in_transformer_moe_act_layers +  num_parameters_in_embedding_layers
    num_total_parameters = num_parameters_in_transformer_layers + num_parameters_in_embedding_layers
    if verbose:
        print(
            f"Number of parameters in transformer layers in billions: "
            f"{num_parameters_in_transformer_layers / 10**9: .2f}"
        )
        print(
            f"Number of parameters in embedding layers in billions: "
            f"{num_parameters_in_embedding_layers / 10**9:.2f}"
        )
        print(f"Total number of parameters in billions: {num_total_parameters / 10**9:.2f}")
        print(f"Act number of parameters in billions: {num_act_parameters / 10**9:.2f}")

    # Most loaded model shard has (1/pp_size transformer layers + 1 embedding layer) / tp_size.
    num_parameters_on_most_loaded_model_shard = (
        (num_parameters_in_transformer_layers / args.pipeline_model_parallel_size) + embedding_size
    ) / args.tensor_model_parallel_size
    if args.untie_embeddings_and_output_weights and args.pipeline_model_parallel_size == 1:
        num_parameters_on_most_loaded_model_shard += (
            embedding_size / args.tensor_model_parallel_size
        )
    if verbose:
        print(
            f"Number of parameters in most loaded shard in billions: "
            f"{num_parameters_on_most_loaded_model_shard / 10**9:.4f}"
        )

    if args.pipeline_model_parallel_size > 1:
        # Other shards just have (1/pp_size transformer layers) / tp_size.
        num_parameters_on_other_model_shards = num_parameters_in_transformer_layers / (
            args.pipeline_model_parallel_size * args.tensor_model_parallel_size
        )
        if verbose:
            print(
                f"Number of parameters in other shards in billions: "
                f"{num_parameters_on_other_model_shards / 10**9:.4f}"
            )

    num_bytes_per_parameter = (
        18 if not args.use_distributed_optimizer else 6 + (12 / args.data_parallel_size)
    )
    weight_and_optimizer_memory = (
        num_parameters_on_most_loaded_model_shard * num_bytes_per_parameter
    )

    return weight_and_optimizer_memory