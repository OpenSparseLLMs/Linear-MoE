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

from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F

from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megablocks.layers.arguments import Arguments as MegablocksArguments

from .experts import (
    GroupedMLP,
    SequentialMLP,
    MemSavingParallelMLP,
    MemSavingParallelDroplessMLP,
)
from .router import TopKRouter
from .token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
)
from ..transformer.mlp import MLPSubmodules, MLP

class BaseMoELayer(MegatronModule, ABC):
    """Base class for a mixture of experts layer.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
    """

    def __init__(self, config: TransformerConfig, layer_number: int = None):
        super(BaseMoELayer, self).__init__(config)
        self.config = config
        self.expert_parallel_size = parallel_state.get_expert_model_parallel_world_size()
        assert self.expert_parallel_size > 0, "Expected non-negative expert parallel size"

        if self.config.moe_extended_tp:
            self.num_local_experts = self.config.num_moe_experts
            local_expert_indices_offset = 0
        else:
            assert self.config.num_moe_experts % self.expert_parallel_size == 0
            self.num_local_experts = self.config.num_moe_experts // self.expert_parallel_size
            local_expert_indices_offset = (
                parallel_state.get_expert_model_parallel_rank() * self.num_local_experts
            )

        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        assert all(map(lambda x: x < self.config.num_moe_experts, self.local_expert_indices))
        self.router = None
        self.experts = None
        self.token_dispatcher = None
        self.layer_number = layer_number

    @abstractmethod
    def forward(self, hidden_states):
        pass

    def set_layer_number(self, layer_number: int):
        self.layer_number = layer_number
        self.router.set_layer_number(layer_number)


class MoELayer(BaseMoELayer):
    """Mixture of experts Layer **currently only supports no token dropping**.

    Args:
        BaseMoELayer (MegatronModule): Base class for MoE layers
    """

    def __init__(
        self, config: TransformerConfig, submodules: MLPSubmodules = None, layer_number: int = None
    ):
        self.submodules = submodules
        super(MoELayer, self).__init__(config=config, layer_number=layer_number)
        self.router = TopKRouter(config=self.config)
        self.enable_shared_experts = config.enable_shared_expert
        if config.enable_shared_expert:
            self.shared_expert = MLP(self.config, submodules, is_expert=False, is_shared_expert=True)
            self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)

        if self.config.moe_megablocks:
            if self.training:
                moe_expert_capacity_factor = self.config.moe_train_capacity_factor
            else:
                moe_expert_capacity_factor = self.config.moe_eval_capacity_factor
            mb_args = MegablocksArguments(
                # ffn settings
                hidden_size=self.config.hidden_size,
                ffn_hidden_size=self.config.moe_ffn_hidden_size,
                bias=self.config.add_bias_linear,
                return_bias=True,  # set to True for interface consistency
                activation_fn=self.config.activation_func,
                mlp_type="glu" if self.config.gated_linear_unit else "mlp",
                mlp_impl="sparse",
                # moe settings
                moe_num_experts=self.config.num_moe_experts,
                moe_top_k=self.config.moe_router_topk,
                moe_capacity_factor=moe_expert_capacity_factor,
                moe_expert_model_parallelism=self.config.expert_model_parallel_size > 1,
                expert_parallel_group=parallel_state.get_expert_model_parallel_group(),
                moe_loss_weight=0,  # set to 0 to disable aux loss calculation here
                # dtype and device
                fp16=self.config.fp16,
                bf16=self.config.bf16,
                device=torch.cuda.current_device(),
            )
            if self.config.moe_token_dropping:
                self.experts = MemSavingParallelMLP(mb_args)
            else:
                self.experts = MemSavingParallelDroplessMLP(mb_args)
        elif self.config.moe_grouped_gemm:
            self.experts = GroupedMLP(self.num_local_experts, self.config)
        else:
            assert isinstance(self.submodules, MLPSubmodules)
            self.experts = SequentialMLP(self.num_local_experts, self.config, self.submodules)
        if config.moe_token_dispatcher_type == "allgather":
            self.token_dispatcher = MoEAllGatherTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        else:
            raise ValueError(
                f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}"
            )
        self.moe_layer_recompute = config.moe_layer_recompute

    def forward(self, hidden_states: torch.Tensor):
        if (
            self.training
            and self.config.tensor_model_parallel_size > 1
            and not self.config.sequence_parallel
        ):
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )

        # process MoE
        def custom_forward(hidden_states):
            probs, indices = self.router(hidden_states)
            if self.config.moe_megablocks:
                # megablocks handles token permutation internally
                expert_output, mlp_bias = self.experts(hidden_states, probs, indices)
            else:
                (dispatched_input, tokens_per_expert) = self.token_dispatcher.token_permutation(
                    hidden_states, probs, indices
                )
                expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)
                expert_output, mlp_bias = self.token_dispatcher.token_unpermutation(expert_output, mlp_bias)
            return expert_output, mlp_bias

        if self.moe_layer_recompute:
            output, mlp_bias = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
        else:
            output, mlp_bias = custom_forward(hidden_states)

        if self.enable_shared_experts:
            shared_expert_output, shared_bias = self.shared_expert(hidden_states)
            shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states).view(-1, 1)) * shared_expert_output.view(-1, hidden_states.shape[-1])
            output = output + shared_expert_output.view(-1, hidden_states.shape[-2], hidden_states.shape[-1])

        return output, mlp_bias
