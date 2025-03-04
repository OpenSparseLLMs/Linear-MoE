from dataclasses import dataclass
import math
from typing import Optional, Union
from einops import rearrange
import torch
from torch.nn import functional as F
from megatron.core import parallel_state
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb
from megatron.core.utils import divide
from linear_moe.model.common_modules.activations import ACT2FN

def transform(x: torch.Tensor, routing_mask: torch.Tensor, num_experts: int, selected_experts: torch.Tensor, capacity: float):
    '''
    transform the hidden_states into chunks by experts (expert_batch, selected_len, hidden_size)
    expert_batch may be close to experts * orginal_batch

    x: (batch_size, seq_len, hidden_size)
    routing_mask: (batch_size, seq_len, num_experts)
    '''
    # 若selected_experts含topk_experts，先取top1
    if selected_experts.dim() == 3:
        topk = selected_experts.shape[2]
        x = x.repeat_interleave(topk, dim=1) 
        selected_experts = selected_experts.reshape(selected_experts.shape[0], -1)

    b, s, d = x.shape
    x_flat = x.reshape(b * s, d)  # [b*s, d]

    with torch.no_grad():
        # 创建批次索引和序列索引
        batch_indices = torch.arange(b, device=x.device).unsqueeze(-1)
        batch_indices = batch_indices.expand(b, s).reshape(-1)

        # 展平后按专家id排序
        
        experts_flat = selected_experts.reshape(-1)  # [b*s]

        combined = batch_indices * (experts_flat.max() + 1) + experts_flat
        # 对复合键进行排序
        sorted_indices = combined.argsort()

    # 根据排序后的索引重排q
    x_sorted = x_flat[sorted_indices]  # [b*s, d]

    with torch.no_grad():
        batch_expert_tokens = routing_mask.sum(dim=1)
        offset = batch_expert_tokens.cumsum(dim=1)
        expert_batch_offset = offset.transpose(0,1)
        batch_offset = torch.arange(0, b*s, s, device=offset.device)
        expert_batch_offset += batch_offset
        flatten_offset = expert_batch_offset.transpose(0, 1).reshape(-1)
        lengths = torch.concat([flatten_offset[:1], flatten_offset[1:] - flatten_offset[:-1]], dim=0)
        max_len = lengths.max()
        capacity_len = math.ceil(s / topk * capacity)
        max_len = min(max_len, capacity_len)

        indices = torch.arange(max_len, device=flatten_offset.device).unsqueeze(0).expand(b*num_experts, -1) + torch.cat([torch.tensor([0], device=flatten_offset.device), flatten_offset[:-1]], dim=0).unsqueeze(1)
        # discard tokens exceed capacity and is far from now
        # left pad
        truncation_indices = indices + batch_expert_tokens.reshape((-1,)).unsqueeze(-1) - max_len
        mask = torch.bitwise_and(truncation_indices < flatten_offset.unsqueeze(-1), truncation_indices >= 0)
        mask = torch.bitwise_and(mask, truncation_indices >= torch.cat((torch.zeros((1,), dtype=flatten_offset.dtype, device=flatten_offset.device), flatten_offset[:-1])).unsqueeze(-1))
        truncation_indices = torch.where(mask, truncation_indices, torch.zeros_like(truncation_indices))

    gathered_x = torch.gather(x_sorted, 0, truncation_indices.reshape(-1).unsqueeze(-1).expand(-1, d))
    ret_x = gathered_x.reshape(b * num_experts, -1, d)
    # with torch.no_grad():
    #     mask = mask.unsqueeze(-1)
    #     mask_x = mask.expand_as(ret_x).bitwise_not()
    #     ret_x[mask_x] = 0.0
    ret_x = ret_x * mask.unsqueeze(-1).expand_as(ret_x)
    pad_x = torch.zeros((b * num_experts, capacity_len-max_len, d), dtype=ret_x.dtype, device=ret_x.device)
    # left pad
    ret_x = torch.cat((pad_x, ret_x), dim=1).reshape((b, num_experts, capacity_len, d)).transpose(0, 1)
    # truncation_indices += capacity_len-max_len

    return ret_x.contiguous(), truncation_indices, sorted_indices, max_len, mask
    
def reconstruct(re_x, indices: torch.Tensor, sorted_indices: torch.Tensor, batch_size: int, seq_len: int, topk: int, routing_weights: torch.Tensor, mask: torch.Tensor):
    re_x = re_x.transpose(0, 1).reshape((-1, re_x.shape[2], re_x.shape[3], re_x.shape[4]))
    b, s, k, h, d = batch_size, seq_len, topk, re_x.shape[2], re_x.shape[3]
    gathered_x = re_x.reshape((re_x.shape[0] * re_x.shape[1], re_x.shape[2], re_x.shape[3]))
    # with torch.no_grad():
    #     gathered_x[mask.reshape(-1).bitwise_not().unsqueeze(-1).unsqueeze(-1).expand_as(gathered_x)]
    # gathered_x = torch.where(mask.reshape(-1).unsqueeze(-1).unsqueeze(-1).expand_as(gathered_x), gathered_x, torch.zeros_like(gathered_x))
    mask_expanded = mask.reshape(-1).unsqueeze(-1).unsqueeze(-1).expand_as(gathered_x)
    gathered_x = gathered_x * mask_expanded

    assert (indices >= 0).all(), "Indices should be non-negative"

    resortd_x = torch.zeros((b * s * k, h, d) ,device=gathered_x.device, dtype=gathered_x.dtype).scatter_add_(
        0,
        indices.reshape(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, h, d),
        gathered_x,
    )
    assert (indices < resortd_x.size(0)).all(), "Indices should be less than resortd_x size"

    max_value = resortd_x.max()
    max_idx = resortd_x.argmax()
    # with open('/mnt/workspace/flash-linear-attention/output', "a+") as f:
    #     f.write('gathered_x\n')
    #     f.write(f'{str(gathered_x)}\n')
    #     f.write('resorted_x\n')
    #     f.write(f'{str(resortd_x)}\n')
    # print(f'{max_idx}, {max_value}')
    if max_value > 2:
        debug = True
    # if indices[0][0] == 0:
    #     resortd_x[0] = gathered_x[0]
    # else:
    #     resortd_x[0] = 0.0
    inverse_indices = sorted_indices.argsort()
    rearranged_x_flat = resortd_x[inverse_indices]
    restored_x = rearranged_x_flat.reshape((b, s * k, h, d))
    restored_x = restored_x.reshape(b, s, k, h, d) * routing_weights.reshape(b, s, k).unsqueeze(-1).unsqueeze(-1)
    restored_x = restored_x.sum(dim=2)
    return restored_x.contiguous()


@dataclass
class MomLinearAttentionSubmodules:
    qkv_proj: Union[ModuleSpec, type] = None
    o_gate_proj: Union[ModuleSpec, type] = None
    gk_proj: Union[ModuleSpec, type] = None
    core_linear_attention: Union[ModuleSpec, type] = None
    o_proj: Union[ModuleSpec, type] = None


class MomLinearAttention(MegatronModule):
    def __init__(
        self,
        config,
        submodules: MomLinearAttentionSubmodules,
        layer_number=None,
    ):
        super().__init__(config)
        self.num_memories = config.num_memories
        self.topk = config.topk
        self.capacity = config.capacity
        # TODO: support shared mem
        self.shared_mem = config.shared_mem

        self.config = config
        self.la_module = self.config.la_module
        self.hidden_size = self.config.hidden_size
        self.query_dim = self.config.hidden_size
        self.key_dim = int(self.config.hidden_size * self.config.expand_k)
        self.value_dim = int(self.config.hidden_size * self.config.expand_v)
        self.head_dim = self.config.kv_channels
        self.num_heads = self.config.num_attention_heads
        self.la_gate_fn = self.config.la_gate_fn
        self.layer_number = layer_number

        # For normal attention without groups, num_query_groups == num_attention_heads,
        # so these two will be the same
        self.query_projection_size = self.config.kv_channels * self.config.num_attention_heads
        self.kv_projection_size = self.config.kv_channels * self.config.num_query_groups

        self.gate = torch.nn.Linear(self.hidden_size, self.num_memories, bias=False)

        # Per attention head and per partition values.
        tensor_model_parallel_world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.hidden_size_per_attention_head = divide(
            self.query_projection_size, self.config.num_attention_heads
        )
        self.num_attention_heads_per_partition = divide(self.config.num_attention_heads, tensor_model_parallel_world_size)
        self.num_query_groups_per_partition = divide(self.config.num_query_groups, tensor_model_parallel_world_size)

        self.qkv_proj = torch.nn.ModuleList([build_module(
            submodules.qkv_proj,
            self.config.hidden_size,
            self.query_dim+2*self.key_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=(self.config.add_bias_linear or self.config.add_qkv_bias) if self.config.base_model=='qwen2' else self.config.add_bias_linear,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='qkv',
        ) for m in range(self.num_memories)])

        if self.shared_mem:
            self.shared_qkv_proj = build_module(
                submodules.qkv_proj,
                self.config.hidden_size,
                self.query_dim+2*self.key_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=(self.config.add_bias_linear or self.config.add_qkv_bias) if self.config.base_model=='qwen2' else self.config.add_bias_linear,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name='qkv',
            )
        
        self.o_gate_proj = build_module(
            submodules.o_gate_proj,
            self.config.hidden_size,
            self.query_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=(self.config.add_bias_linear or self.config.add_qkv_bias) if self.config.base_model=='qwen2' else self.config.add_bias_linear,
            skip_bias_add=False,
            is_expert=False,
        )
        
        if self.la_module == 'mom_gla':
            self.gk_proj = torch.nn.ModuleList([build_module(
                submodules.gk_proj,
                self.config,
            ) for m in range(self.num_memories)])
            if self.shared_mem:
                self.shared_gk_proj = build_module(
                    submodules.gk_proj,
                    self.config,
                )
        
        if self.la_module == 'mom_gated_deltanet':
            self.beta_proj = torch.nn.ModuleList([torch.nn.Linear(self.hidden_size, self.num_heads, bias=False) for m in range(self.num_memories)])
            self.a_proj = torch.nn.ModuleList([torch.nn.Linear(self.hidden_size, self.num_heads, bias=False) for m in range(self.num_memories)])
            if self.shared_mem:
                self.shared_beta_proj = torch.nn.Linear(self.hidden_size, self.num_heads, bias=False)
                self.shared_a_proj = torch.nn.Linear(self.hidden_size, self.num_heads, bias=False)
            A = torch.empty(self.num_heads, dtype=torch.float32).uniform_(0, 16)
            A_log = torch.log(A)
            self.A_log = torch.nn.Parameter(A_log)
            self.A_log._no_weight_decay = True
            self.D = torch.nn.Parameter(torch.ones(self.num_heads))
            self.D._no_weight_decay = True
            # hard coded for now
            dt_min = 0.001
            dt_max = 0.1
            dt_init_floor = 1e-4
            dt = torch.exp(
                torch.rand(self.num_heads) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            )
            dt = torch.clamp(dt, min=dt_init_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_bias = torch.nn.Parameter(inv_dt)
            # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
            # name.endswith("bias") in param_grouping.py
            self.dt_bias._no_weight_decay = True
        
        self.core_linear_attention = build_module(
            submodules.core_linear_attention,
            config=self.config,
            expand_k=self.config.expand_k,
            expand_v=self.config.expand_v,
        )
        
        self.o_proj = build_module(
            submodules.o_proj,
            self.query_dim,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear, # false
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name='proj',
        )
        
        self.la_gate_fn = ACT2FN[self.la_gate_fn]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        inference_params=None,
        rotary_pos_emb=None,
        packed_seq_params=None,
    ) -> torch.Tensor:
        
        original_hidden = hidden_states

        o_gate, _ = self.o_gate_proj(hidden_states)

        hidden_states = hidden_states.transpose(0, 1).contiguous()
        
        router_logits = self.gate(hidden_states)  # (bsz, q_len, num_memories)
        scores = F.softmax(router_logits, dim=2, dtype=torch.float)
        routing_weights, selected_memories = torch.topk(scores, self.topk, dim=-1)  # (bsz, seq, topk)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)  # we cast back to the input dtype
        routing_weights_full = torch.zeros((routing_weights.shape[0], routing_weights.shape[1], self.num_memories), dtype=routing_weights.dtype, device=routing_weights.device).scatter(-1, selected_memories, routing_weights)
        routing_mask = routing_weights_full.bool().int()
        
        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        hidden_states, indices, sorted_indices, max_len, mask = transform(hidden_states, routing_mask, self.num_memories, selected_memories, self.capacity)

        # hidden_states = hidden_states.transpose(1, 2)

        qkv = torch.stack([self.qkv_proj[m](hidden_states[m])[0] for m in range(self.num_memories)], dim=0)
        if self.la_module == 'mom_gla':
            gk = torch.stack([self.gk_proj[m](hidden_states[m]) for m in range(self.num_memories)], dim=0)
        elif self.la_module == 'mom_gated_deltanet':
            beta = torch.stack([self.beta_proj[m](hidden_states[m]).sigmoid() for m in range(self.num_memories)], dim=0)
            gk = torch.stack([-self.A_log.float().exp() * F.softplus(self.a_proj[m](hidden_states[m]).float() + self.dt_bias) for m in range(self.num_memories)], dim=0)
        else:
            beta = None
            gk = None

        q, k, v = torch.split(
            qkv.view(qkv.size()[:-1] + (self.num_attention_heads_per_partition, -1)),
            [self.head_dim, self.head_dim, self.head_dim],
            dim=4,
        )
        
        # dealing with left-padding
        # if attention_mask is not None:
        #     v = v.mul_(attention_mask.unsqueeze(-1))
        
        # For self attention we just duplicate the rotary_pos_emb if it isn't already
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        # ===================================================
        # Adjust key, value, and rotary_pos_emb for inference
        # ===================================================
        _, _, rotary_pos_emb, attn_mask_type = self._adjust_key_value_for_inference(
            inference_params, k, v, rotary_pos_emb
        )

        if packed_seq_params is not None:
            q = q.squeeze(1)
            k = k.squeeze(1)
            v = v.squeeze(1)

        rotary_pos_emb = None # for linear attention
        # ================================================
        # relative positional embedding (rotary embedding)
        # ================================================
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb

            if packed_seq_params is not None:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
            else:
                cu_seqlens_q = cu_seqlens_kv = None
            if self.config.base_model == 'mixtral':
                q = apply_rotary_pos_emb(
                    q, q_pos_emb, fused=self.config.apply_rope_fusion, cu_seqlens=cu_seqlens_q
                )
                k = apply_rotary_pos_emb(
                    k, k_pos_emb, fused=self.config.apply_rope_fusion, cu_seqlens=cu_seqlens_kv
                )
            else:
                q = apply_rotary_pos_emb(
                    q, q_pos_emb, config=self.config, cu_seqlens=cu_seqlens_q,
                )
                k = apply_rotary_pos_emb(
                    k, k_pos_emb, config=self.config, cu_seqlens=cu_seqlens_kv,
                )

            # TODO, can apply positional embedding to value_layer so it has
            # absolute positional embedding.
            # otherwise, only relative positional embedding takes effect
            # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)
        
        # expect q: e n b h d
        o_list = [None for _ in range(self.num_memories)]
        for e in range(self.num_memories):
            if self.la_module == 'mom_gla':
                o_e = self.core_linear_attention(
                    q=q[e].transpose(0, 1),
                    k=k[e].transpose(0, 1),
                    v=v[e].transpose(0, 1),
                    gk=gk[e].transpose(0, 1),
                )
            elif self.la_module == 'mom_deltanet':
                o_e = self.core_linear_attention(
                    q=q[e].transpose(0, 1),
                    k=k[e].transpose(0, 1),
                    v=v[e].transpose(0, 1),
                    beta=beta[e].transpose(0, 1),
                )
            elif self.la_module == 'mom_gated_deltanet':
                o_e = self.core_linear_attention(
                    q=q[e].transpose(0, 1),
                    k=k[e].transpose(0, 1),
                    v=v[e].transpose(0, 1),
                    beta=beta[e].transpose(0, 1),
                    gk=gk[e].transpose(0, 1),
                )
            elif self.la_module == 'mom_mixattention':
                o_e = self.core_linear_attention(
                    x=hidden_states,
                    q=q[e].transpose(0, 1),
                    k=k[e].transpose(0, 1),
                    v=v[e].transpose(0, 1),
                )
            else:
                o_e = self.core_linear_attention(
                    q=q[e].transpose(0, 1),
                    k=k[e].transpose(0, 1),
                    v=v[e].transpose(0, 1),
                )
            
            o_e = rearrange(o_e, 'n b (h d) -> b n h d', h = self.num_heads).contiguous()
            o_e = o_e[:,-max_len:,:,:].to(dtype=q[e].dtype)
            o_list[e] = o_e
        o_list = torch.stack(o_list, dim=0)
        o = reconstruct(o_list, indices=indices, sorted_indices=sorted_indices, batch_size=q.shape[1], seq_len=seq_len, topk=self.topk, routing_weights=routing_weights, mask=mask)
        o = rearrange(o, 'b n h d -> n b (h d)').contiguous()

        if self.shared_mem:
            o += self.shared_forward(original_hidden, attention_mask, inference_params, rotary_pos_emb, packed_seq_params)

        # o: n b (h d)
        o = o * self.la_gate_fn(o_gate)
        o, bias = self.o_proj(o)

        return o, bias
    
    def _allocate_memory(self, inference_max_sequence_length, batch_size, dtype):
        """Allocate memory to store kv cache during inference."""

        return torch.empty(
            inference_max_sequence_length,
            batch_size,
            self.num_query_groups_per_partition,
            self.hidden_size_per_attention_head,
            dtype=dtype,
            device=torch.cuda.current_device(),
        )

    def _adjust_key_value_for_inference(self, inference_params, key, value, rotary_pos_emb):
        """
        Saves the generated key and value tensors to the end of the buffers in inference_params.
        Returns the full size keys and values from the provided inference_params, as well as
        adjusted rotary_pos_emb.

        Returns a tuple: (key, value, rotary_pos_emb)

        """
        attn_mask_type = AttnMaskType.causal # self.attn_mask_type
        if inference_params is None:
            return key, value, rotary_pos_emb, attn_mask_type

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        is_first_step = False
        if self.layer_number not in inference_params.key_value_memory_dict:
            inf_max_seq_length = inference_params.max_sequence_length
            inf_max_batch_size = inference_params.max_batch_size
            inference_key_memory = self._allocate_memory(
                inf_max_seq_length, inf_max_batch_size, key.dtype
            )
            inference_value_memory = self._allocate_memory(
                inf_max_seq_length, inf_max_batch_size, value.dtype
            )
            inference_params.key_value_memory_dict[self.layer_number] = (
                inference_key_memory,
                inference_value_memory,
            )
            is_first_step = True
        else:
            # Get the pre-allocated buffers for this layer
            inference_key_memory, inference_value_memory = inference_params.key_value_memory_dict[
                self.layer_number
            ]
            attn_mask_type = AttnMaskType.no_mask

        batch_start = inference_params.batch_size_offset
        batch_end = batch_start + key.size(1)
        assert batch_end <= inference_key_memory.size(1)
        sequence_start = inference_params.sequence_len_offset
        sequence_end = sequence_start + key.size(0)
        assert sequence_end <= inference_key_memory.size(0)
        # Copy key and values.
        inference_key_memory[sequence_start:sequence_end, batch_start:batch_end, ...] = key
        inference_value_memory[sequence_start:sequence_end, batch_start:batch_end, ...] = value
        key = inference_key_memory[:sequence_end, batch_start:batch_end, ...]
        value = inference_value_memory[:sequence_end, batch_start:batch_end, ...]

        # adjust the key rotary positional embedding
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            # need to cross check this condition during inference
            # if not set_inference_key_value_memory:
            if not is_first_step:
                # In inference, we compute one token at a time.
                # Select the correct positional embedding
                # (only the last token in the sequence)
                q_pos_emb = q_pos_emb[sequence_end - 1 : sequence_end]
            else:
                # In the first forward pass of inference,
                # we use the entire provided prefix.
                # q_pos_emb here has the rope embeddings of the entire
                # prefix + to-be-generated output so
                # we slice to just the prefix.
                q_pos_emb = q_pos_emb[:sequence_end, :, :, :]
            k_pos_emb = k_pos_emb[:sequence_end, :, :, :]
            rotary_pos_emb = (q_pos_emb, k_pos_emb)

        return key, value, rotary_pos_emb, attn_mask_type

    def shared_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        inference_params=None,
        rotary_pos_emb=None,
        packed_seq_params=None,
    ) -> torch.Tensor:

        qkv, _ = self.shared_qkv_proj(hidden_states)
        o_gate, _ = self.o_gate_proj(hidden_states)
        if self.la_module == 'mom_gla':
            gk = self.shared_gk_proj(hidden_states)
        elif self.la_module == 'mom_deltanet':
            beta = rearrange(self.shared_beta_proj(hidden_states), 'b n h -> b h n').sigmoid()
        elif self.la_module == 'mom_gated_deltanet':
            beta = self.shared_beta_proj(hidden_states.transpose(0, 1)).sigmoid().transpose(0, 1)
            gk = -self.A_log.float().exp() * F.softplus(self.shared_a_proj(hidden_states.transpose(0, 1)).float() + self.dt_bias).transpose(0, 1)
        else:
            beta = None
            gk = None

        q, k, v = torch.split(
            qkv.view(qkv.size()[:-1] + (self.num_attention_heads_per_partition, -1)),
            [self.head_dim, self.head_dim, self.head_dim],
            dim=3,
        )
        
        # dealing with left-padding
        # if attention_mask is not None:
        #     v = v.mul_(attention_mask.unsqueeze(-1))
        
        # For self attention we just duplicate the rotary_pos_emb if it isn't already
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        # ===================================================
        # Adjust key, value, and rotary_pos_emb for inference
        # ===================================================
        _, _, rotary_pos_emb, attn_mask_type = self._adjust_key_value_for_inference(
            inference_params, k, v, rotary_pos_emb
        )

        if packed_seq_params is not None:
            q = q.squeeze(1)
            k = k.squeeze(1)
            v = v.squeeze(1)

        rotary_pos_emb = None # for linear attention
        # ================================================
        # relative positional embedding (rotary embedding)
        # ================================================
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb

            if packed_seq_params is not None:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
            else:
                cu_seqlens_q = cu_seqlens_kv = None
            if self.config.base_model == 'mixtral':
                q = apply_rotary_pos_emb(
                    q, q_pos_emb, fused=self.config.apply_rope_fusion, cu_seqlens=cu_seqlens_q
                )
                k = apply_rotary_pos_emb(
                    k, k_pos_emb, fused=self.config.apply_rope_fusion, cu_seqlens=cu_seqlens_kv
                )
            else:
                q = apply_rotary_pos_emb(
                    q, q_pos_emb, config=self.config, cu_seqlens=cu_seqlens_q,
                )
                k = apply_rotary_pos_emb(
                    k, k_pos_emb, config=self.config, cu_seqlens=cu_seqlens_kv,
                )

            # TODO, can apply positional embedding to value_layer so it has
            # absolute positional embedding.
            # otherwise, only relative positional embedding takes effect
            # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)
        
        # expect q: n b h d
        if self.la_module == 'mom_gla':
            o = self.core_linear_attention(
                q=q,
                k=k,
                v=v,
                gk=gk,
            )
        elif self.la_module == 'mom_gated_deltanet':
            o = self.core_linear_attention(
                q=q,
                k=k,
                v=v,
                beta=beta,
                gk=gk,
            )
        else:
            o = self.core_linear_attention(
                q=q,
                k=k,
                v=v,
            )

        return o