# coding=utf-8
# Copyright 2023 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
""" PyTorch LongLLaMA model."""
from dataclasses import dataclass
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_longllama import LongLlamaConfig
from .longllama_utils import mem_apply_update, LongLlamaMemCache, LongLlamaMemConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LongLlamaConfig"


@dataclass
class LongLlamaModelOutputWithPast(BaseModelOutputWithPast):
    """
    Based on BaseModelOutputWithPast

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
        mem_caches  (`tuple(LongLlamaMemCache))`, *optional*, returned for layers with memory cache enabled):
            For the layers without memory None is returned
    """

    mem_caches: Optional[LongLlamaMemCache] = None


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->LongLlama
class LongLlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LongLlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        return (self.weight * hidden_states).to(input_dtype)


# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->LongLlama
class LongLlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        dtype = torch.get_default_dtype()
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(x.dtype), persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(x.dtype), persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Based on transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def rotate_one(x, cos, sin, position_ids):
    if len(position_ids.shape) != 2 or x.shape[0] != position_ids.shape[0] or x.shape[-2] != position_ids.shape[1]:
        raise ValueError(f"Position ids shoud have shape [bsz, seq_len] got {position_ids.shape}")
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


def rotate_as_if_first(x, rotary_emb):
    # x: [bs, num_attention_heads, seq_len, head_size]
    # apply rotary as if all elements were first in the sequence
    cos, sin = rotary_emb(x, x.shape[-2])
    return rotate_one(x, cos, sin, torch.zeros(x.shape[0], x.shape[-2], dtype=torch.long, device=cos.device))


# Copied from transformers.models.llama.modeling_llama.LlamaMLP with Llama->LongLlama
class LongLlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# Modified transformers.models.llama.modeling_llama.LlamaAttention
class LongLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper with FoT modifications"""

    def __init__(self, config: LongLlamaConfig, mem_config: Optional[LongLlamaMemConfig] = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.max_cache = self.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = LongLlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)
        self.mem_config = mem_config

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        mem_cache: Optional[LongLlamaMemCache] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if attention_mask is None:
            tgt_seq_len = hidden_states.shape[-2]
            if past_key_value is not None:
                src_seq_len = past_key_value[0].shape[-2] + tgt_seq_len
            else:
                src_seq_len = tgt_seq_len

            attention_mask = torch.zeros(
                hidden_states.shape[0],
                1,
                tgt_seq_len,
                src_seq_len,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        position_ids = position_ids[:, None, :, None]

        if position_ids.shape != (key_states.shape[0], 1, key_states.shape[-2], 1):
            raise ValueError("position_ids should match batch and seq_len of the input")

        mem_no_local_cache = self.mem_config is not None and past_key_value is None and (not use_cache)
        mem_and_local_cache = self.mem_config is not None and use_cache
        # positonal embeddings can be disabled for memory layers
        use_positionals = self.mem_config is None or self.mem_config.positionals

        if mem_no_local_cache:
            # the whole context window will be moved to memory cache after the attention
            if use_positionals:
                # positionally embedd memory content as first token in the sequence
                rfst_key_states = rotate_as_if_first(key_states, self.rotary_emb)
            else:
                rfst_key_states = key_states
            # attention_mask [bsz, 1, tgt_seq_len, src_seq_len]
            # we base the mask on the last token in the context window
            mem_update = LongLlamaMemCache(
                keys=rfst_key_states.to(self.mem_config.cache_dtype),
                values=value_states.to(self.mem_config.cache_dtype),
                masks=attention_mask[..., -1, :, None],
            )

        if past_key_value is not None:
            past_local_cache_size = past_key_value[0].shape[-2]
            key_states = torch.cat([past_key_value[0], key_states], dim=-2)
            value_states = torch.cat([past_key_value[1], value_states], dim=-2)
            # FoT additionally stores position_ids to support long inputs
            position_ids = torch.cat([past_key_value[2], position_ids], dim=-2)

            if attention_mask.shape[-1] != key_states.shape[-2] and attention_mask.shape[-2] != query_states.shape[-2]:
                raise ValueError("attention_mask should be provided for all key_states in local context")

            # local cache is maintained so that it is <= self.max_cache
            # remaining elements are either dropped or go to memory cache
            if key_states.shape[-2] > self.max_cache:
                num_elems_to_drop = past_local_cache_size

                if mem_and_local_cache:
                    drop_keys = key_states[:, :, :num_elems_to_drop, :]
                    drop_values = value_states[:, :, :num_elems_to_drop, :]
                    # as memory mask use the masking of the last key in context
                    # attention_mask [bsz, 1, tgt_seq_len, src_seq_len]
                    drop_masks = attention_mask[..., -1, :, None]
                    drop_masks = drop_masks[:, :, :num_elems_to_drop, :]

                    if use_positionals:
                        rfst_drop_keys = rotate_as_if_first(drop_keys, self.rotary_emb)
                    else:
                        rfst_drop_keys = drop_keys
                    mem_update = LongLlamaMemCache(
                        keys=rfst_drop_keys.to(self.mem_config.cache_dtype),
                        values=drop_values.to(self.mem_config.cache_dtype),
                        masks=drop_masks,
                    )
                    if mem_cache is None:
                        mem_cache = mem_update
                    else:
                        mem_cache = mem_apply_update(
                            prev_mem_cache=mem_cache, new_mem_content=mem_update, mem_config=self.mem_config
                        )

                key_states = key_states[:, :, num_elems_to_drop:, :]
                value_states = value_states[:, :, num_elems_to_drop:, :]
                position_ids = position_ids[:, :, num_elems_to_drop:, :]
                attention_mask = attention_mask[..., num_elems_to_drop:]


        # FoT additionally stores position_ids to support long inputs
        past_key_value = (key_states, value_states, position_ids) if use_cache else None

        kv_seq_len = key_states.shape[-2]

        if use_positionals:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            rel_pos_ids = position_ids - torch.min(position_ids, dim=-2, keepdim=True)[0]
            rel_pos_ids = rel_pos_ids.squeeze(3).squeeze(1)

            query_states = rotate_one(query_states, cos, sin, rel_pos_ids[:, -query_states.shape[-2] :])
            key_states = rotate_one(key_states, cos, sin, rel_pos_ids)


        if self.mem_config is not None and self.mem_config.attention_grouping is not None:
            attn_grouping_h, attn_grouping_q = self.mem_config.attention_grouping
            if attn_grouping_h <= 0 or attn_grouping_q <= 0:
                raise ValueError("Attention grouping should be positive")
        else:
            attn_grouping_h, attn_grouping_q = self.num_heads, q_len
            
        
        attn_output_h = []
        for beg_h in range(0, self.num_heads, attn_grouping_h):
            end_h = min(beg_h + attn_grouping_h, self.num_heads)

            attn_output_q = []
            for beg_q in range(0, q_len, attn_grouping_q):
                end_q = min(beg_q + attn_grouping_q, q_len)

                attn_weights = torch.matmul(query_states[:, beg_h:end_h, beg_q:end_q], key_states[:, beg_h:end_h].transpose(2, 3)) / math.sqrt(self.head_dim)

                if attn_weights.size() != (bsz, end_h - beg_h, end_q - beg_q, kv_seq_len):
                    raise ValueError(
                        f"Attention weights should be of size {(bsz, end_h - beg_h, end_q - beg_q, kv_seq_len)}, but is"
                        f" {attn_weights.size()}"
                    )

                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask[:, :, beg_q:end_q]
                min_value = torch.finfo(attn_weights.dtype).min if -1000000.0 < torch.finfo(attn_weights.dtype).min else -1000000.0 
                attn_weights = torch.max(attn_weights, torch.tensor(min_value, device=attn_weights.device, dtype=attn_weights.dtype))

                if mem_cache is not None:
                    mem_mask = mem_cache.masks.squeeze(-1).unsqueeze(-2)
                    mem_attn_weights = torch.matmul(
                        query_states[:, beg_h:end_h, beg_q:end_q], mem_cache.keys[:, beg_h:end_h].transpose(2, 3).to(key_states.dtype)
                    ) / math.sqrt(self.head_dim)

                    assert mem_mask.shape[2] == 1
                    mem_attn_weights = mem_attn_weights + mem_mask
                    min_value = torch.finfo(mem_attn_weights.dtype).min if -1000000.0 < torch.finfo(mem_attn_weights.dtype).min else -1000000.0
                    mem_attn_weights = torch.max(mem_attn_weights, torch.tensor(min_value, device=mem_attn_weights.device, dtype=mem_attn_weights.dtype))

                    attn_weights = torch.concat([attn_weights, mem_attn_weights], dim=-1)
                    combined_value_states = torch.concat([value_states[:, beg_h:end_h], mem_cache.values[:, beg_h:end_h].to(value_states.dtype)], dim=-2)
                else:
                    combined_value_states = value_states[:, beg_h:end_h]
                # upcast attention to fp32
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_output = torch.matmul(attn_weights, combined_value_states)
                assert attn_output.shape[-2] == end_q - beg_q
                attn_output_q.append(attn_output)
            attn_output_h.append(torch.concat(attn_output_q, dim=-2))

        attn_output = torch.concat(attn_output_h, dim=-3)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        if mem_no_local_cache:
            if mem_cache is not None:
                mem_cache = mem_apply_update(prev_mem_cache=mem_cache, new_mem_content=mem_update, mem_config=self.mem_config)
            else:
                mem_cache = mem_update

        return attn_output, attn_weights, past_key_value, mem_cache


# Modified transformers.models.llama.modeling_llama.LlamaDecoderLayer
class LongLlamaDecoderLayer(nn.Module):
    def __init__(self, config: LongLlamaConfig, mem_config: Optional[LongLlamaMemConfig] = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LongLlamaAttention(config=config, mem_config=mem_config)
        self.mlp = LongLlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = LongLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LongLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        mem_cache: Optional[LongLlamaMemCache] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor, LongLlamaMemCache]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
                along with information about positions
            mem_cache (`LongLlamaMemCache`, *optional*): memory cache for specific layers
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value, mem_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            mem_cache=mem_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs + (mem_cache, )


LONGLLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LongLlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""
LONGLLAMA_MEML_DOCSTRING = r"""
        mem_layers ([`int`], *optional*):
            Indices of layers to be augmented with memory, if None then parameters from config will be used
        mem_dtype (`str`, *optional*):
            Keys and values will be casted to this type for storage.

"""


@add_start_docstrings(
    "The bare LongLLaMA Model outputting raw hidden-states without any specific head on top.",
    LONGLLAMA_START_DOCSTRING,
)
# Copied from transformers.models.llama.modeling_llama.LlamaPreTrainedModel with Llama->LongLlama
class LongLlamaPreTrainedModel(PreTrainedModel):
    config_class = LongLlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LongLlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LongLlamaModel):
            module.gradient_checkpointing = value


LONGLLAMA_COMMON_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`
            or memory cache is enabled):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 1 additional tensor of shape
            `(batch_size, 1, sequence_length, 1)`. For memory enriched layers it also contains content of memory cache.
            It is padded with empty tensors so when returned it alwyas has 6 elements.

            Contains pre-computed hidden-states (key and values in the self-attention blocks) 
            that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail. This is NOT supported in LongLlamaForCausalLM and LongLlamaForSequenceClassification
            due to the specific input processing.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""
LONGLLAMA_MODEL_INPUTS_DOCSTRING = r"""
        mem_caches (`tuple(LongLlamaMemCache)`, *optional*) 
            Memory caches for specified layers, None for others
"""

LONGLLAMA_ADD_INPUTS_DOCSTRING = r"""
        last_context_length (`int`, *optional*) 
            Useful for generation, specifies number of tokens that won't be loaded to memory and 
            will be left for generation cache
"""


def _prepare_pos_ids(past_key_values, batch_size, input_length, device):
    if past_key_values is not None:
        # take previous max pos_id + 1
        if past_key_values[0][2].shape[0] != batch_size:
            raise ValueError(f"first dimension of past_key_values should match batch size: {batch_size}"
                             f"but got {past_key_values[0][2].shape[0]}")
        next_pos = torch.max(past_key_values[0][2].view(batch_size, -1), dim=-1)[0] + 1
        next_pos = next_pos.view(batch_size, 1)
    else:
        next_pos = torch.zeros(batch_size, 1, device=device, dtype=torch.long)

    position_ids = torch.arange(0, input_length, dtype=torch.long, device=device).view(1, input_length)
    position_ids = position_ids + next_pos
    return position_ids


@add_start_docstrings(
    "The bare LongLLaMA Model outputting raw hidden-states without any specific head on top.",
    LONGLLAMA_START_DOCSTRING,
    LONGLLAMA_MEML_DOCSTRING,
)
# Modified transformers.models.llama.modeling_llama.LlamaModel
class LongLlamaModel(LongLlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LongLlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LongLlamaConfig):
        super().__init__(config)
        self.mem_layers = config.mem_layers
        self.mem_config = LongLlamaMemConfig(positionals=config.mem_positionals, cache_dtype=getattr(torch, config.mem_dtype), attention_grouping=config.mem_attention_grouping)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        

        for mem_layer_id in self.mem_layers :
            if mem_layer_id < 0 or mem_layer_id >= config.num_hidden_layers:
                raise ValueError(f"Memory layer ids should be between 0 and {config.num_hidden_layers}, got {mem_layer_id}")

        layers = []
        for layer_id in range(config.num_hidden_layers):
            if layer_id in self.mem_layers:
                layer = LongLlamaDecoderLayer(config, mem_config=self.mem_config)
            else:
                layer = LongLlamaDecoderLayer(config, mem_config=None)
            layers.append(layer)

        self.layers = nn.ModuleList(layers)
        self.norm = LongLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    @add_start_docstrings_to_model_forward(LONGLLAMA_COMMON_INPUTS_DOCSTRING, LONGLLAMA_MODEL_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        mem_caches: Optional[Tuple[Optional[LongLlamaMemCache]]] = None,
    ) -> Union[Tuple, LongLlamaModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[-2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = _prepare_pos_ids(past_key_values, batch_size, seq_length, device)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = ()
        next_mem_caches = ()
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None
            mem_cache = mem_caches[idx] if mem_caches else None

            if mem_cache is not None and idx not in self.mem_layers:
                raise ValueError("Memory cache provided for a non-memory leayer")


            if self.gradient_checkpointing and self.training and idx not in self.mem_layers:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None, mem_cache=None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    mem_cache=mem_cache,
                )

            new_mem_cache = layer_outputs[-1]
            layer_outputs = layer_outputs[:-1]
            next_mem_caches += (new_mem_cache,)

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
            else:
                next_decoder_cache += (None,)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        mem_cache_returned = False
        for mem_cache in next_mem_caches:
            if mem_cache is not None:
                mem_cache_returned = True
        next_mem_caches = next_mem_caches if mem_cache_returned else None

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, next_mem_caches]
                if v is not None
            )
        return LongLlamaModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            mem_caches=next_mem_caches,
        )


def _handle_output_of_past_key_values(outputs):
    # merges local caches and memory caches into one single tuple of past_key_values
    # in order to support generation
    batch_size = outputs.last_hidden_state.shape[0]
    if outputs.past_key_values is None and outputs.mem_caches is None:
        return None

    if outputs.past_key_values is None:
        out_past_key_values = (None,) * len(outputs.mem_caches)
    else:
        out_past_key_values = outputs.past_key_values

    if outputs.mem_caches is None:
        out_mem_caches = (None,) * len(outputs.past_key_values)
    else:
        out_mem_caches = outputs.mem_caches

    device = outputs.last_hidden_state.device
    past_key_values = ()
    for local_cache, mem_cache in zip(out_past_key_values, out_mem_caches):
        layer = ()
        if local_cache is not None:
            assert len(local_cache) == 3
            layer += local_cache
        else:
            layer += (torch.empty(batch_size, 0, 0, 0, device=device),) * 3

        if mem_cache is not None:
            layer += (mem_cache.keys, mem_cache.values, mem_cache.masks)
        else:
            layer += (torch.empty(batch_size, 0, 0, 0, device=device),) * 3

        assert len(layer) == 6

        past_key_values += (layer,)

    return past_key_values


def _split_past_key_values(past_key_values):
    # splits past_key_values to local cache and memory cache
    local_cache_preset = False
    mem_caches_present = False
    if past_key_values is not None:
        local_caches = ()
        mem_caches = ()
        for layer in past_key_values:
            if len(layer) != 6:
                raise ValueError(
                    "Expected elements of past_key_values to contain 6 elements."
                    "First 3 describing local cache and last 3 describing memory cache."
                    f"Instead got {len(layer)} elements"
                )
            else:
                lk, lv, li, memk, memv, memm = layer
                if lk.shape[-2] != 0:
                    local_cache_preset = True
                    local_caches += ((lk, lv, li),)
                else:
                    local_caches += (None,)

                if memk.shape[-2] != 0:
                    mem_caches_present = True
                    mem_caches += (LongLlamaMemCache(keys=memk, values=memv, masks=memm),)
                else:
                    mem_caches += (None,)

    local_caches = local_caches if local_cache_preset else None
    mem_caches = mem_caches if mem_caches_present else None

    return local_caches, mem_caches


def _handle_long_input(
    model,
    input_ids,
    attention_mask,
    position_ids,
    past_key_values,
    inputs_embeds,
    use_cache,
    output_attentions,
    output_hidden_states,
    return_dict,
    context_window_length,
    last_context_length,
):
    if output_attentions:
        logger.warning(f"Outputing attentions is not supported in LongLlamaForCausalLM and LongLlamaForSequenceClassification. "
                     f"Attention of the last window will be returned")

    past_key_values, mem_caches = _split_past_key_values(past_key_values)

    if past_key_values is not None and use_cache is False:
        raise ValueError("past_key_values it not None should imply use_cache == True")

    if past_key_values is not None:
        initial_past_key_values_length = past_key_values[0][0].shape[-2]
    else:
        initial_past_key_values_length = 0

    if input_ids is not None:
        batch_size, input_length = input_ids.shape
    else:
        batch_size, input_length, _ = inputs_embeds.shape

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = _prepare_pos_ids(past_key_values, batch_size, input_length, device)

    if position_ids.shape != (batch_size, input_length):
        raise ValueError(f"Shape of position_ids [{position_ids}] should match [{batch_size, input_length}]")

    if attention_mask is not None:
        attention_mask = attention_mask[..., -(initial_past_key_values_length + input_length) :]
        if attention_mask is not None and (
            attention_mask.shape != (batch_size, initial_past_key_values_length + input_length)
        ):
            raise ValueError(
                "Attention mask should be provided for both the local cache and the input",
                f"Expected shape {(batch_size, initial_past_key_values_length + input_length)},"
                f"got {attention_mask.shape}.",
            )

    # First we load prefix to memory cache
    mem_input_length = max(input_length - last_context_length, 0)
    outputs_list = []
    attn_offset = initial_past_key_values_length
    if mem_input_length > 0:
        for i in range(0, mem_input_length, context_window_length):
            beg, end = i, min(mem_input_length, i + context_window_length)

            if attention_mask is not None:
                if past_key_values is not None:
                    local_cache_size = past_key_values[0][0].shape[-2]
                else:
                    local_cache_size = 0
                attn_length = attention_mask.shape[-1]
                attn_beg = beg + attn_offset - local_cache_size
                attn_end = end + attn_offset
                assert attn_end <= attn_length
                assert attn_beg >= 0 and attn_end > attn_beg

            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn, mem_caches)
            outputs = model(
                input_ids=input_ids[..., beg:end] if input_ids is not None else None,
                attention_mask=attention_mask[..., attn_beg:attn_end] if attention_mask is not None else None,
                position_ids=position_ids[..., beg:end],
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds[..., beg:end, :] if inputs_embeds is not None else None,
                use_cache=False if past_key_values is None else use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                mem_caches=mem_caches,
            )
            if i > 0:
                if mem_caches is not None and past_key_values is None:
                    for mc_layer in mem_caches:
                        if mc_layer is not None:
                            del mc_layer.keys
                            del mc_layer.values
                            del mc_layer.masks

            mem_caches = outputs.mem_caches
            outputs.mem_caches = None
            past_key_values = outputs.past_key_values
            outputs.past_key_values = None
            outputs_list.append(outputs)

    remaining_input_length = input_length - mem_input_length
    beg = mem_input_length
    attn_length = remaining_input_length
    if past_key_values is not None:
        attn_length += past_key_values[0][0].shape[-2]
    attention_mask = attention_mask[..., -attn_length:] if attention_mask is not None else None

    outputs = model(
        input_ids=input_ids[..., beg:] if input_ids is not None else None,
        attention_mask=attention_mask,
        position_ids=position_ids[..., beg:],
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds[..., beg:, :] if inputs_embeds is not None else None,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        mem_caches=mem_caches,
    )

    outputs_list.append(outputs)

    past_key_values = _handle_output_of_past_key_values(outputs_list[-1])

    if output_hidden_states:
        hidden_states = ()
        for hd in zip(*[x.hidden_states for x in outputs_list]):
            hidden_states += (torch.cat(hd, dim=-2),)
    else:
        hidden_states = None

    outputs = BaseModelOutputWithPast(
        last_hidden_state=torch.concat([x.last_hidden_state for x in outputs_list], dim=-2),
        past_key_values=past_key_values,
        hidden_states=hidden_states,
        attentions=outputs_list[-1].attentions,
    )

    if not return_dict:
        outputs =tuple(
            v for v in [outputs.last_hidden_state, outputs.past_key_values, outputs.hidden_states, outputs.attentions]
            if v is not None
        )
    return outputs


# Modified transformers.models.llama.modeling_llama.LlamaForCausalLM
class LongLlamaForCausalLM(LongLlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(
        self, config
    ):
        super().__init__(config)
        self.context_window_length = config.max_position_embeddings
         
        self.model = LongLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def _has_generation_cache(self, past_key_values):
        if past_key_values is not None:
            assert len(past_key_values[0]) == 6
            return past_key_values[0][0].shape[-2] != 0

        return False

    @add_start_docstrings_to_model_forward(LONGLLAMA_COMMON_INPUTS_DOCSTRING, LONGLLAMA_ADD_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        last_context_length: Optional[int] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        last_context_length = last_context_length if last_context_length is not None else self.config.last_context_length
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = _handle_long_input(
            model=self.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            context_window_length=self.context_window_length,
            last_context_length=last_context_length,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, last_context_length=None, **kwargs
    ):
        if self._has_generation_cache(past_key_values):
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill(position_ids < 0, 0)
            if self._has_generation_cache(past_key_values):
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "last_context_length": last_context_length,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


@add_start_docstrings(
    """
    The LongLLaMA Model transformer with a sequence classification head on top (linear layer).

    [`LongLlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    LONGLLAMA_START_DOCSTRING,
    LONGLLAMA_MEML_DOCSTRING
)
# Modified from transformers.models.llama.modeling_llama.LlamaForSequenceClassification
class LongLlamaForSequenceClassification(LongLlamaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(
        self, config
    ):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.context_window_length = config.max_position_embeddings
        self.model = LongLlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LONGLLAMA_COMMON_INPUTS_DOCSTRING, LONGLLAMA_ADD_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        last_context_length: Optional[int] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        last_context_length = last_context_length if last_context_length is not None else self.config.last_context_length
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        transformer_outputs = _handle_long_input(
            model=self.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            context_window_length=self.context_window_length,
            last_context_length=last_context_length,
        )

        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
