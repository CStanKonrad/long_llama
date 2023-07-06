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
""" LongLLaMA model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

LONGLLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "syzymon/long_llama_3b": "https://huggingface.co/syzymon/long_llama_3b/resolve/main/config.json",
}


class LongLlamaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LongLlamaModel`]. It is used to instantiate an LongLLaMA
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the LongLLaMA-7B.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the LongLLaMA model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`LongLlamaModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings(`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        mem_layers (`List[int]`, defaults to `[]`):
            Layers with memory
        mem_positionals (`bool`, *optional*, defaults to `True`):
            Whether to use positional embeddings in memory layers
        mem_dtype (`str`, *optional*, defaults to `"bfloat16"`):
            Type for keys and values stored in memory
        mem_attention_grouping (`Tuple[int, int]`, *optional*, defaults to `None`):
            One can trade speed for memory by performing attention
            in memory layers sequentially. 
            When equal to `(4, 2048)` the memory layers will process at most 4 heads and 2048 queries from each head at once.
            That is at most 4*2048 queries at once.
        
        Example:

    ```python
    >>> from transformers import LongLlamaModel, LongLlamaConfig

    >>> # Initializing a LongLLaMA longllama-7b style configuration
    >>> configuration = LongLlamaConfig()

    >>> # Initializing a model from the longllama-7b style configuration
    >>> model = LongLlamaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "longllama"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        last_context_length=1024,
        mem_layers=[],
        mem_positionals=True,
        mem_dtype="bfloat16",
        mem_attention_grouping=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.last_context_length = last_context_length
        self.mem_layers = mem_layers
        self.mem_positionals = mem_positionals
        self.mem_dtype = mem_dtype
        self.mem_attention_grouping = mem_attention_grouping
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
