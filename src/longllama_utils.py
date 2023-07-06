from collections import namedtuple
from dataclasses import dataclass
import torch
from typing import Tuple, Optional

@dataclass
class LongLlamaMemConfig:
    """
    Class for configuring memory caches for LongLlama model.

    Args:
        positionals (`boolean`)
            Whether to use positional embeddings in memory layer
        cache_dtype (`torch.dtype`)
            Specifies storing type for keys and values
        attention_grouping (`Tuple[int, int]`, *optional*)
            One can trade speed for memory by performing attention
            in memory layers sequentially. 
            When equal to `(4, 128)` the memory layers will process at most 4 heads and 128 queries
            from each head at once. That is at most 512 queries at once.
    """

    positionals: bool = True
    cache_dtype: torch.dtype = torch.bfloat16
    attention_grouping: Optional[Tuple[int, int]] = None


@dataclass
class LongLlamaMemCache:
    """
    Class with LongLlama's memory cache

    Args:
        keys (`torch.FloatTensor` of shape `(batch_size, num_heads, mem_length, embed_size_per_head)`)
        values (`torch.FloatTensor` of shape `(batch_size, num_heads, mem_length, embed_size_per_head)`)
        masks (`torch.FloatTensor` of shape `(batch_size, 1, mem_length, 1)`)
            For masking out parts of memory
    """

    keys: torch.FloatTensor
    values: torch.FloatTensor
    masks: torch.FloatTensor


def mem_apply_update(prev_mem_cache: LongLlamaMemCache, new_mem_content: LongLlamaMemCache, mem_config: LongLlamaMemConfig):
    def update_one(prev, new):
        if len(prev.shape) != 4 or len(new.shape) != 4:
            raise ValueError(f"Memory cache content should be consistent in shape got {prev.shape} {new.shape}")

        return torch.concat([prev, new], dim=-2)

    insert_size = new_mem_content.keys.shape[-2]

    if new_mem_content.values.shape[-2] != insert_size or new_mem_content.masks.shape[-2] != insert_size:
        raise ValueError(f"Inconsistent mem_length in new_mem_content")

    return LongLlamaMemCache(
        keys=update_one(prev_mem_cache.keys, new_mem_content.keys),
        values=update_one(prev_mem_cache.values, new_mem_content.values),
        masks=update_one(prev_mem_cache.masks, new_mem_content.masks),
    )
