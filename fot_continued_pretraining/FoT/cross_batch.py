from typing import Any, Callable, Optional, Tuple
from flax.linen.linear import PrecisionLike
from flax.linen.dtypes import promote_dtype
import jax.numpy as jnp
import jax
import functools

Array = Any
PRNGKey = Any
Dtype = Any


def ceil_divide(a, b):
    assert b >= 1
    assert a >= 0
    return (a + b - 1) // b


# based on flax.linen.attention.dot_product_attention_weights
def cross_batch_attention(
    query: Array,
    key: Array,
    value: Array,
    bias: Array,
    min_value: float,
    cross_batch_range: int,
    cross_batch_stepping: bool,
    dataset_packing: int,
    pos_encode_as_first: Callable[[Array], Array],
    pos_encode: Callable[[Array, Array], Array],
    broadcast_dropout: bool = True,
    dropout_rng: Optional[PRNGKey] = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype: Optional[Dtype] = None,
    precision: PrecisionLike = None,
    custom_attention_fn=None,
):
    """
    Basic implementation of cross-batch.
    If dataset_packing > 0, it assumes that
    the documents occupy batch entries as follows
    batch[0] = doc_1_part_1
    batch[1] = doc_1_part_2
    ...
    batch[dataset_packing - 1] = doc_1_part_{dataset_packing}
    batch[dataset_packing] = doc_2_part_1
    That is, each document is assigned dataset_packing entries and
    after flattening document tokens are in order. 
    (if the document is too short then instead of padding we add <eos><bos> 
    and load the next document and tread those two documents as one)
    Args:
        query, key, value - tensors of shape [BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM]
        bias -  tensor of shape [BATCH_SIZE, 1, SEQ_LEN, SEQ_LEN] for attention masking.
                It will be added to local context attention
        min_value - used to mask in softmax
        cross_batch_range - the number of additional contexts used in cross-batch
        cross_batch_stepping - whether to use multiple cross-batch ranges
        dataset_packing - number of batch elements occupied by each doc
        pos_encode_as_first - for encoding keys as if they were first in the context
        pos_encode - for pos encoding keys and queries from the local context
        broadcast_dropout, dropout_rng, dropout_rate, deterministic, dtype, precision  - see
                flax.linen.attention.dot_product_attention_weights
        custom_attention_fn - if not None then this function will be used for attention
    """
    if dropout_rate > 0.0:
        raise ValueError("cross_batch_attention: We don't use dropout")

    if (
        len(query.shape) != 4
        or query.shape != key.shape
        or len(query.shape) != len(value.shape)
        or query.shape[:-1] != value.shape[:-1]
    ):
        raise ValueError(f"Queries, keys and values should match got qkv: {query.shape}, {key.shape}, {value.shape}")

    if cross_batch_range <= 0:
        raise ValueError("Cross-Batch should be at least 1")

    if dataset_packing <= 0:
        raise ValueError("Dataset packing should be positive")

    batch_size, seq_len, num_heads, _ = query.shape

    if batch_size % dataset_packing != 0:
        raise ValueError(f"Batch size ({batch_size}) should be divisible by dataset packing ({dataset_packing})")

    if bias.shape != (batch_size, 1, seq_len, seq_len):
        raise ValueError(f"Wrong bias shape got {bias.shape} expected {(batch_size, 1, seq_len, seq_len)}")

    query, key = promote_dtype(query, key, dtype=dtype)
    dtype = query.dtype

    # main cross-batch code starts here
    num_attentions = 1 + cross_batch_range  # local_context + cross_batch_range other contexts

    # keys from other contexts will be encoded as if they
    # were at the beginning of the local context
    pkey_fst = pos_encode_as_first(xk=key)

    # local context keys encoded in the standard way
    pquery, pkey = pos_encode(xq=query, xk=key)

    # otherwise this step will be performed by custom_attention_fn
    if custom_attention_fn is None:
        depth = pquery.shape[-1]
        pquery = pquery / jnp.sqrt(depth).astype(dtype)

    # for each element of the batch we calculate indices of
    # the batch that will be used in cross-batch
    cross_batch_rel_ids = jnp.arange(0, -num_attentions, -1).reshape(1, -1)
    batch_ids = jnp.arange(0, batch_size).reshape(-1, 1)
    cross_batch_selector = cross_batch_rel_ids + batch_ids

    # here we want other contexts
    cross_batch_keys = pkey_fst[cross_batch_selector[:, 1:]]

    # here we concatenate local context with other contexts
    attention_keys = jnp.concatenate([pkey[:, None], cross_batch_keys], axis=1)

    # otherwise this step will be performed by custom_attention_fn
    if custom_attention_fn is None:
        # attention keys is an array of shape [BATCH_SIZE, cross_batch_range + 1, SEQ_LEN, NUM_HEADS, HEAD_DIM]
        # attention_keys[:, 0] contains keys from the local context whereas
        # attention_keys[:, 1:] contains keys from other contexts
        # The einsum formula below can be written as
        # cb_attn_weights[b, h, q, c, k] = \sum_{d}pquery[b,q,h,d]*attention_keys[b,c,k,h,d]
        # In this form for c = 0,  one can see that the query attends to its local context
        # whereas for c != 0 to other contexts
        cb_attn_weights = jnp.einsum("bqhd,bckhd->bhqck", pquery, attention_keys, precision=precision)

        assert cb_attn_weights.shape == (batch_size, num_heads, seq_len, num_attentions, seq_len)

    # cross_batch_stepping allows to use multiple cross_batch_ranges in one batch
    cb_step_size = ceil_divide(num_attentions, max(dataset_packing - 1, 1))
    packing_mask = []
    for i in range(batch_size):
        if dataset_packing == 1 or not cross_batch_stepping:
            # full cross-batch
            pack_size = num_attentions
        else:
            # stepping cross-batch
            in_pack_id = i % dataset_packing
            pack_size = min(in_pack_id * cb_step_size + 1, num_attentions)

        # We don't want to look into the future with large k's to avoid info leak
        pack_size = min(pack_size, i + 1)
        assert pack_size > 0

        pos_mask = jnp.full((1, 1, 1, pack_size, 1), 0.0, dtype=bias.dtype)
        neg_mask = jnp.full(
            (1, 1, 1, num_attentions - pack_size, 1),
            min_value,
            dtype=bias.dtype,
        )

        packing_mask.append(jnp.concatenate([pos_mask, neg_mask], axis=-2))

    packing_mask = jnp.concatenate(packing_mask, axis=0)

    # otherwise this step will be performed by custom_attention_fn
    if custom_attention_fn is None:
        assert len(packing_mask.shape) == len(cb_attn_weights.shape)

        cb_attn_weights = cb_attn_weights + packing_mask
        cb_attn_weights = cb_attn_weights.at[:, :, :, 0, :].add(bias[:, :, :, :])

        cb_attn_weights = cb_attn_weights.reshape((batch_size, num_heads, seq_len, num_attentions * seq_len))

        cb_attn_weights = jax.nn.softmax(cb_attn_weights, axis=(-1)).astype(dtype)

        # apply attention dropout - not used
        if not deterministic and dropout_rate > 0.0:
            keep_prob = 1.0 - dropout_rate
            if broadcast_dropout:
                # dropout is broadcast across the batch + head dimensions
                dropout_shape = tuple([1] * (key.ndim - 2)) + cb_attn_weights.shape[-2:]
                keep = jax.random.bernoulli(dropout_rng, keep_prob, dropout_shape)
            else:
                keep = jax.random.bernoulli(dropout_rng, keep_prob, cb_attn_weights.shape)
            multiplier = keep.astype(dtype) / jnp.asarray(keep_prob, dtype=dtype)
            cb_attn_weights = cb_attn_weights * multiplier

        cb_attn_weights = cb_attn_weights.reshape(batch_size, num_heads, seq_len, num_attentions, seq_len)

        cb_values = value[cross_batch_selector]

        # cb_output[b, q, h, d] = \sum_{c}\sum{k}cb_attn_weights[b, h, q, c, k]*cb_values[b, c, k, h, d]
        cb_output = jnp.einsum("bhqck,bckhd->bqhd", cb_attn_weights, cb_values, precision=precision)
    else:
        cb_values = value[cross_batch_selector]

        cb_output = custom_attention_fn(
            query=pquery,  # [BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM]
            key=attention_keys.reshape(
                attention_keys.shape[0],
                attention_keys.shape[1] * attention_keys.shape[2],
                attention_keys.shape[3],
                attention_keys.shape[4],
            ),  # [BATCH_SIZE, (cross_batch_range + 1) * SEQ_LEN, NUM_HEADS, HEAD_DIM]
            value=cb_values.reshape(
                cb_values.shape[0], cb_values.shape[1] * cb_values.shape[2], cb_values.shape[3], cb_values.shape[4]
            ),  # [BATCH_SIZE, (cross_batch_range + 1) * SEQ_LEN, NUM_HEADS, HEAD_DIM]
            bias=jnp.broadcast_to(packing_mask, (batch_size, 1, seq_len, num_attentions, seq_len))
            .at[:, :, :, 0, :]
            .set(bias)
            .reshape(
                batch_size, 1, seq_len, num_attentions * seq_len
            ),  # [BATCH_SIZE, 1, SEQ_LEN, (cross_batch_range + 1) * SEQ_LEN]
            dropout_rng=dropout_rng,
            attn_pdrop=dropout_rate,
            causal=False,  # handled by bias
            dtype=dtype,
            precision=precision,
            deterministic=deterministic,
        )

    return cb_output
