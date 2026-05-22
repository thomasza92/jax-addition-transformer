import math

import jax
import jax.numpy as jnp
import jax.nn as jnn

from blocks.layers import init_linear, linear


def init_multi_head_attention(
    key: jax.Array,
    d_model: int,
) -> dict:
    """
    Initialize multi-head causal self-attention.

    Uses four projections:

        q_proj
        k_proj
        v_proj
        out_proj
    """
    q_key, k_key, v_key, out_key = jax.random.split(key, 4)

    return {
        "q_proj": init_linear(q_key, d_model, d_model),
        "k_proj": init_linear(k_key, d_model, d_model),
        "v_proj": init_linear(v_key, d_model, d_model),
        "out_proj": init_linear(out_key, d_model, d_model),
    }


def _split_heads(x: jax.Array, n_heads: int) -> jax.Array:
    """
    Convert:

        (batch, seq_len, d_model)

    into:

        (batch, n_heads, seq_len, head_dim)
    """
    batch_size, seq_len, d_model = x.shape

    head_dim = d_model // n_heads

    x = x.reshape(batch_size, seq_len, n_heads, head_dim)

    return jnp.transpose(x, (0, 2, 1, 3))


def _merge_heads(x: jax.Array) -> jax.Array:
    """
    Convert:

        (batch, n_heads, seq_len, head_dim)

    back into:

        (batch, seq_len, d_model)
    """
    batch_size, n_heads, seq_len, head_dim = x.shape

    x = jnp.transpose(x, (0, 2, 1, 3))

    return x.reshape(batch_size, seq_len, n_heads * head_dim)


def multi_head_causal_self_attention(
    params: dict,
    x: jax.Array,
    n_heads: int,
) -> jax.Array:
    """
    Apply multi-head causal self-attention.

    Input:
        x shape = (batch, seq_len, d_model)

    Output:
        shape = (batch, seq_len, d_model)
    """
    batch_size, seq_len, d_model = x.shape

    if d_model % n_heads != 0:
        raise ValueError(
            f"d_model={d_model} must be divisible by n_heads={n_heads}"
        )

    head_dim = d_model // n_heads

    q = linear(params["q_proj"], x)
    k = linear(params["k_proj"], x)
    v = linear(params["v_proj"], x)

    q = _split_heads(q, n_heads)
    k = _split_heads(k, n_heads)
    v = _split_heads(v, n_heads)

    scores = jnp.einsum("bhtd,bhsd->bhts", q, k)

    scores = scores / math.sqrt(head_dim)

    causal_mask = jnp.tril(
        jnp.ones((seq_len, seq_len), dtype=bool)
    )

    scores = jnp.where(
        causal_mask[None, None, :, :],
        scores,
        -1e9,
    )

    weights = jnn.softmax(scores, axis=-1)

    out = jnp.einsum("bhts,bhsd->bhtd", weights, v)

    out = _merge_heads(out)

    out = linear(params["out_proj"], out)

    return out