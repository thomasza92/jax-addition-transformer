import math

import jax
import jax.numpy as jnp
import jax.nn as jnn


def init_linear(
    key: jax.Array,
    in_dim: int,
    out_dim: int,
) -> dict:
    """
    Initialize a linear layer:

        y = x @ W + b
    """
    limit = math.sqrt(6.0 / (in_dim + out_dim))

    W = jax.random.uniform(
        key,
        shape=(in_dim, out_dim),
        minval=-limit,
        maxval=limit,
    )

    b = jnp.zeros((out_dim,))

    return {
        "W": W,
        "b": b,
    }


def linear(params: dict, x: jax.Array) -> jax.Array:
    """
    Apply a linear layer.

    x shape:
        (..., in_dim)

    output shape:
        (..., out_dim)
    """
    return x @ params["W"] + params["b"]


def init_layer_norm(d_model: int) -> dict:
    """
    Initialize LayerNorm parameters.
    """
    return {
        "gamma": jnp.ones((d_model,)),
        "beta": jnp.zeros((d_model,)),
    }


def layer_norm(
    params: dict,
    x: jax.Array,
    eps: float = 1e-5,
) -> jax.Array:
    """
    Apply LayerNorm over the final dimension.
    """
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)

    normalized = (x - mean) / jnp.sqrt(variance + eps)

    return params["gamma"] * normalized + params["beta"]


def init_mlp(
    key: jax.Array,
    d_model: int,
    d_ff: int,
) -> dict:
    """
    Initialize the Transformer feed-forward MLP.

    Shape:
        d_model -> d_ff -> d_model
    """
    key1, key2 = jax.random.split(key)

    return {
        "fc1": init_linear(key1, d_model, d_ff),
        "fc2": init_linear(key2, d_ff, d_model),
    }


def mlp(params: dict, x: jax.Array) -> jax.Array:
    """
    Apply the Transformer MLP.
    """
    x = linear(params["fc1"], x)
    x = jnn.gelu(x)
    x = linear(params["fc2"], x)

    return x