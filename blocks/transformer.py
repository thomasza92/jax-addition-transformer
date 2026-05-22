import jax
import jax.numpy as jnp

from blocks.attention import (
    init_multi_head_attention,
    multi_head_causal_self_attention,
)
from blocks.layers import (
    init_layer_norm,
    init_linear,
    init_mlp,
    layer_norm,
    linear,
    mlp,
)
from config import TransformerConfig


def init_transformer_block(
    key: jax.Array,
    config: TransformerConfig,
) -> dict:
  """
  Initialize one decoder Transformer block.
  """
  attn_key, mlp_key = jax.random.split(key)

  return {
      "ln1": init_layer_norm(config.d_model),
      "attn": init_multi_head_attention(attn_key, config.d_model),
      "ln2": init_layer_norm(config.d_model),
      "mlp": init_mlp(mlp_key, config.d_model, config.d_ff),
  }


def transformer_block(
    params: dict,
    x: jax.Array,
    config: TransformerConfig,
) -> jax.Array:
  """
  Apply one pre-norm decoder Transformer block.

  Structure:

      x = x + attention(layer_norm(x))
      x = x + mlp(layer_norm(x))
  """
  x = x + multi_head_causal_self_attention(
      params["attn"],
      layer_norm(params["ln1"], x),
      config.n_heads,
  )

  x = x + mlp(
      params["mlp"],
      layer_norm(params["ln2"], x),
  )

  return x


def init_transformer_model(
    key: jax.Array,
    config: TransformerConfig,
) -> dict:
  """
  Initialize the full decoder-only Transformer.
  """
  keys = jax.random.split(key, config.n_layers + 3)

  token_key = keys[0]
  pos_key = keys[1]
  output_key = keys[2]
  block_keys = keys[3:]

  token_embedding = (
      jax.random.normal(
          token_key,
          shape=(config.vocab_size, config.d_model),
      )
      * 0.02
  )

  position_embedding = (
      jax.random.normal(
          pos_key,
          shape=(config.seq_len, config.d_model),
      )
      * 0.02
  )

  blocks = [
      init_transformer_block(block_key, config) for block_key in block_keys
  ]

  final_ln = init_layer_norm(config.d_model)

  output = init_linear(
      output_key,
      in_dim=config.d_model,
      out_dim=config.vocab_size,
  )

  return {
      "token_embedding": token_embedding,
      "position_embedding": position_embedding,
      "blocks": blocks,
      "final_ln": final_ln,
      "output": output,
  }


def transformer_forward(
    params: dict,
    x: jax.Array,
    config: TransformerConfig,
) -> jax.Array:
  """
  Forward pass.

  Args:
      x:
          Token IDs with shape:

              (batch_size, seq_len)

  Returns:
      logits with shape:

              (batch_size, seq_len, vocab_size)
  """
  batch_size, seq_len = x.shape

  if seq_len > config.seq_len:
    raise ValueError(
        f"Input seq_len={seq_len} exceeds config.seq_len={config.seq_len}"
    )

  token_emb = params["token_embedding"][x]

  positions = jnp.arange(seq_len)
  pos_emb = params["position_embedding"][positions]

  hidden = token_emb + pos_emb

  for block in params["blocks"]:
    hidden = transformer_block(block, hidden, config)

  hidden = layer_norm(params["final_ln"], hidden)

  logits = linear(params["output"], hidden)

  return logits


def count_parameters(params: dict) -> int:
  """
  Count trainable parameters.
  """
  leaves = jax.tree_util.tree_leaves(params)

  return int(sum(leaf.size for leaf in leaves))
