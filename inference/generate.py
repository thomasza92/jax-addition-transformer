import jax
import jax.numpy as jnp

from blocks.transformer import TransformerConfig, transformer_forward
from data.tokenizer import PAD_ID, decode, encode


def format_prompt(a: int, b: int) -> str:
  """
  Format an addition prompt without the answer.

  Example:
      123, 45 -> "123+045="
  """
  return f"{a:03d}+{b:03d}="


def generate_text(
    params: dict,
    config: TransformerConfig,
    prompt: str,
    num_new_tokens: int,
) -> str:
  """
  Greedy autoregressive generation.
  """
  ids = [
      token_id
      for token_id in encode(prompt, max_len=config.seq_len)
      if token_id != PAD_ID
  ]

  for _ in range(num_new_tokens):
    if len(ids) > config.seq_len:
      raise ValueError("Generated sequence exceeded model context length")

    padded = ids + [PAD_ID] * (config.seq_len - len(ids))

    x = jnp.array([padded], dtype=jnp.int32)

    logits = transformer_forward(
        params,
        x,
        config,
    )

    next_logits = logits[0, len(ids) - 1, :]

    next_id = int(jnp.argmax(next_logits))

    ids.append(next_id)

  return decode(ids)


def generate_answer(
    params: dict,
    config: TransformerConfig,
    a: int,
    b: int,
) -> str:
  """
  Generate only the 4-digit answer.
  """
  prompt = format_prompt(a, b)

  full_text = generate_text(
      params,
      config,
      prompt,
      num_new_tokens=4,
  )

  return full_text[len(prompt) :]


def exact_match_accuracy(
    params: dict,
    config: TransformerConfig,
    key: jax.Array,
    num_examples: int,
) -> float:
  """
  Evaluate exact-match accuracy over random examples.
  """
  key, a_key, b_key = jax.random.split(key, 3)

  a_values = jax.random.randint(
      a_key,
      shape=(num_examples,),
      minval=0,
      maxval=1000,
  )

  b_values = jax.random.randint(
      b_key,
      shape=(num_examples,),
      minval=0,
      maxval=1000,
  )

  correct = 0

  for a_raw, b_raw in zip(a_values, b_values, strict=True):
    a = int(a_raw)
    b = int(b_raw)

    pred = generate_answer(params, config, a, b)
    expected = f"{a + b:04d}"

    correct += int(pred == expected)

  return correct / num_examples
