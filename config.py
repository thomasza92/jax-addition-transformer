from dataclasses import dataclass

from data.tokenizer import VOCAB_SIZE


@dataclass(frozen=True)
class TransformerConfig:
  vocab_size: int
  seq_len: int
  d_model: int
  n_heads: int
  d_ff: int
  n_layers: int


def debug_config() -> TransformerConfig:
  """
  Small config for debugging.
  """
  return TransformerConfig(
      vocab_size=VOCAB_SIZE,
      seq_len=15,
      d_model=64,
      n_heads=4,
      d_ff=256,
      n_layers=2,
  )


def final_10m_config() -> TransformerConfig:
  """
  Approximate 10M parameter config.
  """
  return TransformerConfig(
      vocab_size=VOCAB_SIZE,
      seq_len=15,
      d_model=256,
      n_heads=8,
      d_ff=1024,
      n_layers=12,
  )
