CHARS = "0123456789+= "
PAD = "<PAD>"

itos = list(CHARS) + [PAD]
stoi = {ch: i for i, ch in enumerate(itos)}

VOCAB_SIZE = len(itos)
PAD_ID = stoi[PAD]


def encode(s: str, max_len: int) -> list[int]:
  """
  Convert a string into token IDs and pad to max_len.
  """
  ids = [stoi[ch] for ch in s]

  if len(ids) > max_len:
    raise ValueError(
        f"String too long: {s!r}; length={len(ids)}, max_len={max_len}"
    )

  ids = ids + [PAD_ID] * (max_len - len(ids))
  return ids


def decode(ids: list[int]) -> str:
  """
  Convert token IDs back into a string.
  Ignores PAD tokens.
  """
  chars = []

  for i in ids:
    ch = itos[int(i)]

    if ch != PAD:
      chars.append(ch)

  return "".join(chars)
