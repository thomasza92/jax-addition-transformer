from functools import partial

import jax
import jax.numpy as jnp

from data.tokenizer import encode, PAD_ID, stoi

DEFAULT_MAX_LEN = 16

PLUS_ID = stoi["+"]
EQUAL_ID = stoi["="]

def format_addition_example(a: int, b: int) -> str:
    """
    Format one fixed-width addition example.

    Example:
        a = 7, b = 5
        returns "007+005=0012"
    """
    if not 0 <= a <= 999:
        raise ValueError(f"a must be between 0 and 999, got {a}")

    if not 0 <= b <= 999:
        raise ValueError(f"b must be between 0 and 999, got {b}")

    return f"{a:03d}+{b:03d}={a + b:04d}"

def make_example(
    a: int,
    b: int,
    max_len: int = DEFAULT_MAX_LEN,
    answer_only_loss: bool = True,
) -> dict:
    """
    Python/debug version for one example.

    This is useful for inspection and tests, but the training batch
    generator below is JAX-native.
    """
    text = format_addition_example(a, b)

    tokens = encode(text, max_len=max_len)

    x = tokens[:-1]
    y = tokens[1:]

    if answer_only_loss:
        eq_pos = text.index("=")

        mask = [
            1.0 if token_position > eq_pos and target_id != PAD_ID else 0.0
            for token_position, target_id in enumerate(y, start=1)
        ]
    else:
        mask = [
            1.0 if target_id != PAD_ID else 0.0
            for target_id in y
        ]

    return {
        "text": text,
        "tokens": tokens,
        "x": x,
        "y": y,
        "mask": mask,
    }

def _three_digit_tokens(values: jax.Array) -> jax.Array:
    """
    Convert integers from 0 to 999 into fixed-width digit token IDs.

    Example:
        7   -> [0, 0, 7]
        45  -> [0, 4, 5]
        123 -> [1, 2, 3]

    Since digit token IDs are the same as their digit values,
    the output can be used directly as token IDs.
    """
    hundreds = values // 100
    tens = (values // 10) % 10
    ones = values % 10

    return jnp.stack(
        [hundreds, tens, ones],
        axis=1,
    ).astype(jnp.int32)

def _four_digit_tokens(values: jax.Array) -> jax.Array:
    """
    Convert integers from 0 to 1998 into fixed-width digit token IDs.

    Example:
        12   -> [0, 0, 1, 2]
        168  -> [0, 1, 6, 8]
        1998 -> [1, 9, 9, 8]

    Since digit token IDs are the same as their digit values,
    the output can be used directly as token IDs.
    """
    thousands = values // 1000
    hundreds = (values // 100) % 10
    tens = (values // 10) % 10
    ones = values % 10

    return jnp.stack(
        [thousands, hundreds, tens, ones],
        axis=1,
    ).astype(jnp.int32)


@partial(
    jax.jit,
    static_argnames=("batch_size", "max_len", "answer_only_loss"),
)
def make_batch(
    key: jax.Array,
    batch_size: int,
    max_len: int = DEFAULT_MAX_LEN,
    answer_only_loss: bool = True,
) -> tuple[jax.Array, dict]:
    """
    Create a random fixed-width addition batch using JAX only.

    This generates examples equivalent to strings like:

        123+045=0168

    But it directly creates token IDs instead of formatting strings.

    Returns:
        new_key
        batch dict with:
            x:    shape (batch_size, max_len - 1)
            y:    shape (batch_size, max_len - 1)
            mask: shape (batch_size, max_len - 1)
    """
    if max_len < 12:
        raise ValueError(
            f"max_len must be at least 12 for AAA+BBB=CCCC, got {max_len}"
        )

    key, a_key, b_key = jax.random.split(key, 3)

    a_values = jax.random.randint(
        a_key,
        shape=(batch_size,),
        minval=0,
        maxval=1000,
        dtype=jnp.int32,
    )

    b_values = jax.random.randint(
        b_key,
        shape=(batch_size,),
        minval=0,
        maxval=1000,
        dtype=jnp.int32,
    )

    answer_values = a_values + b_values

    a_tokens = _three_digit_tokens(a_values)
    b_tokens = _three_digit_tokens(b_values)
    answer_tokens = _four_digit_tokens(answer_values)

    plus_col = jnp.full(
        (batch_size, 1),
        PLUS_ID,
        dtype=jnp.int32,
    )

    equal_col = jnp.full(
        (batch_size, 1),
        EQUAL_ID,
        dtype=jnp.int32,
    )

    pad_cols = jnp.full(
        (batch_size, max_len - 12),
        PAD_ID,
        dtype=jnp.int32,
    )

    tokens = jnp.concatenate(
        [
            a_tokens,
            plus_col,
            b_tokens,
            equal_col,
            answer_tokens,
            pad_cols,
        ],
        axis=1,
    )

    x = tokens[:, :-1]
    y = tokens[:, 1:]

    if answer_only_loss:
        target_positions = jnp.arange(1, max_len)

        position_mask = (
            (target_positions >= 8)
            & (target_positions <= 11)
        )

        mask = jnp.broadcast_to(
            position_mask[None, :],
            y.shape,
        )
    else:
        mask = y != PAD_ID

    mask = mask.astype(jnp.float32)

    batch = {
        "x": x,
        "y": y,
        "mask": mask,
    }

    return key, batch