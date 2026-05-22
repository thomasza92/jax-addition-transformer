import jax.numpy as jnp
import jax.nn as jnn

def cross_entropy_loss(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    mask: jnp.ndarray,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """
    Compute masked cross-entropy loss.

    Args:
        logits:
            Shape (batch_size, seq_len, vocab_size)

        targets:
            Shape (batch_size, seq_len)

        mask:
            Shape (batch_size, seq_len)
            1.0 means include this target token in the loss.
            0.0 means ignore it.

    Returns:
        Scalar loss.
    """
    log_probs = jnn.log_softmax(logits, axis=-1)

    target_log_probs = jnp.take_along_axis(
        log_probs,
        targets[..., None],
        axis=-1,
    )

    target_log_probs = jnp.squeeze(target_log_probs, axis=-1)

    per_token_loss = -target_log_probs

    masked_loss = per_token_loss * mask

    loss = jnp.sum(masked_loss) / (jnp.sum(mask) + eps)

    return loss