from functools import partial

import jax

from blocks.transformer import transformer_forward
from config import TransformerConfig
from training.loss import cross_entropy_loss
from training.optim import adamw_update


def model_loss(
    params: dict,
    batch: dict,
    config: TransformerConfig,
):
  """
  Compute model loss for one batch.
  """
  logits = transformer_forward(
      params,
      batch["x"],
      config,
  )

  loss = cross_entropy_loss(
      logits=logits,
      targets=batch["y"],
      mask=batch["mask"],
  )

  return loss


@partial(
    jax.jit,
    static_argnames=("config",),
    donate_argnums=(0, 1),
)
def train_step(
    params: dict,
    opt_state: dict,
    batch: dict,
    config: TransformerConfig,
    learning_rate: float,
):
  """
  One jitted training step.

  donate_argnums=(0, 1) lets JAX reuse buffers for params and opt_state.
  Do not use old params/opt_state after calling this function.
  """
  loss, grads = jax.value_and_grad(model_loss)(
      params,
      batch,
      config,
  )

  params, opt_state = adamw_update(
      params,
      grads,
      opt_state,
      learning_rate=learning_rate,
  )

  return params, opt_state, loss
