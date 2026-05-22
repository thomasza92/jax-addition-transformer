import jax
import jax.numpy as jnp

def init_adamw_state(params: dict) -> dict:
    """
    Initialize AdamW optimizer state.
    """
    return {
        "step": jnp.array(0, dtype=jnp.int32),
        "m": jax.tree_util.tree_map(jnp.zeros_like, params),
        "v": jax.tree_util.tree_map(jnp.zeros_like, params),
    }

def adamw_update(
    params: dict,
    grads: dict,
    state: dict,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
) -> tuple[dict, dict]:
    """
    Apply one AdamW update.
    """
    step = state["step"] + 1

    m = jax.tree_util.tree_map(
        lambda old_m, g: beta1 * old_m + (1.0 - beta1) * g,
        state["m"],
        grads,
    )

    v = jax.tree_util.tree_map(
        lambda old_v, g: beta2 * old_v + (1.0 - beta2) * (g * g),
        state["v"],
        grads,
    )

    bias_correction1 = 1.0 - beta1 ** step
    bias_correction2 = 1.0 - beta2 ** step

    m_hat = jax.tree_util.tree_map(
        lambda x: x / bias_correction1,
        m,
    )

    v_hat = jax.tree_util.tree_map(
        lambda x: x / bias_correction2,
        v,
    )

    new_params = jax.tree_util.tree_map(
        lambda p, mh, vh: p - learning_rate * (
            mh / (jnp.sqrt(vh) + eps) + weight_decay * p
        ),
        params,
        m_hat,
        v_hat,
    )

    new_state = {
        "step": step,
        "m": m,
        "v": v,
    }

    return new_params, new_state