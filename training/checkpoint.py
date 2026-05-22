from dataclasses import asdict
from pathlib import Path
import pickle
from typing import Any

import jax

from blocks.transformer import TransformerConfig

def save_checkpoint(
    path: str | Path,
    *,
    params: dict,
    opt_state: dict,
    config: TransformerConfig,
    step: int,
    metrics: dict[str, Any],
    rng_state: dict[str, Any],
) -> None:
    """
    Save model state to disk.

    Uses jax.device_get so arrays are moved from device to host
    before pickling.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "params": jax.device_get(params),
        "opt_state": jax.device_get(opt_state),
        "config": asdict(config),
        "step": int(step),
        "metrics": metrics,
        "rng_state": jax.device_get(rng_state),
    }

    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with open(tmp_path, "wb") as f:
        pickle.dump(payload, f)

    tmp_path.replace(path)

def load_checkpoint(path: str | Path):
    """
    Load a checkpoint from disk.

    Returns:
        params
        opt_state
        config
        step
        metrics
        rng_state
    """
    path = Path(path)

    with open(path, "rb") as f:
        payload = pickle.load(f)

    config = TransformerConfig(**payload["config"])

    return (
        payload["params"],
        payload["opt_state"],
        config,
        payload["step"],
        payload["metrics"],
        payload["rng_state"],
    )