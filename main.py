from pathlib import Path
import argparse

import jax
from tqdm import tqdm

from training.checkpoint import save_checkpoint, load_checkpoint
from data.addition import make_batch
from inference.generate import exact_match_accuracy, generate_answer
from blocks.transformer import (
    debug_config,
    final_10m_config,
    init_transformer_model,
    count_parameters,
)
from training.optim import init_adamw_state
from training.train_step import train_step

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--final",
        action="store_true",
        help="Use the ~10M parameter model config.",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=2_000,
        help="Number of training steps.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Training batch size.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate.",
    )

    parser.add_argument(
        "--eval-every",
        type=int,
        default=200,
        help="Run exact-match eval every N steps.",
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Update tqdm loss display every N steps.",
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=1000,
        help="Save checkpoints every N steps.",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for checkpoints.",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from.",
    )

    return parser.parse_args()

def print_examples(params, config):
    for a, b in [
        (123, 45),
        (7, 5),
        (999, 999),
    ]:
        pred = generate_answer(params, config, a, b)
        expected = f"{a + b:04d}"

        tqdm.write(
            f"  {a:03d}+{b:03d}= "
            f"predicted {pred} "
            f"expected {expected}"
        )

def main():
    args = parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    latest_path = checkpoint_dir / "latest.pkl"
    best_path = checkpoint_dir / "best.pkl"

    if args.resume is not None:
        (
            params,
            opt_state,
            config,
            start_step,
            metrics,
            rng_state,
        ) = load_checkpoint(args.resume)

        data_key = rng_state["data_key"]
        eval_key = rng_state["eval_key"]

        best_acc = float(metrics.get("best_acc", 0.0))

        print(f"Resumed from {args.resume}")
        print(f"Starting at step {start_step}")
    else:
        config = final_10m_config() if args.final else debug_config()

        key = jax.random.PRNGKey(0)
        key, model_key, data_key, eval_key = jax.random.split(key, 4)

        params = init_transformer_model(model_key, config)
        opt_state = init_adamw_state(params)

        start_step = 0
        best_acc = 0.0

    max_len = config.seq_len + 1

    print("config:", config)
    print("parameters:", count_parameters(params))
    print("device:", jax.default_backend())
    print()

    progress = tqdm(
        range(start_step + 1, args.steps + 1),
        desc="training",
        dynamic_ncols=True,
    )

    last_loss_value = None

    for step in progress:
        data_key, batch = make_batch(
            data_key,
            batch_size=args.batch_size,
            max_len=max_len,
            answer_only_loss=True,
        )

        params, opt_state, loss = train_step(
            params,
            opt_state,
            batch,
            config,
            args.lr,
        )

        if step % args.log_every == 0:
            loss = loss.block_until_ready()
            last_loss_value = float(loss)

            progress.set_postfix(
                loss=f"{last_loss_value:.4f}",
                best=f"{best_acc:.3f}",
            )

        should_eval = (
            step % args.eval_every == 0
            or step == args.steps
        )

        if should_eval:
            loss = loss.block_until_ready()
            last_loss_value = float(loss)

            eval_key, subkey = jax.random.split(eval_key)

            acc = exact_match_accuracy(
                params,
                config,
                subkey,
                num_examples=100,
            )

            tqdm.write(
                f"\nstep {step:5d} | "
                f"loss {last_loss_value:.4f} | "
                f"exact {acc:.3f} | "
                f"best {best_acc:.3f}"
            )

            print_examples(params, config)

            should_save = (
                step % args.save_every == 0
                or step == args.steps
            )

            if should_save:
                rng_state = {
                    "data_key": data_key,
                    "eval_key": eval_key,
                }

                latest_metrics = {
                    "loss": last_loss_value,
                    "accuracy": float(acc),
                    "best_acc": float(best_acc),
                }

                save_checkpoint(
                    latest_path,
                    params=params,
                    opt_state=opt_state,
                    config=config,
                    step=step,
                    metrics=latest_metrics,
                    rng_state=rng_state,
                )

                tqdm.write(f"saved latest checkpoint: {latest_path}")

                if acc >= best_acc:
                    best_acc = float(acc)

                    best_metrics = {
                        "loss": last_loss_value,
                        "accuracy": float(acc),
                        "best_acc": float(best_acc),
                    }

                    save_checkpoint(
                        best_path,
                        params=params,
                        opt_state=opt_state,
                        config=config,
                        step=step,
                        metrics=best_metrics,
                        rng_state=rng_state,
                    )

                    tqdm.write(f"saved new best checkpoint: {best_path}")

                tqdm.write("")


if __name__ == "__main__":
    main()