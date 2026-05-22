import argparse

from training.checkpoint import load_checkpoint
from inference.generate import generate_answer, generate_text, format_prompt

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best.pkl",
        help="Path to checkpoint file.",
    )

    parser.add_argument(
        "--a",
        type=int,
        required=True,
        help="First number, 0 to 999.",
    )

    parser.add_argument(
        "--b",
        type=int,
        required=True,
        help="Second number, 0 to 999.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    params, _opt_state, config, step, metrics, _rng_state = load_checkpoint(
        args.checkpoint
    )

    prompt = format_prompt(args.a, args.b)

    answer = generate_answer(
        params=params,
        config=config,
        a=args.a,
        b=args.b,
    )

    full_text = generate_text(
        params=params,
        config=config,
        prompt=prompt,
        num_new_tokens=4,
    )

    expected = f"{args.a + args.b:04d}"

    print(f"checkpoint: {args.checkpoint}")
    print(f"checkpoint step: {step}")
    print(f"checkpoint metrics: {metrics}")
    print()
    print(f"prompt:    {prompt}")
    print(f"generated: {full_text}")
    print(f"answer:    {answer}")
    print(f"expected:  {expected}")
    print(f"correct:   {answer == expected}")

if __name__ == "__main__":
    main()