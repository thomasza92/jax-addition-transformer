# JAX Addition Transformer

A decoder-only Transformer implemented in JAX that learns fixed-width addition over a small character vocabulary.

The model is trained on examples of the form:

```text
123+045=0168
007+005=0012
999+999=1998
```

At inference time, the model receives a prompt such as:

```text
123+045=
```

and generates a four-digit answer.

## Inspiration

Created for learning purposes, inspired by [this blog post](https://vladfeinberg.com/2026/05/10/how-to-land-a-job-at-a-frontier-lab.html).

## Features

- Character-level tokenizer
- JAX-native fixed-width addition dataset generator
- Decoder-only Transformer
- Multi-head causal self-attention
- LayerNorm, MLP blocks, residual connections
- AdamW optimizer
- JIT-compiled training step
- Checkpoint saving and loading
- Command-line inference
- Gradio frontend

## Project Structure

```text
.
├── main.py              # Training entry point
├── infer.py             # Command-line inference
├── app.py               # Gradio frontend
├── config.py            # Model configuration
├── blocks/              # Transformer building blocks
├── data/                # Tokenizer and dataset generation
├── training/            # Loss, optimizer, train step, checkpoints
├── inference/           # Generation utilities
└── checkpoints/         # Saved checkpoints
```

## Setup

```bash
uv sync
```

## Training

Train the debug model:

```bash
uv run python main.py
```

Train the larger approximately 10M parameter model:

```bash
uv run python main.py --final --steps 10000 --batch-size 128
```

Common options:

```bash
uv run python main.py --steps 5000
uv run python main.py --batch-size 256
uv run python main.py --lr 3e-4
uv run python main.py --eval-every 500
uv run python main.py --save-every 1000
```

Resume from a checkpoint:

```bash
uv run python main.py --resume checkpoints/latest.pkl --steps 10000
```

## Checkpoints

Training writes checkpoints to:

```text
checkpoints/latest.pkl
checkpoints/best.pkl
```

Use `latest.pkl` to resume training and `best.pkl` for inference.

## Command-Line Inference

```bash
uv run python infer.py --a 123 --b 45
```

Example output:

```text
prompt:    123+045=
generated: 123+045=0168
answer:    0168
expected:  0168
correct:   True
```

Use a specific checkpoint:

```bash
uv run python infer.py --checkpoint checkpoints/latest.pkl --a 999 --b 999
```

Inputs must be integers from `0` to `999`.

## Frontend

Run the Gradio app:

```bash
uv run python app.py
```

Open the local URL printed in the terminal, usually:

```text
http://127.0.0.1:7860
```

The app loads `checkpoints/best.pkl` by default.

## Notes

This project is intended as a learning exercise. It is not a practical calculator. The objective is to implement and train the core components of a small Transformer in JAX.