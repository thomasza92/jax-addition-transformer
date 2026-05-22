import gradio as gr

from inference.generate import generate_answer
from training.checkpoint import load_checkpoint

CHECKPOINT_PATH = "checkpoints/best.pkl"


def load_model():
  params, _opt_state, config, step, metrics, _rng_state = load_checkpoint(
      CHECKPOINT_PATH
  )

  return {
      "params": params,
      "config": config,
      "step": step,
      "metrics": metrics,
  }


MODEL = load_model()


def predict_addition(a, b):
  """
  Gradio passes numeric inputs as floats, so convert them to ints.
  """
  try:
    a = int(a)
    b = int(b)
  except Exception:
    return "Invalid input"

  if not 0 <= a <= 999:
    return "Invalid input"

  if not 0 <= b <= 999:
    return "Invalid input"

  predicted = generate_answer(
      params=MODEL["params"],
      config=MODEL["config"],
      a=a,
      b=b,
  )

  return str(int(predicted))


with gr.Blocks(title="JAX Addition Transformer") as demo:
  gr.Markdown("# JAX Addition Transformer")
  gr.Markdown(
      "Enter two numbers from 0 to 999. "
      "The model sees a fixed-width prompt like `123+045=` "
      "and generates a 4-digit answer."
  )

  with gr.Row():
    a_input = gr.Number(
        label="First number",
        value=123,
        precision=0,
        minimum=0,
        maximum=999,
    )

    b_input = gr.Number(
        label="Second number",
        value=45,
        precision=0,
        minimum=0,
        maximum=999,
    )

  run_button = gr.Button("Run model")

  output = gr.Textbox(
      label="Result",
      lines=5,
  )

  gr.Markdown(
      f"Loaded checkpoint: `{CHECKPOINT_PATH}`  \n"
      f"Checkpoint step: `{MODEL['step']}`  \n"
      f"Checkpoint metrics: `{MODEL['metrics']}`"
  )

  run_button.click(
      fn=predict_addition,
      inputs=[a_input, b_input],
      outputs=output,
  )


if __name__ == "__main__":
  demo.launch()
