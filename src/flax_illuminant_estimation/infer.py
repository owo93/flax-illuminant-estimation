import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from absl import flags
from flax import nnx
from PIL import Image, ImageDraw
from rich.pretty import pprint

from flax_illuminant_estimation.checkpoint import latest, list_checkpoints, load
from flax_illuminant_estimation.config import Config
from flax_illuminant_estimation.model import ViT

FLAGS = flags.FLAGS


def estimate_illuminant(model, image_path, img_size):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
    img = jnp.array(img, dtype=jnp.float32) / 255.0

    img_batch = jnp.expand_dims(img, axis=0)
    pred = model(img_batch, train=False)
    return pred[0]


def show(image, pred, size=224):
    img = Image.open(image).convert("RGB")
    r, g, b = float(pred[0]), float(pred[1]), float(pred[2])

    w, h = img.width // 4, img.height // 4
    img_arr = jnp.array(img, dtype=jnp.float32) / 255.0

    illum = jnp.array([r, g, b], dtype=jnp.float32)
    corrected = img_arr / (illum + 1e-8)
    corrected = corrected / (corrected[..., 1:2].max() + 1e-8)
    corrected = jnp.clip(corrected, 0.0, 1.0)

    corrected_img = Image.fromarray((np.array(corrected) * 255.0).astype("uint8"))

    canvas = Image.new("RGB", (img.width * 2, img.height * 2))
    swatch = Image.new("RGB", (w, h), (int(r * 255), int(g * 255), int(b * 255)))
    canvas.paste(img, (0, 0))
    canvas.paste(swatch, (0, img.height))
    canvas.paste(corrected_img, (img.width, 0))

    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), "Input", fill="white")
    draw.text((img.width + 10, 10), "corrected image", fill="white")

    info = [
        f"Chromaticity: ({r:.8f}, {g:.8f}, {b:.8f})",
        f"RGB:          ({r * 255:.0f}, {g * 255:.0f}, {b * 255:.0f})",
    ]
    for i, line in enumerate(info):
        draw.text((img.width + 10, img.height + 10 + i * 20), line, fill="white")

    canvas.show()
    return canvas


def main():
    if FLAGS.config:
        config = Config.from_yaml(FLAGS.config)
    else:
        config = Config()

    checkpoint_path = Path(FLAGS.checkpoint) if FLAGS.checkpoint else None

    if checkpoint_path and not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        print("Available checkpoints:")
        for p in list_checkpoints(config.trainer.checkpoint_dir):
            print(f"  {p}")
        sys.exit(1)

    if checkpoint_path is None:
        checkpoint_path = latest(config.trainer.checkpoint_dir)
        if checkpoint_path is None:
            print(f"Error: No checkpoints found in {config.trainer.checkpoint_dir}")
            sys.exit(1)
        print(f"Using latest checkpoint: {checkpoint_path}")

    state = load(checkpoint_path)

    if state.config:
        pprint(state.config, expand_all=True, indent_guides=False)

    model = ViT(
        img_size=state.config["model"]["img_size"],
        patch_size=state.config["model"]["patch_size"],
        dim=state.config["model"]["dim"],
        depth=state.config["model"]["depth"],
        num_heads=state.config["model"]["num_heads"],
        rngs=nnx.Rngs(0),
    )
    nnx.update(model, state.model_state)

    print(f"\nRestored from checkpoint at epoch {state.epoch}")

    print(f"\nEstimating illuminant for: {FLAGS.image}")
    pred = estimate_illuminant(model, FLAGS.image, config.model.img_size)

    show(FLAGS.image, pred)


if __name__ == "__main__":
    main()
