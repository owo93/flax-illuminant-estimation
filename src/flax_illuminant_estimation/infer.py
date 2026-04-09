import sys
from pathlib import Path

import jax.numpy as jnp
from absl import flags
from flax import nnx
from PIL import Image
from rich.pretty import pprint

from flax_illuminant_estimation.checkpoint import latest, list_checkpoints, load
from flax_illuminant_estimation.config import Config
from flax_illuminant_estimation.model import ViT

FLAGS = flags.FLAGS


def preprocess_image(path):
    img = Image.open(path).convert("RGB")
    img = jnp.array(img, dtype=jnp.float32) / 255.0
    return img


def estimate_illuminant(model, image_path):
    img = preprocess_image(image_path)
    img_batch = jnp.expand_dims(img, axis=0)
    pred = model(img_batch, train=False)
    return pred[0]


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
    pred = estimate_illuminant(model, FLAGS.image)

    r, g, b = float(pred[0]), float(pred[1]), float(pred[2])
    print(f"Chromaticity: {r:.8f}, {g:.8f}, {b:.8f}")
    print(f"RGB: rgb({r * 255:.0f}, {g * 255:.0f}, {b * 255:.0f})")


if __name__ == "__main__":
    main()
