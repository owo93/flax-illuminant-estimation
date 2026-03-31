import sys
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx
from PIL import Image

from flax_illuminant_estimation.checkpoint import latest, load
from flax_illuminant_estimation.checkpoint import list as list_checkpoints
from flax_illuminant_estimation.config import ModelConfig
from flax_illuminant_estimation.model import ViT


def preprocess_image(path, size=224):
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    img = jnp.array(img, dtype=jnp.float32) / 255.0
    return img


def estimate_illuminant(model, image_path, rngs=None):
    img = preprocess_image(image_path)
    img_batch = jnp.expand_dims(img, axis=0)
    pred = model(img_batch, train=False, rngs=rngs or nnx.Rngs())
    return pred[0]


def main(args):
    if args.config:
        config = ModelConfig.from_yaml(args.config)
        print(f"Loaded config from {args.config}")
    else:
        config = ModelConfig()

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None

    if checkpoint_path and not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        print("Available checkpoints:")
        for p in list_checkpoints(config.checkpoint_dir):
            print(f"  {p}")
        sys.exit(1)

    if checkpoint_path is None:
        checkpoint_path = latest(config.checkpoint_dir)
        if checkpoint_path is None:
            print(f"Error: No checkpoints found in {config.checkpoint_dir}")
            sys.exit(1)
        print(f"Using latest checkpoint: {checkpoint_path}")

    state = load(checkpoint_path)

    if state.config:
        print(f"Checkpoint config: {state.config}")

    model = ViT(
        img_size=config.img_size,
        patch_size=config.patch_size,
        dim=config.dim,
        depth=config.depth,
        num_heads=config.num_heads,
        rngs=nnx.Rngs(0),
    )
    nnx.update(model, state.model_state)

    print(f"Loaded model from epoch {state.epoch}")

    print(f"\nEstimating illuminant for: {args.image}")
    rngs = nnx.Rngs(dropout=jax.random.key(0))
    pred = estimate_illuminant(model, args.image, rngs)

    r, g, b = float(pred[0]), float(pred[1]), float(pred[2])
    print(f"Chromaticity: {r:.4f}, {g:.4f}, {b:.4f}")
    print(f"RGB: rgb({r * 255:.0f}, {g * 255:.0f}, {b * 255:.0f})")


if __name__ == "__main__":
    main(sys.argv[1:])
