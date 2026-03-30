import sys
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx
from flax.training import checkpoints
from PIL import Image

from flax_illuminant_estimation.config import ModelConfig
from flax_illuminant_estimation.model import ViT


def preprocess_image(path, size=224):
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.Resampling.BILINEAR)
    img = jnp.array(img, dtype=jnp.float32) / 255.0
    return img


def estimate_illuminant(model, image_path, rngs=None):
    img = preprocess_image(image_path)
    img_batch = jnp.expand_dims(img, axis=0)
    pred = model(img_batch, train=False, rngs=rngs or nnx.Rngs())
    return pred[0]


def main(args):
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        print("Available checkpoints:")
        for p in sorted(Path("checkpoints").glob("checkpoint_*")):
            print(f"  {p}")
        return

    print(f"Loading checkpoint: {checkpoint_path}")
    restored = checkpoints.restore_checkpoint(ckpt_dir=str(checkpoint_path.resolve()), target=None)
    model_state = nnx.restore_int_paths(restored["model"])

    config = ModelConfig()
    model = ViT(
        img_size=config.img_size,
        patch_size=config.patch_size,
        dim=config.dim,
        depth=config.depth,
        num_heads=config.num_heads,
        rngs=nnx.Rngs(0),
    )
    nnx.update(model, model_state)

    print(f"Loaded model from epoch {restored['epoch']}")

    print(f"\nEstimating illuminant for: {args.image}")
    rngs = nnx.Rngs(dropout=jax.random.key(0))
    pred = estimate_illuminant(model, args.image, rngs)

    r, g, b = float(pred[0]), float(pred[1]), float(pred[2])
    print(f"Chromaticity: {r:.4f}, {g:.4f}, {b:.4f}")
    print(f"RGB: rgb({r * 255:.0f}, {g * 255:.0f}, {b * 255:.0f})")


if __name__ == "__main__":
    main(sys.argv[1:])
