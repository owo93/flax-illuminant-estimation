import argparse
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx
from PIL import Image

from flax_illuminant_estimation.model import ViT

IMG_SIZE = 224
PATCH_SIZE = 16
DIM = 384
DEPTH = 6
NUM_HEADS = 6


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


def main():
    parser = argparse.ArgumentParser(description="Estimate illuminant from an image")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/checkpoint_epoch_010.pkl",
        help="Path to checkpoint file",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        print("Available checkpoints:")
        for p in Path("checkpoints").glob("*.pkl"):
            print(f"  {p}")
        return

    print(f"Loading checkpoint: {checkpoint_path}")
    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)

    model = ViT(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        dim=DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        rngs=nnx.Rngs(0),
    )
    nnx.update(model, checkpoint["model"])

    print(
        f"Loaded model from epoch {checkpoint['epoch']} (test_loss: {checkpoint['test_loss']:.4f})"
    )

    print(f"\nEstimating illuminant for: {args.image}")
    rngs = nnx.Rngs(dropout=jax.random.key(0))
    pred = estimate_illuminant(model, args.image, rngs)

    pred_sum = pred[0] + pred[1] + pred[2]
    if pred_sum > 0:
        print(f"Chromaticity: [{pred[0]:.4f}, {pred[1]:.4f}, {pred[2]:.4f}]")
        print(
            f"Normalized: [{pred[0] / pred_sum:.4f}, {pred[1] / pred_sum:.4f}, {pred[2] / pred_sum:.4f}]"
        )
        print(f"RGB: [{pred[0] * 255:.0f}, {pred[1] * 255:.0f}, {pred[2] * 255:.0f}]")


if __name__ == "__main__":
    main()
