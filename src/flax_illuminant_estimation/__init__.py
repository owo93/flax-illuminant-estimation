import pickle
import uuid
from pathlib import Path

import jax
import optax
from flax import nnx
from tqdm import tqdm

import wandb
from flax_illuminant_estimation.eval import evaluate
from flax_illuminant_estimation.model import ViT
from flax_illuminant_estimation.train import train_step

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)


def save_checkpoint(model, epoch, test_loss):
    checkpoint = {
        "model": nnx.state(model),
        "epoch": epoch,
        "test_loss": test_loss,
    }
    path = CHECKPOINT_DIR / f"checkpoint_epoch_{epoch:03d}.pkl"
    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)
    return path


def load_checkpoint(path, model):
    with open(path, "rb") as f:
        checkpoint = pickle.load(f)
    nnx.update(model, checkpoint["model"])
    return checkpoint


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="flax-illuminant-estimation",
        help="Wandb project name",
    )
    args = parser.parse_args()

    from data.loader import SimpleCubePPDataset

    train_ds = SimpleCubePPDataset("train")
    test_ds = SimpleCubePPDataset("test")

    rngs = nnx.Rngs(0)
    model = ViT(img_size=224, patch_size=16, dim=384, depth=6, num_heads=6, rngs=rngs)
    optimizer = nnx.ModelAndOptimizer(model, optax.adamw(1e-5), wrt=nnx.Param)

    rng_key = jax.random.key(42)

    config = {
        "img_size": 224,
        "patch_size": 16,
        "dim": 384,
        "depth": 6,
        "num_heads": 6,
        "batch_size": 32,
        "lr": 1e-4,
        "epochs": 10,
    }

    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=str(uuid.uuid4()),
            config=config,
        )

    print(f"Training on {jax.devices()}")

    for epoch in range(10):
        train_loss = 0.0
        num_train_batches = 0

        pbar = tqdm(train_ds.batches(32), desc="Train", leave=False, ncols=80)
        for batch_images, batch_illum in pbar:
            rng_key, params_key, dropout_key = jax.random.split(rng_key, num=3)
            rngs = nnx.Rngs(params=params_key, dropout=dropout_key)
            loss = train_step(model, optimizer, rngs, batch_images, batch_illum)
            train_loss += float(loss)
            num_train_batches += 1
            pbar.set_postfix(loss=f"{loss:.4f}", refresh=True)

        train_loss /= num_train_batches

        test_loss = 0.0
        num_test_batches = 0
        eval_rngs = nnx.Rngs(dropout=jax.random.key(0))
        for batch_images, batch_illum in test_ds.batches(32, shuffle=False):
            loss = evaluate(model, eval_rngs, batch_images, batch_illum)
            test_loss += float(loss)
            num_test_batches += 1

        test_loss /= num_test_batches

        path = save_checkpoint(model, epoch + 1, test_loss)

        if args.wandb:
            wandb.log(
                {
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "epoch": epoch + 1,
                }
            )

        tqdm.write(
            f"Epoch {epoch + 1:>2}/10 | Train: {train_loss:.4f} | Test: {test_loss:.4f} | Saved: {path.name}"
        )

    if args.wandb:
        wandb.finish()
