import pickle
import uuid
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from tqdm import tqdm

import wandb
from data.loader import SimpleCubePPDataset
from flax_illuminant_estimation.model import ViT

DATASET_ROOT = Path("src/data")
IMG_SIZE = 224
PATCH_SIZE = 16
DIM = 384
DEPTH = 6
NUM_HEADS = 6
BATCH_SIZE = 48
LEARNING_RATE = 1e-5
EPOCHS = 10

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


def normalize_illuminant(illum):
    norm = jnp.linalg.norm(illum, axis=-1, keepdims=True)
    norm = jnp.where(norm < 1e-8, 1.0, norm)
    return illum / norm


def angular_loss(pred, target):
    return jnp.mean(jnp.arccos(jnp.clip(jnp.sum(pred * target, axis=-1), -1, 1)))


def train_step(model, optimizer, rngs, batch_images, batch_illum):
    def loss_fn(model):
        pred = model(batch_images, train=True, rngs=rngs)
        target = normalize_illuminant(batch_illum)
        loss = angular_loss(pred, target)
        return loss

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    if jnp.isnan(loss):
        grads = jax.tree.map(jnp.zeros_like, grads)
        print("Warning: NaN loss detected, skipping gradient update")
    optimizer.update(grads)
    return loss


def evaluate(model, rngs, batch_images, batch_illum):
    pred = model(batch_images, train=False, rngs=rngs if rngs else nnx.Rngs())
    target = normalize_illuminant(batch_illum)
    return angular_loss(pred, target)


def main():
    train_ds = SimpleCubePPDataset("train")
    test_ds = SimpleCubePPDataset("test")

    rngs = nnx.Rngs(0)
    model = ViT(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        dim=DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        rngs=rngs,
    )
    optimizer = nnx.ModelAndOptimizer(model, optax.adamw(LEARNING_RATE), wrt=nnx.Param)

    rng_key = jax.random.key(42)

    config = {
        "img_size": IMG_SIZE,
        "patch_size": PATCH_SIZE,
        "dim": DIM,
        "depth": DEPTH,
        "num_heads": NUM_HEADS,
        "batch_size": BATCH_SIZE,
        "lr": LEARNING_RATE,
        "epochs": EPOCHS,
    }

    wandb.init(
        project="flax-illuminant-estimation", name=str(uuid.uuid4()), config=config
    )

    print(f"Training on {jax.devices()}")

    for epoch in range(EPOCHS):
        train_loss = 0.0
        num_train_batches = 0

        pbar = tqdm(train_ds.batches(BATCH_SIZE), desc="Train", leave=False, ncols=80)
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
        for batch_images, batch_illum in test_ds.batches(BATCH_SIZE, shuffle=False):
            loss = evaluate(model, eval_rngs, batch_images, batch_illum)
            test_loss += float(loss)
            num_test_batches += 1

        test_loss /= num_test_batches

        path = save_checkpoint(model, epoch + 1, test_loss)

        wandb.log(
            {
                "train_loss": train_loss,
                "test_loss": test_loss,
                "epoch": epoch + 1,
            }
        )

        tqdm.write(
            f"Epoch {epoch + 1:>2}/{EPOCHS} | Train: {train_loss:.4f} | Test: {test_loss:.4f} | Saved: {path.name}"
        )

    wandb.finish()


if __name__ == "__main__":
    main()
