import pickle
import uuid

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from tqdm import tqdm

import wandb
from data.loader import SimpleCubePPDataset
from flax_illuminant_estimation.config import TrainerConfig
from flax_illuminant_estimation.model import ViT


DTYPE_MAP = {
    "float16": jnp.float16,
    "bfloat16": jnp.bfloat16,
    "float32": jnp.float32,
}


def save_checkpoint(model, epoch, test_loss, config: TrainerConfig):
    checkpoint = {
        "model": nnx.state(model),
        "epoch": epoch,
        "test_loss": test_loss,
    }
    path = config.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pkl"
    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)
    return path


def load_checkpoint(path, model):
    with open(path, "rb") as f:
        checkpoint = pickle.load(f)
    nnx.update(model, checkpoint["model"])
    return checkpoint


def normalize_illuminant(illum):
    norm = jnp.linalg.norm(illum.astype(jnp.float32), axis=-1, keepdims=True)
    norm = jnp.where(norm < 1e-8, 1.0, norm)
    return (illum.astype(jnp.float32) / norm).astype(illum.dtype)


def angular_loss(pred, target):
    pred_f32 = jnp.asarray(pred, dtype=jnp.float32)
    target_f32 = jnp.asarray(target, dtype=jnp.float32)
    cos_sim = jnp.sum(pred_f32 * target_f32, axis=-1)
    cos_sim = jnp.clip(cos_sim, -1.0 + 1e-7, 1.0 - 1e-7)
    return jnp.mean(jnp.arccos(cos_sim))


def train_step(model, optimizer, rngs, batch_images, batch_illum, dtype):
    def loss_fn(model):
        images = batch_images.astype(dtype)
        illum = batch_illum.astype(dtype)
        pred = model(images, train=True, rngs=rngs)
        target = normalize_illuminant(illum)
        loss = angular_loss(pred, target)
        return loss.astype(jnp.float32)

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


def main(config: TrainerConfig | None = None):
    if config is None:
        config = TrainerConfig()

    jax.config.update("jax_default_matmul_precision", "high")

    dtype = DTYPE_MAP.get(config.dtype, jnp.float32)

    train_ds = SimpleCubePPDataset("train")
    test_ds = SimpleCubePPDataset("test")

    rngs = nnx.Rngs(config.seed)
    model = ViT(
        img_size=config.img_size,
        patch_size=config.patch_size,
        dim=config.dim,
        depth=config.depth,
        num_heads=config.num_heads,
        rngs=rngs,
    )
    optimizer = nnx.ModelAndOptimizer(
        model, optax.adamw(config.learning_rate), wrt=nnx.Param
    )

    rng_key = jax.random.key(config.seed)

    wandb.init(
        project="flax-illuminant-estimation",
        name=str(uuid.uuid4()),
        config=config.to_dict(),
    )

    print(f"Training on {jax.devices()} | Precision: {config.precision} ({dtype})")

    for epoch in range(config.epochs):
        train_loss = 0.0
        num_train_batches = 0

        pbar = tqdm(
            train_ds.batches(config.batch_size), desc="Train", leave=False, ncols=80
        )
        for batch_images, batch_illum in pbar:
            rng_key, params_key, dropout_key = jax.random.split(rng_key, num=3)
            rngs = nnx.Rngs(params=params_key, dropout=dropout_key)
            loss = train_step(model, optimizer, rngs, batch_images, batch_illum, dtype)
            train_loss += float(loss)
            num_train_batches += 1
            pbar.set_postfix(loss=f"{loss:.4f}", refresh=True)

        train_loss /= num_train_batches

        test_loss = 0.0
        num_test_batches = 0
        eval_rngs = nnx.Rngs(dropout=jax.random.key(0))
        for batch_images, batch_illum in test_ds.batches(
            config.batch_size, shuffle=False
        ):
            loss = evaluate(model, eval_rngs, batch_images, batch_illum)
            test_loss += float(loss)
            num_test_batches += 1

        test_loss /= num_test_batches

        path = save_checkpoint(model, epoch + 1, test_loss, config)

        wandb.log(
            {
                "train_loss": train_loss,
                "test_loss": test_loss,
                "epoch": epoch + 1,
            }
        )

        tqdm.write(
            f"Epoch {epoch + 1:>2}/{config.epochs} | Train: {train_loss:.4f} | Test: {test_loss:.4f} | Saved: {path.name}"
        )

    wandb.finish()


if __name__ == "__main__":
    main()
