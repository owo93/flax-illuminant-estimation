import jax
import jax.numpy as jnp
import optax
from flax import nnx
from tqdm import tqdm

from flax_illuminant_estimation.model import ViT


def normalize_illuminant(illum):
    return illum / jnp.linalg.norm(illum, axis=-1, keepdims=True)


def angular_loss(pred, target):
    return jnp.mean(jnp.arccos(jnp.clip(jnp.sum(pred * target, axis=-1), -1, 1)))


def train_step(model, optimizer, rngs, batch_images, batch_illum):
    def loss_fn(model):
        pred = model(batch_images, train=True, rngs=rngs)
        target = normalize_illuminant(batch_illum)
        return angular_loss(pred, target)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)
    return loss


def evaluate(model, rngs, batch_images, batch_illum):
    pred = model(batch_images, train=False, rngs=rngs if rngs else nnx.Rngs())
    target = normalize_illuminant(batch_illum)
    return angular_loss(pred, target)


def main() -> None:
    from data.loader import SimpleCubePPDataset

    train_ds = SimpleCubePPDataset("train")
    test_ds = SimpleCubePPDataset("test")

    rngs = nnx.Rngs(0)
    model = ViT(img_size=224, patch_size=16, dim=384, depth=6, num_heads=6, rngs=rngs)
    optimizer = nnx.ModelAndOptimizer(model, optax.adamw(1e-4), wrt=nnx.Param)

    rng_key = jax.random.key(42)

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

        tqdm.write(
            f"Epoch {epoch + 1:>2}/10 | Train: {train_loss:.4f} | Test: {test_loss:.4f}"
        )
