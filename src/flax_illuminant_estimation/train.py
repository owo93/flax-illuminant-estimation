import jax
import jax.numpy as jnp
from flax import nnx


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
