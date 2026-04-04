import jax
import jax.numpy as jnp


@jax.jit
def angular_error(pred, target):
    cos_sim = jnp.sum(pred * target, axis=-1) / (
        jnp.linalg.norm(pred, axis=-1) * jnp.linalg.norm(target, axis=-1) + 1e-8
    )
    return jnp.arccos(jnp.clip(cos_sim, -1.0, 1.0))


# https://doi.org/10.1109/TPAMI.2016.2582171
def reproduction_angular_error(image, pred, gt):
    rendered_pred = image / (pred + 1e-8)
    rendered_gt = image / (gt + 1e-8)

    rgb_pred = jnp.sum(rendered_pred, axis=(0, 1))
    rgb_gt = jnp.sum(rendered_gt, axis=(0, 1))

    cos_sim = jnp.dot(rgb_pred, rgb_gt) / (
        jnp.linalg.norm(rgb_pred) * jnp.linalg.norm(rgb_gt) + 1e-8
    )
    return jnp.arccos(jnp.clip(cos_sim, -1.0, 1.0))
