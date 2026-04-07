import jax
import jax.numpy as jnp
import optax


@jax.jit
def angular_error(cos_sim, epsilon: float = 1e-8):
    return jnp.arccos(jnp.clip(cos_sim, -1.0 + epsilon, 1.0 - epsilon))


# https://doi.org/10.1109/TPAMI.2016.2582171
@jax.jit
def reproduction_angular_error(image, pred, gt, epsilon: float = 1e-8):
    rendered_pred = image / (pred + epsilon)
    rendered_gt = image / (gt + epsilon)

    rgb_pred = jnp.sum(rendered_pred, axis=(0, 1))
    rgb_gt = jnp.sum(rendered_gt, axis=(0, 1))

    cos_sim = optax.losses.cosine_similarity(rgb_pred[None], rgb_gt[None], epsilon=epsilon)
    return jax.vmap(angular_error)(cos_sim)
