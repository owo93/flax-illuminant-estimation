import jax
import jax.numpy as jnp
import optax
from flax import nnx

from flax_illuminant_estimation.config import TrainerConfig
from flax_illuminant_estimation.lib import angular_error, reproduction_angular_error
from flax_illuminant_estimation.model import ViT


# subclass nnx.Optimizer to directly call .update()
class TrainState(nnx.Optimizer):
    def __init__(self, model: ViT, tx: optax.GradientTransformation, schedule: optax.Schedule):
        super().__init__(model, tx, wrt=nnx.Param)
        self.schedule = schedule
        self.model = model

    @property
    def lr(self):
        return self.schedule(self.step.value)


class Trainer:
    def __init__(self, config: TrainerConfig):
        self.config = config

    def create_schedule(self, epochs, peak_lr, steps_per_epoch):
        warmup_epochs = 3
        warmup_steps = warmup_epochs * steps_per_epoch

        total_steps = epochs * steps_per_epoch

        return optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=peak_lr,
            warmup_steps=warmup_steps,
            decay_steps=total_steps - warmup_steps,
            end_value=peak_lr * 0.01,
        )

    def create_train_state(self, model: ViT, steps_per_epoch) -> TrainState:
        config: TrainerConfig = self.config
        schedule = self.create_schedule(config.epochs, config.learning_rate, steps_per_epoch)

        tx = optax.chain(
            optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=config.weight_decay)
        )

        return TrainState(model, tx, schedule)


@nnx.jit(static_argnames=("dtype",))
def train_step(state: TrainState, model: ViT, batch_images, batch_illum, dtype):
    def loss_fn(model: ViT):
        pred = model(batch_images.astype(dtype), train=True).astype(jnp.float32)
        illum = batch_illum.astype(jnp.float32)

        cos_sim = optax.losses.cosine_similarity(pred, illum, epsilon=1e-8)
        ae = angular_error(cos_sim)

        loss = jnp.mean(1.0 - cos_sim)  # scalar
        return loss, ae

    (loss, ae), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    # grads = jax.tree.map(lambda g: jnp.where(jnp.isnan(g), jnp.zeros_like(g), g), grads)
    state.update(model, grads)
    return {
        "train/loss": loss,
        "train/ae": jnp.degrees(ae),  # (B, )
        "train/lr": state.lr,
    }


@nnx.jit(static_argnames=("dtype",))
def eval_step(model: ViT, batch_images, batch_illum, dtype) -> dict:
    pred = model(batch_images.astype(dtype), train=False).astype(jnp.float32)
    illum = batch_illum.astype(jnp.float32)
    image = batch_images.astype(jnp.float32)

    cos_sim = optax.losses.cosine_similarity(pred, illum, epsilon=1e-8)
    ae = angular_error(cos_sim)
    loss = 1.0 - cos_sim  # (B, )

    return {
        "eval/loss": loss,
        "eval/ae": jnp.degrees(ae),
        "eval/rae": jnp.degrees(jax.vmap(reproduction_angular_error)(image, pred, illum)),
    }


def create_eval_metrics() -> nnx.MultiMetric:
    return nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
        ae=nnx.metrics.Average("ae"),
        rae=nnx.metrics.Average("rae"),
    )


def create_train_metrics() -> nnx.MultiMetric:
    return nnx.MultiMetric(loss=nnx.metrics.Average("loss"), ae=nnx.metrics.Average("ae"))


def compute_metrics(errors):
    n = len(errors)
    sorted_e = jnp.sort(errors)
    q1 = float(jnp.percentile(errors, 25).squeeze())
    q2 = float(jnp.percentile(errors, 50).squeeze())
    q3 = float(jnp.percentile(errors, 75).squeeze())
    return {
        "mean": float(jnp.mean(errors)),
        "median": q2,
        "trimean": 0.25 * q1 + 0.5 * q2 + 0.25 * q3,
        "best_25": float(jnp.mean(sorted_e[: n // 4])),
        "worst_25": float(jnp.mean(sorted_e[n - n // 4 :])),
        "worst": float(sorted_e[-1].squeeze()),
    }
