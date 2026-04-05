import jax
import jax.numpy as jnp
import optax
from flax import nnx

from flax_illuminant_estimation.config import TrainerConfig
from flax_illuminant_estimation.lib import angular_error, reproduction_angular_error
from flax_illuminant_estimation.model import ViT


def compute_metrics(errors):
    n = len(errors)
    sorted_e = jnp.sort(errors)
    q1 = float(jnp.percentile(errors, 25))
    q2 = float(jnp.percentile(errors, 50))
    q3 = float(jnp.percentile(errors, 75))
    return {
        "mean": float(jnp.mean(errors)),
        "median": q2,
        "trimean": 0.25 * q1 + 0.5 * q2 + 0.25 * q3,
        "best_25": float(jnp.mean(sorted_e[: n // 4])),
        "worst_25": float(jnp.mean(sorted_e[n - n // 4 :])),
        "worst": float(sorted_e[-1]),
    }


# subclass nnx.Optimizer to directly call .update()
class TrainState(nnx.Optimizer):
    def __init__(
        self,
        model: ViT,
        tx: optax.GradientTransformation,
        schedule: optax.Schedule,
    ):
        super().__init__(model, tx, wrt=nnx.Param)
        self.schedule = schedule
        self.global_step = nnx.Variable(0)
        self.model = model

    @property
    def lr(self):
        return self.schedule(self.global_step.value)


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

        tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=0.05))

        return TrainState(model, tx, schedule)

    def train_step(
        self, state: TrainState, batch_images: jnp.ndarray, batch_illum: jnp.ndarray, dtype
    ):
        loss, grads = nnx.value_and_grad(self._loss_fn)(
            state.model, batch_images, batch_illum, dtype
        )

        grads = jax.tree.map(lambda g: jnp.where(jnp.isnan(g), jnp.zeros_like(g), g), grads)
        state.update(state.model, grads)

        state.global_step.value += 1

        return {"train/loss": loss, "train/lr": state.lr}

    @staticmethod
    def _loss_fn(model: ViT, batch_images, batch_illum, dtype):
        pred = model(batch_images.astype(dtype), train=True)
        return jnp.mean(
            angular_error(
                pred.astype(jnp.float32),
                batch_illum.astype(jnp.float32),
            )
        )

    def eval_step(self, state: TrainState, batch_images, batch_illum, dtype) -> dict:
        pred = state.model(batch_images.astype(dtype), train=False)
        images = batch_images.astype(jnp.float32)
        illums = batch_illum.astype(jnp.float32)

        ae = angular_error(pred, illums)
        rae = jax.vmap(reproduction_angular_error)(images, pred, illums)

        return {
            "eval/loss": ae,
            "eval/ae": jnp.degrees(ae),
            "eval/rae": jnp.degrees(rae),
        }

    @staticmethod
    def create_metrics() -> nnx.MultiMetric:
        return nnx.MultiMetric(loss=nnx.metrics.Average("loss"))
