import sys
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import yaml
from flax import nnx
from flax.training import checkpoints
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


class TrainState:
    def __init__(
        self,
        graphdef: nnx.GraphDef,
        model_state: nnx.State,
        epoch: int,
    ):
        self.graphdef = graphdef
        self.model_state = model_state
        self.epoch = epoch


def save_checkpoint(train_state: TrainState, config: TrainerConfig):
    ckpt = {
        "graphdef": train_state.graphdef,
        "model": nnx.to_pure_dict(train_state.model_state),
        "epoch": train_state.epoch,
    }
    path = config.checkpoint_dir / f"checkpoint_{train_state.epoch:03d}"
    checkpoints.save_checkpoint(
        ckpt_dir=str(path.resolve()),
        target=ckpt,
        step=train_state.epoch,
        overwrite=True,
    )
    return path


def load_checkpoint(path: Path) -> TrainState:
    restored = checkpoints.restore_checkpoint(ckpt_dir=str(path.resolve()), target=None)
    model_state = nnx.restore_int_paths(restored["model"])
    return TrainState(
        graphdef=restored["graphdef"],
        model_state=nnx.State(model_state),
        epoch=restored["epoch"],
    )


def mse_loss(pred, target):
    return jnp.mean((pred - target) ** 2)


def angular_error_deg(pred, target):
    pred_norm = jnp.linalg.norm(pred, axis=-1, keepdims=True)
    pred_norm = jnp.maximum(pred_norm, 1e-6)
    target_norm = jnp.linalg.norm(target, axis=-1, keepdims=True)
    target_norm = jnp.maximum(target_norm, 1e-6)
    pred_unit = pred / pred_norm
    target_unit = target / target_norm
    cos_sim = jnp.sum(pred_unit * target_unit, axis=-1)
    cos_sim = jnp.clip(cos_sim, -1.0, 1.0)
    return jnp.mean(jnp.arccos(cos_sim)) * 180.0 / jnp.pi


def train_step(model, optimizer, rngs, batch_images, batch_illum, dtype):
    def loss_fn(model):
        images = batch_images.astype(dtype)
        illum = batch_illum.astype(dtype)
        pred = model(images, train=True, rngs=rngs)
        return mse_loss(pred.astype(jnp.float32), illum.astype(jnp.float32)).astype(
            jnp.float32
        )

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    if jnp.isnan(loss):
        grads = jax.tree.map(jnp.zeros_like, grads)
        print("Warning: NaN loss detected, using zero grads")
    optimizer.update(grads)
    return loss


def evaluate(model, rngs, metrics, batch_images, batch_illum):
    pred = model(batch_images, train=False, rngs=rngs)
    loss = mse_loss(pred, batch_illum)
    error = angular_error_deg(pred, batch_illum)
    metrics.update(loss=loss, angular_error=error)


def load_config(path) -> TrainerConfig:
    with open(path) as f:
        data = yaml.safe_load(f) or {}

    config = asdict(TrainerConfig())
    merge = {**config, **data}
    merge["checkpoint_dir"] = Path(merge["checkpoint_dir"])
    return TrainerConfig(**merge)


def create_metrics() -> nnx.MultiMetric:
    return nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
        angular_error=nnx.metrics.Average("angular_error"),
    )


def main(args):
    if args.config:
        config = load_config(args.config)
    else:
        config = TrainerConfig()

    jax.config.update("jax_default_matmul_precision", "high")

    dtype = DTYPE_MAP.get(config.dtype, jnp.float32)

    train_ds = SimpleCubePPDataset("train")
    test_ds = SimpleCubePPDataset("test")

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        train_state = load_checkpoint(Path(args.resume))
        rngs = nnx.Rngs(config.seed)
        model = ViT(
            img_size=config.img_size,
            patch_size=config.patch_size,
            dim=config.dim,
            depth=config.depth,
            num_heads=config.num_heads,
            rngs=rngs,
        )
        nnx.update(model, train_state.model_state)
        optimizer = nnx.ModelAndOptimizer(
            model, optax.adamw(config.learning_rate), wrt=nnx.Param
        )
        start_epoch = train_state.epoch
        rng_key = jax.random.key(config.seed)
        print(f"Resumed from epoch {start_epoch}")
    else:
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
        start_epoch = 0
        rng_key = jax.random.key(config.seed)

    train_metrics = create_metrics()
    eval_metrics = create_metrics()

    with wandb.init(
        project="flax-illuminant-estimation",
        config=config.to_dict(),
    ) as run:
        print(
            f"Training on {jax.devices()} | Precision: {run.config.precision} ({dtype})"
        )
        print(f"Config: {config.to_dict()}")

        for epoch in range(start_epoch, run.config.epochs):
            train_metrics.reset()

            pbar = tqdm(
                train_ds.batches(run.config.batch_size),
                desc="Train",
                leave=False,
                ncols=80,
            )
            for batch_images, batch_illum in pbar:
                rng_key, params_key, dropout_key = jax.random.split(rng_key, num=3)
                rngs = nnx.Rngs(params=params_key, dropout=dropout_key)
                loss = train_step(
                    model,
                    optimizer,
                    rngs,
                    batch_images,
                    batch_illum,
                    dtype,
                )
                pred = model(batch_images, train=False, rngs=rngs)
                error = angular_error_deg(pred, batch_illum)
                train_metrics.update(loss=loss, angular_error=error)
                m = train_metrics.compute()
                pbar.set_postfix(loss=f"{m['loss']:.6f}", refresh=True)

            eval_metrics.reset()
            eval_rngs = nnx.Rngs(dropout=jax.random.key(0))
            for batch_images, batch_illum in test_ds.batches(
                run.config.batch_size, shuffle=False
            ):
                evaluate(model, eval_rngs, eval_metrics, batch_images, batch_illum)

            train_m = train_metrics.compute()
            eval_m = eval_metrics.compute()

            graphdef, model_state = nnx.split(model)
            train_state = TrainState(
                graphdef=graphdef,
                model_state=model_state,
                epoch=epoch + 1,
            )
            path = save_checkpoint(train_state, config)

            run.log(
                {
                    "train_loss": train_m["loss"],
                    "train_angular_error": train_m["angular_error"],
                    "eval_loss": eval_m["loss"],
                    "eval_angular_error": eval_m["angular_error"],
                    "epoch": epoch + 1,
                }
            )

            tqdm.write(
                f"Epoch {epoch + 1:>2}/{run.config.epochs} | "
                f"Train: loss={train_m['loss']:.6f}, error={train_m['angular_error']:.2f}° | "
                f"Eval: loss={eval_m['loss']:.6f}, error={eval_m['angular_error']:.2f}° | "
                f"Saved: {path}"
            )

        run.finish()


if __name__ == "__main__":
    main(sys.argv[1:])
