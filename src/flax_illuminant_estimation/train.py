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
        test_loss: float,
    ):
        self.graphdef = graphdef
        self.model_state = model_state
        self.epoch = epoch
        self.test_loss = test_loss


def save_checkpoint(train_state: TrainState, config: TrainerConfig):
    ckpt = {
        "graphdef": train_state.graphdef,
        "model": nnx.to_pure_dict(train_state.model_state),
        "epoch": train_state.epoch,
        "test_loss": train_state.test_loss,
    }
    path = config.checkpoint_dir / f"checkpoint_{train_state.epoch:03d}"
    checkpoints.save_checkpoint(
        ckpt_dir=str(path.resolve()),
        target=ckpt,
        step=train_state.epoch,
        overwrite=True,
    )
    return path


def load_checkpoint(path, config: TrainerConfig) -> TrainState:
    restored = checkpoints.restore_checkpoint(ckpt_dir=str(path.resolve()), target=None)
    model_state = nnx.restore_int_paths(restored["model"])
    return TrainState(
        graphdef=restored["graphdef"],
        model_state=model_state,  # pyright: ignore[reportArgumentType]
        epoch=restored["epoch"],
        test_loss=restored["test_loss"],
    )


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
        print("Warning: NaN loss detected, using zero grads")
    optimizer.update(grads)
    return loss


def evaluate(model, rngs, batch_images, batch_illum):
    pred = model(batch_images, train=False, rngs=rngs if rngs else nnx.Rngs())
    target = normalize_illuminant(batch_illum)
    return angular_loss(pred, target)


def load_config(path) -> TrainerConfig:
    with open(path) as f:
        data = yaml.safe_load(f) or {}

    config = asdict(TrainerConfig())
    merge = {**config, **data}
    merge["checkpoint_dir"] = Path(merge["checkpoint_dir"])
    return TrainerConfig(**merge)


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
        train_state = load_checkpoint(Path(args.resume), config)
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

    with wandb.init(
        project="flax-illuminant-estimation",
        config=config.to_dict(),
    ) as run:
        print(
            f"Training on {jax.devices()} | Precision: {run.config.precision} ({dtype})"
        )

        for epoch in range(start_epoch, run.config.epochs):
            train_loss = 0.0
            num_train_batches = 0

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
                    model, optimizer, rngs, batch_images, batch_illum, dtype
                )
                train_loss += float(loss)
                num_train_batches += 1
                pbar.set_postfix(loss=f"{loss:.4f}", refresh=True)

            train_loss /= num_train_batches

            test_loss = 0.0
            num_test_batches = 0
            eval_rngs = nnx.Rngs(dropout=jax.random.key(0))
            for batch_images, batch_illum in test_ds.batches(
                run.config.batch_size, shuffle=False
            ):
                loss = evaluate(model, eval_rngs, batch_images, batch_illum)
                test_loss += float(loss)
                num_test_batches += 1

            test_loss /= num_test_batches

            graphdef, model_state = nnx.split(model)
            train_state = TrainState(
                graphdef=graphdef,
                model_state=model_state,
                epoch=epoch + 1,
                test_loss=test_loss,
            )
            path = save_checkpoint(train_state, config)

            run.log(
                {
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "epoch": epoch + 1,
                }
            )

            tqdm.write(
                f"Epoch {epoch + 1:>2}/{run.config.epochs} | Train: {train_loss:.4f} | Test: {test_loss:.4f} | Saved: {path}"
            )

        run.finish()


if __name__ == "__main__":
    main(sys.argv[1:])
