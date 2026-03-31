import sys

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from tqdm import tqdm

import wandb
from data.loader import SimpleCubePPDataset
from flax_illuminant_estimation.checkpoint import CheckpointState, load, save
from flax_illuminant_estimation.config import TrainerConfig
from flax_illuminant_estimation.model import ViT

__import__("dotenv").load_dotenv()


def mse_loss(pred, target):
    return jnp.mean((pred - target) ** 2)


def angular_error(pred, target):
    pred_norm = pred / jnp.maximum(jnp.linalg.norm(pred, axis=-1, keepdims=True), 1e-6)
    target_norm = target / jnp.maximum(jnp.linalg.norm(target, axis=-1, keepdims=True), 1e-6)
    cos_sim = jnp.sum(pred_norm * target_norm, axis=-1)
    cos_sim = jnp.clip(cos_sim, -1.0, 1.0)

    # jnp.degrees?
    return jnp.arccos(cos_sim) * (180.0 / jnp.pi)


def iec_metrics(errors):
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
    }


def train_step(model, optimizer, rngs, batch_images, batch_illum, dtype):
    def loss_fn(model):
        images = batch_images.astype(dtype)
        illum = batch_illum.astype(dtype)
        pred = model(images, train=True, rngs=rngs)
        return mse_loss(pred.astype(jnp.float32), illum.astype(jnp.float32)).astype(jnp.float32)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    if jnp.isnan(loss):
        grads = jax.tree.map(jnp.zeros_like, grads)
        print("Warning: NaN loss detected, using zero grads")
    optimizer.update(grads)
    return loss


def evaluate(model, rngs, metrics, batch_images, batch_illum):
    pred = model(batch_images, train=False, rngs=rngs)
    loss = mse_loss(pred, batch_illum)
    error = angular_error(pred, batch_illum)
    metrics.update(loss=loss, angular_error=jnp.mean(error))
    return error


def create_metrics() -> nnx.MultiMetric:
    return nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
        angular_error=nnx.metrics.Average("angular_error"),
    )


def main(args):
    if args.config:
        config = TrainerConfig.from_yaml(args.config)
    else:
        config = TrainerConfig()

    jax.config.update("jax_default_matmul_precision", "high")

    dtype = config.dtype

    train_ds = SimpleCubePPDataset("train")
    test_ds = SimpleCubePPDataset("test")

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        state = load(args.resume)
        rngs = nnx.Rngs(config.seed)
        model = ViT(
            img_size=config.img_size,
            patch_size=config.patch_size,
            dim=config.dim,
            depth=config.depth,
            num_heads=config.num_heads,
            rngs=rngs,
        )
        nnx.update(model, state.model_state)
        optimizer = nnx.ModelAndOptimizer(model, optax.adamw(config.learning_rate), wrt=nnx.Param)
        start_epoch = state.epoch
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
        optimizer = nnx.ModelAndOptimizer(model, optax.adamw(config.learning_rate), wrt=nnx.Param)
        start_epoch = 0
        rng_key = jax.random.key(config.seed)

    train_metrics = create_metrics()
    eval_metrics = create_metrics()

    use_wandb = config.wandb

    if use_wandb:
        wandb.init(project="flax-illuminant-estimation", config=config.to_dict())
        run_config = wandb.config
    else:
        run_config = config

    print(f"Training on {jax.devices()} | Precision: {run_config.precision} ({dtype})")
    print(f"Config: {config.to_dict()}")

    for epoch in range(start_epoch, run_config.epochs):
        train_metrics.reset()

        pbar = tqdm(
            train_ds.batches(run_config.batch_size),
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
            error = angular_error(pred, batch_illum)
            train_metrics.update(loss=loss, angular_error=error)
            m = train_metrics.compute()
            pbar.set_postfix(loss=f"{m['loss']:.6f}", refresh=True)

        eval_metrics.reset()
        eval_rngs = nnx.Rngs(dropout=jax.random.key(0))

        all_errors = []
        for batch_images, batch_illum in test_ds.batches(run_config.batch_size, shuffle=False):
            errors = evaluate(model, eval_rngs, eval_metrics, batch_images, batch_illum)
            all_errors.append(errors)

        all_errors = jnp.concatenate(all_errors, axis=0)
        iec = iec_metrics(all_errors)

        train_m = train_metrics.compute()
        eval_m = eval_metrics.compute()

        graphdef, model_state = nnx.split(model)
        state = CheckpointState(
            graphdef=graphdef,
            model_state=model_state,
            epoch=epoch + 1,
            config=config.to_dict(),
        )
        checkpoint_path = save(state, config.checkpoint_dir)

        if use_wandb:
            wandb.log(
                {
                    "train_loss": train_m["loss"],
                    "train_angular_error": train_m["angular_error"],
                    "eval_loss": eval_m["loss"],
                    "eval_angular_error": eval_m["angular_error"],
                    "mean": iec["mean"],
                    "median": iec["median"],
                    "trimean": iec["trimean"],
                    "best_25": iec["best_25"],
                    "worst_25": iec["worst_25"],
                    "epoch": epoch + 1,
                }
            )

        tqdm.write(
            f"Epoch {epoch + 1:>2}/{run_config.epochs} | "
            f"Train: loss={train_m['loss']:.6f}, error={train_m['angular_error']:.2f}° | "
            f"Eval: loss={eval_m['loss']:.6f}, error={eval_m['angular_error']:.2f}° | "
            f"Mean: {iec['mean']:.2f}°, Median: {iec['median']:.2f}°, Trimean: {iec['trimean']:.2f}°, "
            f"Saved: {checkpoint_path}"
        )

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main(sys.argv[1:])
