import sys

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from rich.console import Console, Group
from rich.live import Live
from rich.pretty import pprint
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

import wandb
from data.loader import SimpleCubePPDataset
from flax_illuminant_estimation.checkpoint import CheckpointState, load, save
from flax_illuminant_estimation.config import Config
from flax_illuminant_estimation.model import ViT

__import__("dotenv").load_dotenv()


@jax.jit
def angular_loss(pred, target):
    cos_sim = jnp.sum(pred * target, axis=-1) / (
        jnp.linalg.norm(pred, axis=-1) * jnp.linalg.norm(target, axis=-1) + 1e-8
    )
    return jnp.arccos(jnp.clip(cos_sim, -1.0, 1.0))


def train_step(model, optimizer, rngs, batch_images, batch_illum, dtype):
    def loss_fn(model):
        images = batch_images.astype(dtype)
        illum = batch_illum.astype(dtype)
        pred = model(images, train=True, rngs=rngs)
        loss = angular_loss(pred.astype(jnp.float32), illum.astype(jnp.float32))

        return jnp.mean(loss)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    grads = jax.tree.map(lambda g: jnp.where(jnp.isnan(g), jnp.zeros_like(g), g), grads)

    optimizer.update(grads)
    return loss


@nnx.jit
def evaluate(model, rngs, batch_images, batch_illum):
    pred = model(batch_images, train=False, rngs=rngs)
    errors = angular_loss(pred, batch_illum)
    return jnp.degrees(errors)


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
        "worst": float(sorted_e[-1]),
    }


def create_metrics() -> nnx.MultiMetric:
    return nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
    )


def main(args):
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()

    jax.config.update("jax_default_matmul_precision", "high")

    train_ds = SimpleCubePPDataset("train")
    test_ds = SimpleCubePPDataset("test")

    rngs = nnx.Rngs(config.trainer.seed)
    rng_key = jax.random.key(config.trainer.seed)
    model = ViT(
        img_size=config.model.img_size,
        patch_size=config.model.patch_size,
        dim=config.model.dim,
        depth=config.model.depth,
        num_heads=config.model.num_heads,
        rngs=rngs,
    )

    steps_per_epoch = len(train_ds) // config.trainer.batch_size
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.trainer.learning_rate,
        warmup_steps=3 * steps_per_epoch,
        decay_steps=config.trainer.epochs * steps_per_epoch,
        end_value=config.trainer.learning_rate * 0.01,
    )
    optimizer = nnx.ModelAndOptimizer(
        model,
        optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(schedule, weight_decay=0.05),
        ),
        wrt=nnx.Param,
    )

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        state = load(args.resume)
        nnx.update(model, state.model_state)
        start_epoch = state.epoch
        print(f"Resumed from epoch {start_epoch}")
    else:
        print("No checkpoint found, starting training from scratch")
        start_epoch = 0

    train_metrics = create_metrics()
    use_wandb: bool = config.trainer.wandb

    if use_wandb:
        wandb.init(project="flax-illuminant-estimation", tags=["IEC"], config=config.to_dict())

    print(
        f"Training on {jax.devices()} | Precision: {config.trainer.precision} ({config.trainer.dtype})"
    )
    pprint(config.to_dict(), expand_all=True, indent_guides=False)

    console = Console()
    table = Table(title="Training Progress", expand=True, width=100)
    [
        table.add_column(x)
        for x in [
            "epoch",
            "train",
            "mean",
            "median",
            "trimean",
            "best 25%",
            "worst 25%",
            "worst",
        ]
    ]
    progress = Progress(
        TimeElapsedColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    task = progress.add_task("Train", total=steps_per_epoch)

    with Live(Group(table, progress), console=console, refresh_per_second=4):
        for epoch in range(start_epoch, config.trainer.epochs):
            train_metrics.reset()
            progress.reset(task)
            progress.update(
                task, description=f"epoch {epoch + 1}/{config.trainer.epochs}: training..."
            )

            # Train
            for batch_images, batch_illum in train_ds.batches(config.trainer.batch_size):
                rng_key, params_key, dropout_key = jax.random.split(rng_key, num=3)
                rngs = nnx.Rngs(params=params_key, dropout=dropout_key)
                loss = train_step(
                    model,
                    optimizer,
                    rngs,
                    batch_images,
                    batch_illum,
                    config.trainer.dtype,
                )

                train_metrics.update(loss=loss, angular_error=jnp.degrees(loss))
                m = train_metrics.compute()
                progress.update(
                    task,
                    advance=1,
                    description=f"epoch {epoch + 1}/{config.trainer.epochs}: train loss {m['loss']:.7f} \u2192 {jnp.degrees(m['loss']):.3f}\xb0",
                )

            # Eval
            eval_rngs = nnx.Rngs(dropout=jax.random.key(0))

            all_errors = []

            for batch_images, batch_illum in test_ds.batches(
                config.trainer.batch_size, shuffle=False
            ):
                errors = evaluate(model, eval_rngs, batch_images, batch_illum)
                all_errors.append(errors)

            all_errors = jnp.concatenate(all_errors, axis=0)
            iec = iec_metrics(all_errors)

            train_m = train_metrics.compute()

            graphdef, model_state = nnx.split(model)
            state = CheckpointState(
                graphdef=graphdef,
                model_state=model_state,
                epoch=epoch + 1,
                config=config.to_dict(),
            )
            save(state, config.trainer.checkpoint_dir)

            if use_wandb:
                wandb.log(
                    {
                        "train/loss": float(train_m["loss"]),
                        "train/loss_deg": float(jnp.degrees(train_m["loss"])),
                        "iec/mean": iec["mean"],
                        "iec/median": iec["median"],
                        "iec/trimean": iec["trimean"],
                        "iec/best_25": iec["best_25"],
                        "iec/worst_25": iec["worst_25"],
                        "iec/worst": iec["worst"],
                        "epoch": epoch + 1,
                    }
                )

            table.add_row(
                str(epoch + 1).zfill(2),
                f"{train_m['loss']:.3f} \u2192 {jnp.degrees(train_m['loss']):.3f}\xb0",
                f"{iec['mean']:.2f}\xb0",
                f"{iec['median']:.2f}\xb0",
                f"{iec['trimean']:.2f}\xb0",
                f"{iec['best_25']:.2f}\xb0",
                f"{iec['worst_25']:.2f}\xb0",
                f"{iec['worst']:.2f}\xb0",
            )

        if use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main(sys.argv[1:])
