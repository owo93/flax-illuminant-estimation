import math
import sys

import jax
import jax.numpy as jnp
from flax import nnx
from rich.console import Console, Group
from rich.live import Live
from rich.pretty import pprint
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

import wandb
from data.loader import SimpleCubePPDataset
from flax_illuminant_estimation.checkpoint import CheckpointState, save
from flax_illuminant_estimation.config import Config
from flax_illuminant_estimation.lib import Trainer, TrainState, compute_metrics
from flax_illuminant_estimation.model import ViT


def main(args):
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()

    jax.config.update("jax_default_matmul_precision", "high")

    train_ds = SimpleCubePPDataset("train", seed=config.trainer.seed)
    test_ds = SimpleCubePPDataset("test", seed=config.trainer.seed + 1)

    steps_per_epoch = math.ceil(len(train_ds) / config.trainer.batch_size)
    total_steps = config.trainer.epochs * steps_per_epoch

    rngs = nnx.Rngs(config.trainer.seed)
    model = ViT(
        img_size=config.model.img_size,
        patch_size=config.model.patch_size,
        dim=config.model.dim,
        depth=config.model.depth,
        num_heads=config.model.num_heads,
        rngs=rngs,
    )

    trainer = Trainer(config.trainer)
    state: TrainState = trainer.create_train_state(model, steps_per_epoch)

    # Logging
    use_wandb: bool = config.trainer.wandb
    if use_wandb:
        wandb.init(
            project="flax-illuminant-estimation",
            group=config.trainer.wandb_group,
            tags=["Step", "IEC", "RE"],
            config=config.to_dict() | {"total_steps": total_steps},
            settings=wandb.Settings(console="off"),
        )
    print(
        f"Training on {jax.devices()} | Precision: {config.trainer.precision} ({config.trainer.dtype})"
    )
    pprint(config.to_dict(), expand_all=True, indent_guides=False)

    console = Console()
    table = Table(title="Training Progress", expand=True, row_styles=["dim", ""], min_width=120)
    [
        table.add_column(x)
        for x in [
            "epoch",
            "train_loss",
            "eval_loss",
            "ae_mean",
            "ae_median",
            "ae_trimean",
            "ae_b25%",
            "ae_w25%",
            "ae_worst",
            "rep_mean",
            "rep_median",
            "rep_trimean",
            "rep_b25%",
            "rep_w25%",
            "rep_worst",
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
    train_metrics = Trainer.create_metrics()
    eval_metrics = Trainer.create_metrics()

    with Live(Group(table, progress), console=console, refresh_per_second=4):
        for epoch in range(0, config.trainer.epochs):
            train_metrics.reset()
            eval_metrics.reset()
            progress.reset(task)
            progress.update(
                task, description=f"epoch {epoch + 1}/{config.trainer.epochs}: training..."
            )

            # Training
            for batch_images, batch_illum in train_ds.batches(config.trainer.batch_size):
                step_metrics = trainer.train_step(
                    state, batch_images, batch_illum, config.trainer.dtype
                )
                train_metrics.update(loss=step_metrics["train/loss"])
                m = train_metrics.compute()

                if use_wandb:
                    wandb.define_metric("step/*", step_metric="step/global")
                    wandb.log(
                        {
                            "step/global": state.global_step.value,
                            "step/loss": float(step_metrics["train/loss"]),
                            "step/lr": float(step_metrics["train/lr"]),
                        },
                    )

                progress.update(
                    task,
                    advance=1,
                    description=f"epoch {epoch + 1}/{config.trainer.epochs}: train loss {float(m['loss']):.7f} \u2192 {jnp.degrees(float(m['loss'])):.3f}\xb0",
                )

            # Evaluation
            all_ae, all_repro = [], []

            for batch_images, batch_illum in test_ds.batches(
                config.trainer.batch_size, shuffle=False
            ):
                step = trainer.eval_step(state, batch_images, batch_illum, config.trainer.dtype)
                all_ae.append(step["eval/ae"])
                all_repro.append(step["eval/rae"])
                eval_metrics.update(loss=step["eval/loss"])

            all_ae = jnp.concatenate(all_ae, axis=0)
            all_repro = jnp.concatenate(all_repro, axis=0)

            errors = {"angular": compute_metrics(all_ae), "repro": compute_metrics(all_repro)}

            train_m = train_metrics.compute()
            eval_m = eval_metrics.compute()

            graphdef, model_state = nnx.split(model)
            ckpt = CheckpointState(
                graphdef=graphdef,
                model_state=model_state,
                epoch=epoch + 1,
                config=config.to_dict(),
            )
            save(ckpt, config.trainer.checkpoint_dir)

            if use_wandb:
                wandb.define_metric("train/*", step_metric="epoch")
                wandb.define_metric("eval/*", step_metric="epoch")
                wandb.define_metric("iec/*", step_metric="epoch")
                wandb.define_metric("repro/*", step_metric="epoch")
                wandb.log(
                    {
                        "train/loss": float(train_m["loss"]),
                        "train/loss_deg": float(jnp.degrees(train_m["loss"])),
                        "eval/loss": float(eval_m["loss"]),
                        "eval/loss_deg": float(jnp.degrees(eval_m["loss"])),
                        "iec/mean": errors["angular"]["mean"],
                        "iec/median": errors["angular"]["median"],
                        "iec/trimean": errors["angular"]["trimean"],
                        "iec/best_25": errors["angular"]["best_25"],
                        "iec/worst_25": errors["angular"]["worst_25"],
                        "iec/worst": errors["angular"]["worst"],
                        "repro/mean": errors["repro"]["mean"],
                        "repro/median": errors["repro"]["median"],
                        "repro/trimean": errors["repro"]["trimean"],
                        "repro/best_25": errors["repro"]["best_25"],
                        "repro/worst_25": errors["repro"]["worst_25"],
                        "repro/worst": errors["repro"]["worst"],
                        "epoch": epoch + 1,
                    }
                )

            table.add_row(
                f"{str(epoch + 1).zfill(2)}",
                f"{jnp.degrees(train_m['loss']):.3f}\xb0",
                f"{jnp.degrees(eval_m['loss']):.3f}\xb0",
                f"{errors['angular']['mean']:.2f}\xb0",
                f"{errors['angular']['median']:.2f}\xb0",
                f"{errors['angular']['trimean']:.2f}\xb0",
                f"{errors['angular']['best_25']:.2f}\xb0",
                f"{errors['angular']['worst_25']:.2f}\xb0",
                f"{errors['angular']['worst']:.2f}\xb0",
                f"{errors['repro']['mean']:.4f}",
                f"{errors['repro']['median']:.4f}",
                f"{errors['repro']['trimean']:.4f}",
                f"{errors['repro']['best_25']:.4f}",
                f"{errors['repro']['worst_25']:.4f}",
                f"{errors['repro']['worst']:.4f}",
            )

        if use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main(sys.argv[1:])
