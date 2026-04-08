import math
import sys

import jax
import jax.numpy as jnp
from flax import nnx
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.pretty import pprint
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

import wandb
from data.loader import SimpleCubePPDataset
from flax_illuminant_estimation.checkpoint import CheckpointState, save
from flax_illuminant_estimation.config import Config
from flax_illuminant_estimation.lib import (
    Trainer,
    TrainState,
    compute_metrics,
    create_train_metrics,
    eval_step,
    train_step,
)
from flax_illuminant_estimation.lib.trainer import create_eval_metrics
from flax_illuminant_estimation.model import ViT


def main(args):
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()

    jax.config.update("jax_default_matmul_precision", "high")

    train_ds = SimpleCubePPDataset("train", seed=config.trainer.seed)
    test_ds = SimpleCubePPDataset("test", seed=config.trainer.seed + 1)

    train_steps = math.ceil(len(train_ds) / config.trainer.batch_size)
    eval_steps = math.ceil(len(test_ds) / config.trainer.batch_size)
    total_steps = config.trainer.epochs * train_steps

    rngs = nnx.Rngs(config.trainer.seed)
    model = ViT(
        img_size=config.model.img_size,
        patch_size=config.model.patch_size,
        dim=config.model.dim,
        depth=config.model.depth,
        num_heads=config.model.num_heads,
        rngs=rngs,
    )
    graphdef, _ = nnx.split(model)

    trainer = Trainer(config.trainer)
    state: TrainState = trainer.create_train_state(model, train_steps)

    # Logging
    use_wandb: bool = config.trainer.wandb
    if use_wandb:
        wandb.init(
            project="flax-illuminant-estimation",
            group=config.trainer.wandb_group,
            tags=config.trainer.wandb_tags,
            config=config.to_dict() | {"total_steps": total_steps},
            settings=wandb.Settings(console="off"),
        )
        wandb.define_metric("step/*", step_metric="step/global")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("eval/*", step_metric="epoch")
        wandb.define_metric("iec/*", step_metric="epoch")
        wandb.define_metric("repro/*", step_metric="epoch")
    pprint(config.to_dict(), expand_all=True, indent_guides=False)

    console = Console()
    table = Table(
        title="Training Progress",
        expand=True,
        row_styles=["", "dim"],
        box=box.SIMPLE,
        min_width=120,
    )
    table.add_column("epoch", justify="right", style="on black")
    table.add_column("train_loss")
    table.add_column("train_ae", style="cyan")
    table.add_column("eval_loss")
    table.add_column("eval_ae", style="cyan")
    table.add_column("eval_rae")

    progress = Progress(
        MofNCompleteColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    train = progress.add_task("Train", total=train_steps)
    eval = progress.add_task("Evaluation", total=eval_steps)
    train_metrics = create_train_metrics()
    eval_metrics = create_eval_metrics()

    with Live(Group(table, progress), console=console, refresh_per_second=4):
        for epoch in range(0, config.trainer.epochs):
            train_metrics.reset()
            eval_metrics.reset()
            progress.reset(train)
            progress.reset(eval)
            progress.update(
                train,
                description=f"epoch [u bold green]{epoch + 1}/{config.trainer.epochs}: training...",
            )
            progress.update(
                eval,
                description="[red]evaluating...",
            )

            # Training
            for i, (batch_images, batch_illum) in enumerate(
                train_ds.batches(config.trainer.batch_size)
            ):
                step: dict = train_step(
                    state, model, batch_images, batch_illum, config.trainer.dtype
                )
                train_metrics.update(loss=step["train/loss"], ae=step["train/ae"])

                # NOTE: .compute() here accumulates the metrics across all batches in this epoch
                _m = train_metrics.compute()

                if use_wandb and i % 5 == 0:
                    wandb.log(
                        {
                            "step/global": state.step.value,
                            "step/loss": float(jnp.mean(step["train/loss"])),
                            "step/lr": float(step["train/lr"]),
                        },
                    )

                progress.update(
                    train,
                    advance=1,
                    description=f"epoch {epoch + 1}/{config.trainer.epochs}: train loss \u2192 [i bold cyan]{float(_m['loss']):.7f}",
                )

            # Evaluation
            all_ae, all_repro = [], []

            for batch_images, batch_illum in test_ds.batches(
                config.trainer.batch_size, shuffle=False
            ):
                step: dict = eval_step(model, batch_images, batch_illum, config.trainer.dtype)
                all_ae.append(step["eval/ae"])
                all_repro.append(step["eval/rae"])
                eval_metrics.update(
                    loss=step["eval/loss"],
                    ae=step["eval/ae"],
                    rae=step["eval/rae"],
                )

                _m = eval_metrics.compute()

                progress.update(
                    eval,
                    advance=1,
                    description=f"epoch {epoch + 1}/{config.trainer.epochs}: eval loss \u2192 [i bold magenta]{float(_m['loss']):.7f}",
                )

            # Full distribution stats
            all_ae = jnp.concatenate(all_ae, axis=0)
            all_repro = jnp.concatenate(all_repro, axis=0)
            errors = {
                "angular": compute_metrics(all_ae),
                "repro": compute_metrics(all_repro),
            }

            train_m = train_metrics.compute()
            eval_m = eval_metrics.compute()

            model_state = nnx.state(model)
            ckpt = CheckpointState(
                graphdef=graphdef,
                model_state=model_state,
                epoch=epoch + 1,
                config=config.to_dict(),
            )
            save(ckpt, config.trainer.checkpoint_dir)

            if use_wandb:
                wandb.log(
                    {
                        "train/loss": float(train_m["loss"]),
                        "train/ae": float(train_m["ae"]),
                        "eval/loss": float(eval_m["loss"]),
                        "eval/ae": float(eval_m["ae"]),
                        "eval/rae": float(eval_m["rae"]),
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
                f"{float(train_m['loss']):.6f}",
                f"{float(train_m['ae']):.3f}\xb0",
                f"{float(eval_m['loss']):.6f}",
                f"{float(eval_m['ae']):.3f}\xb0",
                f"{float(eval_m['rae']):.3f}\xb0",
            )

        if use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main(sys.argv[1:])
