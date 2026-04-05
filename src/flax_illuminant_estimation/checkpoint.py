from dataclasses import dataclass
from pathlib import Path

import flax.training.orbax_utils
import orbax.checkpoint as ocp
from flax import nnx


@dataclass
class CheckpointState:
    graphdef: nnx.GraphDef
    model_state: nnx.State
    epoch: int
    config: dict


_checkpointer = ocp.PyTreeCheckpointer()


def save(state: CheckpointState, checkpoint_dir: Path):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "graphdef": state.graphdef,
        "model": nnx.to_pure_dict(state.model_state),
        "epoch": state.epoch,
        "config": state.config,
    }
    step_dir = checkpoint_dir.resolve() / f"checkpoint_{int(state.epoch):02}"
    _checkpointer.save(
        step_dir, ckpt, save_args=flax.training.orbax_utils.save_args_from_target(ckpt), force=True
    )
    return step_dir


def load(path: Path, target: CheckpointState | None = None) -> CheckpointState:
    path = path.resolve()
    abstract_target = None
    if target is not None:
        abstract_target = {
            "graphdef": target.graphdef,
            "model": nnx.to_pure_dict(target.model_state),
            "epoch": target.epoch,
            "config": target.config,
        }
    restored = _checkpointer.restore(path, item=abstract_target)
    model_state = nnx.State(nnx.restore_int_paths(restored["model"]))
    return CheckpointState(
        graphdef=restored["graphdef"],
        model_state=model_state,
        epoch=int(restored["epoch"]),
        config=restored.get("config"),
    )


def list_checkpoints(checkpoint_dir: Path):
    if not checkpoint_dir.exists():
        return []

    return sorted(checkpoint_dir.glob("checkpoint_*"))


def latest(checkpoint_dir: Path) -> Path | None:
    found = list_checkpoints(checkpoint_dir)
    return found[-1] if found else None
