from dataclasses import dataclass
from pathlib import Path

from flax import nnx
from flax.training import checkpoints


@dataclass
class CheckpointState:
    graphdef: nnx.GraphDef
    model_state: nnx.State
    epoch: int
    config: dict


def save(state: CheckpointState, checkpoint_dir: Path):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "graphdef": state.graphdef,
        "model": nnx.to_pure_dict(state.model_state),
        "epoch": state.epoch,
        "config": state.config,
    }
    path = checkpoints.save_checkpoint(
        ckpt_dir=str(checkpoint_dir.resolve()),
        target=ckpt,
        step=state.epoch,
        prefix="checkpoint_",
        overwrite=True,
        keep=0,
    )
    return Path(path)


def load(checkpoint_dir: Path) -> CheckpointState:
    restored = checkpoints.restore_checkpoint(ckpt_dir=str(checkpoint_dir.resolve()), target=None)
    model_state = nnx.restore_int_paths(restored["model"])
    return CheckpointState(
        graphdef=restored["graphdef"],
        model_state=nnx.State(model_state),
        epoch=restored["epoch"],
        config=restored.get("config", {}),
    )


def list(checkpoint_dir: Path):
    if not checkpoint_dir.exists():
        return []
    return sorted(checkpoint_dir.glob("checkpoint_*"))


def latest(checkpoint_dir: Path) -> Path | None:
    found = list(checkpoint_dir)
    return found[-1] if found else None
