from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import jax.numpy as jnp
import yaml

DTYPE_MAP = {
    "float16": jnp.float16,
    "bfloat16": jnp.bfloat16,
    "float32": jnp.float32,
}


@dataclass
class ModelConfig:
    img_size: int = 224
    patch_size: int = 16
    dim: int = 384
    depth: int = 6
    num_heads: int = 6


@dataclass
class TrainerConfig:
    batch_size: int = 32
    learning_rate: float = 1e-4
    epochs: int = 10
    seed: int = 42
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    precision: Literal["float16", "bfloat16", "float32"] = "float32"
    wandb: bool = False
    wandb_group: str = "A"
    wandb_tags: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.checkpoint_dir, Path):
            self.checkpoint_dir = Path(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @property
    def dtype(self):
        return DTYPE_MAP.get(self.precision, jnp.float32)


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}

        model_d = raw.get("model", {})
        trainer_d = raw.get("trainer", {})

        return cls(model=ModelConfig(**model_d), trainer=TrainerConfig(**trainer_d))

    def to_dict(self):
        def convert(obj):
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, (list, tuple)):
                return [convert(v) for v in obj]
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}

            return obj

        return convert(asdict(self))
