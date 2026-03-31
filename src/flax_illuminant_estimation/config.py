from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import jax.numpy as jnp
import yaml

DTYPE_MAP = {
    "float16": jnp.float16,
    "bfloat16": jnp.bfloat16,
    "float32": jnp.float32,
}


@dataclass
class BaseConfig:
    img_size: int = 224
    patch_size: int = 16
    dim: int = 384
    depth: int = 6
    num_heads: int = 6
    batch_size: int = 32
    learning_rate: float = 1e-6
    epochs: int = 10
    seed: int = 42
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    precision: str = "float32"
    wandb: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BaseConfig":
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f) or {}

        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in config_dict.items() if k in valid_fields}

        if "checkpoint_dir" in filtered and not isinstance(filtered["checkpoint_dir"], Path):
            filtered["checkpoint_dir"] = Path(filtered["checkpoint_dir"])

        return cls(**filtered)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return d

    @property
    def dtype(self):
        return DTYPE_MAP.get(self.precision, jnp.float32)


@dataclass
class ModelConfig(BaseConfig):
    pass


@dataclass
class TrainerConfig(BaseConfig):
    def __post_init__(self):
        if isinstance(self.checkpoint_dir, str):
            self.checkpoint_dir = Path(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
