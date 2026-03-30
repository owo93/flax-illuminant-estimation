from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    img_size: int = 224
    patch_size: int = 16
    dim: int = 384
    depth: int = 6
    num_heads: int = 6

    def to_dict(self):
        return {
            "img_size": self.img_size,
            "patch_size": self.patch_size,
            "dim": self.dim,
            "depth": self.depth,
            "num_heads": self.num_heads,
        }


@dataclass
class TrainerConfig:
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

    def __post_init__(self):
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

    @property
    def dtype(self):
        if self.precision == "float16":
            return "float16"
        elif self.precision == "bfloat16":
            return "bfloat16"
        return "float32"

    def to_dict(self):
        return {
            "img_size": self.img_size,
            "patch_size": self.patch_size,
            "dim": self.dim,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "precision": self.precision,
        }
