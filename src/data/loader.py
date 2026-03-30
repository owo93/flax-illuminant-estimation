# Load a simplecube++ dataset as train/test split in jnp.array
import csv
from pathlib import Path

import jax.numpy as jnp
from jax import random
from PIL import Image
from tqdm import tqdm


class SimpleCubePPDataset:
    def __init__(self, split, root=None):
        self.root = Path(__file__).parent / "SimpleCube++"
        self.split = split
        self.samples = self._load_split(split)

    def _load_split(self, split):
        split_root = self.root / split
        img_dir = split_root / "PNG"
        gt_path = split_root / "gt.csv"

        samples = []

        with open(gt_path, "r", newline="") as f:
            total_rows = sum(1 for _ in csv.DictReader(f))

        with open(gt_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in tqdm(reader, total=total_rows, ncols=80, desc=f"Loading {split} data"):
                filename = row["image"]
                illum = jnp.array(
                    [row["mean_r"], row["mean_g"], row["mean_b"]], dtype=jnp.float32
                )

                samples.append(
                    {"image_path": img_dir / f"{filename}.png", "illuminant": illum}
                )

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample["image_path"]).convert("RGB")
        img = img.resize((224, 224), Image.Resampling.BILINEAR)
        img = jnp.array(img, dtype=jnp.float32) / 255.0

        return jnp.array(img), sample["illuminant"]

    def batches(self, batch_size, shuffle=True):
        indices = jnp.arange(len(self))

        if shuffle:
            key = random.key(69)
            indices = random.permutation(key, indices)

        for start_idx in range(0, len(self), batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]

            images = []
            illuminants = []

            for idx in batch_indices:
                img, illum = self[idx]
                images.append(img)
                illuminants.append(illum)

            if len(images) == batch_size:
                yield jnp.stack(images), jnp.stack(illuminants)
