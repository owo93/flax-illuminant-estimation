# Load a simplecube++ dataset as train/test split in jnp.array
import csv
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import lax, random
from PIL import Image
from tqdm import tqdm


@jax.jit
def augment(img, key):
    k1, k2, k3 = random.split(key, 3)
    # flip horizontal
    img = jnp.where(random.bernoulli(k1), jnp.flip(img, axis=1), img)

    # rotate
    k = random.randint(k2, shape=(), minval=0, maxval=4)
    img = lax.switch(
        k,
        [
            lambda x: x,
            lambda x: jnp.rot90(x, k=1, axes=(0, 1)),
            lambda x: jnp.rot90(x, k=2, axes=(0, 1)),
            lambda x: jnp.rot90(x, k=3, axes=(0, 1)),
        ],
        img,
    )

    # flip vertical
    img = jnp.where(random.bernoulli(k3), jnp.flip(img, axis=0), img)

    return img


batched_augment = jax.jit(jax.vmap(augment, in_axes=(0, 0)))


class SimpleCubePPDataset:
    def __init__(self, split, root=None, seed=42, img_size=224):
        self.root = Path(__file__).parent / "SimpleCube++"
        self.split = split
        self.augment = split == "train"
        self.key = random.key(seed)
        self.samples = self._load_split(split)
        self.img_size = img_size

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
                illum = jnp.array([row["mean_r"], row["mean_g"], row["mean_b"]], dtype=jnp.float32)

                samples.append({"image_path": img_dir / f"{filename}.png", "illuminant": illum})

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample["image_path"]).convert("RGB")
        img = img.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
        img = jnp.array(img, dtype=jnp.float32) / 255.0

        return img, sample["illuminant"]

    def batches(self, batch_size: int, shuffle=True):
        self.key, shuffle_key = random.split(self.key)

        indices = jnp.arange(len(self))
        if shuffle:
            indices = random.permutation(shuffle_key, indices)

        for start_idx in range(0, len(self), batch_size):
            self.key, augment_key = random.split(self.key)

            batch_indices = indices[start_idx : start_idx + batch_size]
            if len(batch_indices) < batch_size:
                continue

            images, illuminants = [], []
            for idx in batch_indices:
                img, illum = self[int(idx)]
                images.append(img)
                illuminants.append(illum)

            images = jnp.stack(images)

            if self.augment:
                batch_keys = random.split(augment_key, batch_size)
                images = batched_augment(images, batch_keys)

            yield images, jnp.stack(illuminants)
