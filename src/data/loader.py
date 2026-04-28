import jax
import jax.numpy as jnp
from datasets import load_dataset
from jax import lax, random
from PIL import Image


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
    def __init__(self, split, seed=42, img_size=224):
        self.split = split
        self.augment = split == "train"
        self.key = random.key(seed)
        self.samples = self._load_split(split)
        self.img_size = img_size

    def _load_split(self, split):
        return load_dataset("owo93/SimpleCubePP", split=split)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = sample["image"].convert("RGB")
        img = img.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
        img = jnp.array(img, dtype=jnp.float32) / 255.0
        illum = jnp.array(sample["illuminant"], dtype=jnp.float32)

        return img, illum

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
