import jax.numpy as jnp
from flax import nnx


class PatchEmbedding(nnx.Module):
    def __init__(self, *, img_size, patch_size, dim, rngs):
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.dim = dim

        self.project = nnx.Conv(
            in_features=3,
            out_features=dim,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="VALID",
            rngs=rngs,
        )

        self.pos_embed = nnx.Param(jnp.zeros((1, self.num_patches, dim)))

    def __call__(self, x):
        B = x.shape[0]
        x = self.project(x)

        x = x.reshape(B, -1, self.dim)
        x += self.pos_embed[...]

        return x


class MLP(nnx.Module):
    def __init__(self, dim, mlp_dim, dropout_rate, rngs) -> None:
        self.fc1 = nnx.Linear(dim, mlp_dim, rngs=rngs)
        self.fc2 = nnx.Linear(mlp_dim, dim, rngs=rngs)

        self.dropout = nnx.Dropout(dropout_rate)

    def __call__(self, x, *, train, rngs):
        x = self.fc1(x)
        x = nnx.gelu(x)
        x = self.dropout(x, deterministic=not train, rngs=rngs)
        x = self.fc2(x)
        x = self.dropout(x, deterministic=not train, rngs=rngs)
        return x


class Encoder(nnx.Module):
    def __init__(self, dim, num_heads, mlp_ratio, dropout_rate, rngs):
        mlp_dim = int(dim * mlp_ratio)

        self.n1 = nnx.LayerNorm(dim, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=dim,
            qkv_features=dim,
            out_features=dim,
            dropout_rate=dropout_rate,
            decode=False,
            rngs=rngs,
        )

        self.n2 = nnx.LayerNorm(dim, rngs=rngs)
        self.mlp = MLP(dim, mlp_dim, dropout_rate, rngs)

        self.dropout = nnx.Dropout(dropout_rate)

    def __call__(self, x, *, train, rngs):
        h = self.n1(x)
        h = self.attn(
            inputs_q=h,
            inputs_k=h,
            inputs_v=h,
            deterministic=not train,
            rngs=rngs,
        )

        h = self.dropout(h, deterministic=not train, rngs=rngs)
        x += h

        h = self.n2(x)
        h = self.mlp(h, train=train, rngs=rngs)

        x += h

        return x


class ViT(nnx.Module):
    blocks = nnx.data()

    def __init__(
        self,
        *,
        img_size=224,
        patch_size=16,
        dim=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        dropout_rate=0.0,
        rngs: nnx.Rngs,
    ):
        self.patch_embed = PatchEmbedding(img_size=img_size, patch_size=patch_size, dim=dim, rngs=rngs)

        self.blocks = [Encoder(dim, num_heads, mlp_ratio, dropout_rate, rngs) for _ in range(depth)]

        self.norm = nnx.LayerNorm(dim, rngs=rngs)

        self.head = nnx.Linear(dim, 3, rngs=rngs)

    def __call__(self, x, *, train, rngs):
        x = self.patch_embed(x)

        for block in self.blocks:
            x = block(x, train=train, rngs=rngs)

        x = self.norm(x)

        feat = jnp.mean(x, axis=1)

        out = self.head(feat)
        out = nnx.softmax(out, axis=-1)

        return out
