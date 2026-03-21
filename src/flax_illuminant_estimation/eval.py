from flax import nnx

from flax_illuminant_estimation.train import angular_loss, normalize_illuminant


def evaluate(model, rngs, batch_images, batch_illum):
    pred = model(batch_images, train=False, rngs=rngs if rngs else nnx.Rngs())
    target = normalize_illuminant(batch_illum)
    return angular_loss(pred, target)
