# ViT for Illuminant estimation

An implementation of Vision Transformer for predicting single illuminant color in images trained on the [SimpleCube++](https://github.com/Visillect/CubePlusPlus)  dataset, built with Flax NNX

## Usage
By default, [`rich`](https://github.com/textualize/rich) terminal output is logged to `stdout` and `logging` log messages to `stderr`. 
> [!TIP]
> Live tail the logs with `tail -f output.log` in a separate terminal.

## Training
- `--config`: path to yaml file with hyperparameters (see [example](config.yaml.example))
- `--sync`: enables live-syncing to W&B
```bash
illum train --config "path_to_config" --sync 2> output.log
```

## Inference
To infer illuminant chromaticities on an image:
- `--image`: path to input image
- `--checkpoint`: path to model checkpoint to use
```bash
illum infer --image img.png --checkpoint checkpoints/checkpoint_13 2> output.log
```

## Acknowledgements
- [ ] TODO
