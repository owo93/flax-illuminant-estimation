# ViT for Illuminant estimation

An implementation of Vision Transformer for predicting single illuminant color in images trained on the [SimpleCube++](https://github.com/Visillect/CubePlusPlus)  dataset, built with Flax NNX

```
src/
├── config.yaml          hyperparameters
├── data/
│   ├── SimpleCube++     dataset
│   └── loader.py
└── flax_illuminant_estimation/
    ├── config.py        trainer & model config
    ├── model.py         ViT definition
    ├── train.py         training loop
    └── infer.py         inference script
```

## Usage
By default, [`rich`](https://github.com/textualize/rich) terminal output is logged to `stdout` and `logging` log messages to `stderr`. 

### Training
To train model with given hyperparameters:
```bash
illum train --config "path_to_config" 2 > logs
```

### Inference
To infer illuminant chromaticities on an image:
```bash
illum infer --image "path_to_image" --checkpoint "path_to_checkpoint" 2 > logs
```
