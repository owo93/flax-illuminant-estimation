# ViT for Illuminant estimation

A regressive implementation of Vision Transformer for predicting single illuminant color in images trained on the [SimpleCube++](https://github.com/Visillect/CubePlusPlus)  dataset, built with Flax NNX

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

### Training
To train model with given hyperparameters:
```bash
uv run illum train --config <path_to_config>
```
or resume from a checkpoint:
```bash
uv run illum train --resume <path_to_checkpoint>
```
- each run is automatically logged to W&B
- restored runs are also treated as new runs


### Inference
To infer illuminant chromaticities on an image:
```bash
uv run illum infer <path_to_img> --checkpoint <path_to_checkpoint>
```
