# ViT for Illuminant estimation

A vision transformer (ViT) model for estimating illuminant color in images.
- Cube+ dataset for training and evaluation.
- Built with Flax

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

## Commands
To train model with given hyperparameters:
```bash
uv run illum train --config <path_to_config>
```

To infer illuminant chromaticities on an image:
```bash
uv run illum infer <path_to_img> --checkpoint <path_to_checkpoint>
```
