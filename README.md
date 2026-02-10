# ViT for Illuminant estimation

A vision transformer (ViT) model for estimating illuminant color in images.
- Cube+ dataset for training and evaluation.
- Built with Flax

```
src/
└── data/
    ├── SimpleCube++/   dataset
    └── loader.py       data-loading helper
└── flax_illuminant_estimation/
    ├── model.py        ViT Definition
    ├── train.py        training loop
    └── eval.py         evaluation metrics
```
