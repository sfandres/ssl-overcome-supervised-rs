#!/bin/bash
## Shebang.

## Array of models.
array_models=("simsiam" "simclr" "barlowtwins")
model=${array_models[1]}

## Execute the Python script and pass the arguments.
python3 03_1-PyTorch-Sentinel-2_SSL_pretraining.py \
${model} \
--backbone_name=resnet18 \
--dataset_name=Sentinel2GlobalLULC_SSL \
--dataset_ratio=\(0.400,0.1500,0.4500\) \
--epochs=100 \
--batch_size=64 \
--ini_weights=random \
--cluster
