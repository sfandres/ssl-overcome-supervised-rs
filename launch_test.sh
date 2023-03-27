#!/bin/bash
## Shebang.

## Array of models.
array_models=("simsiam" "simclr" "barlowtwins")
model=${array_models[1]}

## Execute the Python script and pass the arguments.
python3 03_1-PyTorch-Sentinel-2_SSL_pretraining.py \
${model} \
--dataset_name=Sentinel2GlobalLULC_SSL \
--dataset_ratio=\(0.020,0.0196,0.9604\) \
--epochs=25 \
--batch_size=64 \
--ini_weights=random \
--cluster
