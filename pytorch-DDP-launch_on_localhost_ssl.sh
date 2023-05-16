#!/bin/bash
## Shebang.

## Array of models.
array_models=("SimSiam" "SimCLR" "SimCLRv2" "BarlowTwins" "MoCov2")
model=${array_models[3]}

## Execution options.
options="${model} \
--backbone_name=resnet18 \
--dataset_name=Sentinel2GlobalLULC_SSL \
--dataset_ratio=(0.900,0.0250,0.0750) \
--epochs=5 \
--batch_size=64 \
--num_workers=4 \
--ini_weights=random \
--cluster "
##--reduced_dataset"

echo "---------------------"
echo "Command executed: python3 pytorch-DDP-Sentinel-2_SSL_pretraining.py $options"
echo "---------------------"

## Execute the Python script and pass the arguments.
python3 pytorch-DDP-Sentinel-2_SSL_pretraining.py $options
