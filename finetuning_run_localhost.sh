#!/bin/bash

# Define the variables.
downstream=("multiclass" "multilabel")
train_rates=("1." ".25" ".1" ".05")
learning_rates=("0.1" "0.01" "0.01")
models=("Random" "Imagenet" "BarlowTwins" "MoCov2" "SimCLR" "SimCLRv2" "SimSiam")
backbones=("resnet18")  # "resnet50")

# Loop over the sbatch commands.
for d in "${downstream[@]}"; do
    for tr in "${train_rates[@]}"; do
        for lr in "${learning_rates[@]}"; do
            for m in "${models[@]}"; do
                for b in "${backbones[@]}"; do
                    echo torchrun finetuning.py $m $d -nw 4 -bs 64 -e 50 -lr $lr --dataset_train_pc=$tr
                    torchrun finetuning.py $m $d -nw 4 -bs 64 -e 50 -lr $lr --dataset_train_pc=$tr
                done
            done
        done
    done
done
