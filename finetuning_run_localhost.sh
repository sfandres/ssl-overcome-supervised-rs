#!/bin/bash

# Define the variables.
downstream=("multiclass" "multilabel")
train_rates=("0.05" "0.1" "0.25" "0.5")  # "1.0"
learning_rates=("0.025" "0.05" "0.1")
models=("Random" "Imagenet" "BarlowTwins" "MoCov2" "SimCLR" "SimCLRv2" "SimSiam")
backbones=("resnet50" "resnet18")

# Loop over the sbatch commands.
for d in "${downstream[@]}"; do
    for tr in "${train_rates[@]}"; do
        for lr in "${learning_rates[@]}"; do
            for m in "${models[@]}"; do
                for b in "${backbones[@]}"; do
                    echo torchrun finetuning.py $m $d -bn $b -tr $tr -e 50 -lr $lr -bs 64 -nw 4
                    torchrun finetuning.py $m $d -bn $b -tr $tr -e 50 -lr $lr -bs 64 -nw 4
                done
            done
        done
    done
done
