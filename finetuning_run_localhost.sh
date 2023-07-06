#!/bin/bash

# Define the variables.
backbones=("resnet18")                                  # "resnet18" "resnet50"
train_rates=("0.01" "0.05" "0.1" "0.25" "0.5" "1.0")    # "0.01" "0.05" "0.1" "0.25" "0.5" "1.0"
downstream=("multiclass" "multilabel")                  # "multiclass" "multilabel"
learning_rates=("0.01")                                 # Not used for Ray Tune or --load_best_hyperparameters
models=("Supervised")                                   # "BarlowTwins" "MoCov2" "SimCLR" "SimCLRv2" "SimSiam"
ini_weights=("random" "imagenet")                       # "random" "imagenet"
balanced_dataset=(" " "-bd")                            # " " "-bd"
transfer_learning=("LP" "FT" "LP+FT")                   # "LP" "FT" "LP+FT"
epochs=12
batch_size=64
num_workers=2                                           # 4
seed=42
# more_options=""
# more_options="--ray_tune=gridsearch --grace_period=4 --num_samples_trials=3 --gpus_per_trial=1"
more_options="--load_best_hyperparameters"

# Troubleshooting.
# export LOGLEVEL=INFO
# export NCCL_DEBUG=INFO
export RAY_PICKLE_VERBOSE_DEBUG=1

# Loop over the sbatch commands.
for b in "${backbones[@]}"; do
    for tr in "${train_rates[@]}"; do
        for d in "${downstream[@]}"; do
            for lr in "${learning_rates[@]}"; do
                for tl in "${transfer_learning[@]}"; do
                    for m in "${models[@]}"; do
                        # Not considering balanced_dataset.
                        if [ "$m" = "Supervised" ]; then
                            for iw in "${ini_weights[@]}"; do
                                command="torchrun finetuning.py $m $d -bn $b -tr $tr -e $epochs -lr $lr -bs $batch_size -nw $num_workers -iw $iw -tl $tl -s $seed $more_options"
                                echo $command; $command >> out_finetuning_$d.out
                            done
                        else
                            for bd in "${balanced_dataset[@]}"; do
                                command="torchrun finetuning.py $m $d -bn $b -tr $tr -e $epochs -lr $lr -bs $batch_size -nw $num_workers -iw random -tl $tl -s $seed $bd $more_options"
                                echo $command; $command >> out_finetuning_$d.out
                            done
                        fi
                    done
                done
            done
        done
    done
done
