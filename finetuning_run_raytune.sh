#!/bin/bash

# Define the variables.
backbones=("resnet18")                                      # "resnet18" "resnet50"
train_rates=("5" "10")                                      # "0.01" "0.05" "0.1" "0.25" "0.5" "1.0" "25" "50" "75" "100"
downstream=("multiclass")                                   # "multiclass" "multilabel"
models=("Supervised")                                       # "Supervised" "BarlowTwins" "MoCov2" "SimCLR" "SimCLRv2" "SimSiam"
ini_weights=("random")                                      # "random" "imagenet"
balanced_dataset=(" ")                                      # " " "-bd" (" " for Ray Tune)
transfer_learning=("LP")                                    # "LP" "FT" "LP+FT" (only the first two for Ray Tune)
seeds=("42")                                                # Random: "42" (only this for Ray Tune) "5" "97"
epochs=100                                                  # 15 for Ray Tune; otherwise 100
learning_rate=0.01                                          # Not used for Ray Tune or --load_best_hyperparameters
save_every=5
batch_size=32
num_workers=4                                               # 2 for Ray Tune; otherwise 4
# more_options=""
# more_options="--verbose"
more_options="--ray_tune=gridsearch --grace_period=75 --num_samples_trials=3 --gpus_per_trial=1"
# more_options="--load_best_hyperparameters"

# Troubleshooting.
# export LOGLEVEL=INFO
# export NCCL_DEBUG=INFO
export RAY_PICKLE_VERBOSE_DEBUG=1

# Loop over the sbatch commands.
for b in "${backbones[@]}"; do
    for tr in "${train_rates[@]}"; do
        for d in "${downstream[@]}"; do
            for tl in "${transfer_learning[@]}"; do
                for s in "${seeds[@]}"; do
                    for m in "${models[@]}"; do
                        # Not considering balanced_dataset.
                        if [ "$m" = "Supervised" ]; then
                            for iw in "${ini_weights[@]}"; do
                                command_arg="torchrun finetuning.py $m $d -bn $b -tr $tr -e $epochs -lr $learning_rate -se $save_every -bs $batch_size -nw $num_workers -iw $iw -tl $tl -s $s $more_options"
                                raytune_args="--exp-name rt_ft --partition volta --num-nodes 1 --num-gpus 4 --load-env \"ssl-bsu-conda\" --command \"$command_arg\"" # --node aap04 
                                final_command="python slurm-launch.py $raytune_args"
                                echo $final_command
                                eval $final_command
                            done
                        # else
                        #     for bd in "${balanced_dataset[@]}"; do
                        #         command="torchrun finetuning.py $m $d -bn $b -tr $tr -e $epochs -lr $learning_rate -se $save_every -bs $batch_size -nw $num_workers -iw random -tl $tl -s $s $bd $more_options"
                        #         echo $command; $command # >> out_finetuning_$d_$m.out
                        #     done
                        fi
                    done
                done
            done
        done
    done
done
