#!/bin/bash
# Shebang.

# Help function.
function display_help {
    echo "Usage: ./$0 MODEL BACKBONE"
    echo "Arguments:"
    echo "  MODEL         Specify the model ('BarlowTwins', 'MoCov2', 'SimCLR', 'SimCLRv2', or 'SimSiam')"
    echo "  BACKBONE      Specify the backbone ('resnet18' or 'resnet50')"
    echo "  -h, --help    Display this help message"
    exit 0
}

# Parse arguments.
model=$1
backbone_name=$2

if [[ $1 == "-h" || $1 == "--help" ]]; then
    display_help

elif [[ -z $model || -z $backbone_name ]]; then
    echo "Error: Both model and backbone arguments are required."
    display_help

elif [[ $model != "BarlowTwins" && $model != "MoCov2" && $model != "SimCLR" && $model != "SimCLRv2" && $model != "SimSiam" ]]; then
    echo "Error: Invalid model. Supported models are 'BarlowTwins', 'MoCov2', 'SimCLR', 'SimCLRv2', and 'SimSiam'."
    display_help

elif [[ $backbone_name != "resnet18" && $backbone_name != "resnet50" ]]; then
    echo "Error: Invalid backbone. Supported backbones are 'resnet18' and 'resnet50'."
    display_help
fi

# Env variables.
export RAY_PICKLE_VERBOSE_DEBUG=1

# Define settings for the experiments.
dataset_name="Sentinel2GlobalLULC_SSL"
dataset_ratio="(0.020,0.0196,0.9604)"
# dataset_ratio="(0.400,0.1500,0.4500)"
# dataset_ratio="(0.900,0.0250,0.0750)"
epochs=4
save_every=2
batch_size=32
num_workers=2
ini_weights="random"

# Python script to be executed with the options and flags.
script="--standalone --nnodes=1 --nproc_per_node=1 pytorch-DDP-Sentinel-2_SSL_pretraining.py $model \
--backbone_name=$backbone_name \
--dataset_name=$dataset_name \
--dataset_ratio=$dataset_ratio \
--epochs=$epochs \
--save_every=$save_every \
--batch_size=$batch_size \
--num_workers=$num_workers \
--ini_weights=$ini_weights \
"

# --ray_tune=gridsearch \
# --grace_period=1 \
# --num_samples_trials=1 \
# --gpus_per_trial=1

# --partially_frozen \
# --reduced_dataset \
# --balanced_dataset \
# --distributed \
# --verbose \
# --show \
# --input_data="" \

# Show the chosen options.
echo "---------------------"
echo "Command executed: >> torchrun $script"
echo "---------------------"

# Execute the script.
# torchrun $script > out_pretraining_localhost_${model}_${backbone_name}.out
torchrun $script
