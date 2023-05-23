#!/bin/bash
## Shebang.


function show_help {
    echo "Usage: $0 [OPTION] [MODEL] [BACKBONE]"
    echo "  -t, --training           Runs normal training."
    echo "  -r, --resume-training    Resumes the training from a previous saved checkpoint."
    echo "  -h, --help               Display the help message."
}


## Catch the arguments.
if [[ "$1" == "-t" ]] || [[ "$1" == "--training" ]]; then
    echo "You chose normal training"
    exp_options=""

elif [[ "$1" == "-r" ]] || [[ "$1" == "--resume-training" ]]; then
    echo "You chose resume training"
    exp_options="--resume_training"

elif [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    show_help
    exit 0

else
    echo "Invalid option. Use -h or --help to display available options."
    exit 1
fi

if [ -z "$2" ] || [ -z "$3" ]; then
    echo "Second (model) or third (backbone) argument is empty."
    show_help
    exit 0
fi

## Define settings for the experiments.
model=$2
backbone_name=$3
## input_data=""
dataset_name="Sentinel2GlobalLULC_SSL"
dataset_ratio="(0.900,0.0250,0.0750)"
epochs=300
save_every=3
if [ "${backbone_name}" == "resnet50" ]; then
    batch_size=32  ##128
else
    batch_size=64  ##512
fi
num_workers=4
ini_weights="random"

## Python script to be executed with the options and flags.
script="--standalone --nproc_per_node=1 pytorch-DDP-Sentinel-2_SSL_pretraining.py $model \
--backbone_name=$backbone_name \
--dataset_name=$dataset_name \
--dataset_ratio=$dataset_ratio \
--epochs=$epochs \
--save_every=$save_every \
--batch_size=$batch_size \
--num_workers=$num_workers \
--ini_weights=$ini_weights \
--distributed \
$exp_options"

## Show the chosen options.
echo "---------------------"
echo "Specific options of the current experiment: $model $backbone_name $exp_options"
echo "Command executed:"
echo ">> torchrun $script"
echo "---------------------"

## Execute the script.
torchrun $script
