#!/bin/bash

# Resource request.

# Juelich.
##SBATCH --cpus-per-task=1                           # Number of cpu-cores per task (>1 if multi-threaded tasks).
##SBATCH --gpus-per-task=1                           # Number of GPUs per task.
##SBATCH --mail-type=ALL                             # Type of notification via email.
##SBATCH --mail-user=sfandres@unex.es                # User to receive the email notification.

# Turgalium.
#SBATCH --nodes=1                                   # Number of nodes.
#SBATCH --ntasks=1                                  # Number of tasks.
#SBATCH --partition=volta                           # Request specific partition.
#SBATCH --time=72:00:00                             # Job duration.
#SBATCH --cpus-per-task=4                           # Number of cpu-cores per task (>1 if multi-threaded tasks).
#SBATCH --gpus-per-node=2                           # Min. number of GPUs on each node.
#SBATCH --mail-type=ALL                             # Type of notification via email.
#SBATCH --mail-user=sfandres@unex.es                # User to receive the email notification.


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

# Torchrun configuration for Slurm.
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo Nodes array:  $nodes_array
echo Head node:    $head_node
echo Head node IP: $head_node_ip

# Troubleshooting.
export LOGLEVEL=INFO
export NCCL_DEBUG=INFO

# Load virtual environment.
# source /p/project/joaiml/hetgrad/anaconda3/etc/profile.d/conda.sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate lulc2-conda

# Define settings for the experiments.
# input_data="/p/project/prcoe12"
dataset_name="Sentinel2GlobalLULC_SSL"
dataset_ratio="(0.900,0.0250,0.0750)"
epochs=250
save_every=10
batch_size=128
num_workers=4
ini_weights="random"

# Run experiment (--standalone).
# $SLURM_GPUS_PER_TASK $SLURM_NTASKS
srun torchrun --standalone \
--nnodes $SLURM_JOB_NUM_NODES \
--nproc_per_node $SLURM_NTASKS \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:29500 \
pytorch-DDP-Sentinel-2_SSL_pretraining.py $model \
--backbone_name $backbone_name \
--dataset_name $dataset_name \
--dataset_ratio $dataset_ratio \
--epochs $epochs \
--save_every $save_every \
--batch_size $batch_size \
--num_workers $num_workers \
--ini_weights $ini_weights \
--balanced_dataset

# --balanced_dataset \
# --input_data $input_data \
# --distributed \
