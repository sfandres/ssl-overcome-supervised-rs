#!/bin/bash


# General resource requests.
#--------------------------------------------
#---> COMMON OPTIONS
#--------------------------------------------
#SBATCH --time=48:00:00                             # Job duration (72h is the limit).
#SBATCH --nodes=1                                   # Number of nodes.
#SBATCH --ntasks-per-node=1                         # Number of tasks per node.
#SBATCH --mem=0                                     # Real memory required per node.
#SBATCH --gres=gpu:1                                # The specified resources will be allocated to the job on each node.
#   #SBATCH --exclusive                                 # The job can not share nodes with other running jobs.

#--------------------------------------------
#---> TURGALIUM
#--------------------------------------------
#SBATCH --partition=volta                           # Request specific partition.
#   #SBATCH --exclude=aap[01-04],acp02                  # Explicitly exclude certain nodes from the resources granted to the job.
#   #SBATCH --cpus-per-task=4                           # Number of cpu-cores per task (>1 if multi-threaded tasks).
#   #SBATCH --exclude=acp[01-04]                        # Explicitly exclude certain nodes from the resources granted to the job.
#   #SBATCH --nodelist=aap03                            # acp04

#--------------------------------------------
#---> NGPU.URG
#--------------------------------------------
#   #SBATCH --partition=dios                            # Request specific partition (dios, dgx).

#--------------------------------------------
#---> UNUSED OPTIONS
#--------------------------------------------
#   #SBATCH --nodes=1                                   # Number of nodes.
#   #SBATCH --exclusive                                 # The job can not share nodes with other running jobs.
#   #SBATCH --gpus-per-node=4                           # Specify the number of GPUs required for the job on each node.


# Current exp configuration --> Imbalanced/Balanced
#--------------------------------------------
# --> INFO: Specific configurations for the experiments.
#--------------------------------------------
# * RayTune:            --ntasks=1, --gpus-per-node=2/4, --exclusive
# * DDP-4GPUs:          --ntasks=4, --gpus-per-node=4,   --exclusive
# * Imbalanced (1-GPU): --ntasks=1, --gpus-per-node=2/4,
# * Balanced (1-GPU):   --ntasks=1, --gpus-per-node=2/4
#--------------------------------------------


# Help function.
function display_help {
    echo "Usage: ./$0 MODEL BACKBONE EXPERIMENT"
    echo "Arguments:"
    echo "  MODEL         Specify the model ('BarlowTwins', 'MoCov2', 'SimCLR', 'SimCLRv2', or 'SimSiam')"
    echo "  BACKBONE      Specify the backbone ('resnet18' or 'resnet50')"
    echo "  EXPERIMENT    Type of experiment to carry out ('RayTune', 'DDP', 'Imbalanced', or 'Balanced')"
    echo "  -h, --help    Display this help message"
    exit 0
}

# Parse and check the first and second arguments.
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

else
    echo "Model:      $model"
    echo "Backbone:   $backbone_name"
fi

# Parse and check the third argument.
experiment=$3

if [[ -z $experiment ]]; then
    echo "Error: Selecting a type of experiment is required."
    display_help

elif [[ $experiment != "RayTune" && $experiment != "DDP" && $experiment != "Imbalanced" && $experiment != "Balanced" ]]; then
    echo "Error: Invalid type of experiment. Supported ones are 'RayTune', 'DDP', 'Imbalanced', and 'Balanced'."
    display_help

else
    echo "Experiment: $experiment"
fi

# Configure the target experiment.
if [[ $experiment == "RayTune" ]]; then
    dataset_ratio="(0.400,0.1500,0.4500)"
    epochs=12
    more_options="--ray_tune=gridsearch --grace_period=4 --num_samples_trials=1 --gpus_per_trial=1"
    export RAY_PICKLE_VERBOSE_DEBUG=1
    echo "RayTune experiment has been successfully set up!"

elif [[ $experiment == "DDP" ]]; then
    dataset_ratio="(0.900,0.0250,0.0750)"
    epochs=500
    more_options="--distributed"
    echo "DDP experiment has been successfully set up!"

elif [[ $experiment == "Imbalanced" ]]; then
    dataset_ratio="(0.900,0.0250,0.0750)"
    epochs=500
    more_options=""
    echo "Imbalanced experiment has been successfully set up!"

elif [[ $experiment == "Balanced" ]]; then
    dataset_ratio="(0.900,0.0250,0.0750)"
    epochs=500
    more_options="--balanced_dataset"
    echo "Balanced experiment has been successfully set up!"
fi

# Troubleshooting.
export LOGLEVEL=INFO
export NCCL_DEBUG=INFO

# Torchrun configuration for Slurm.
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo Nodes array:  $nodes_array
echo Head node:    $head_node
echo Head node IP: $head_node_ip

# Load virtual environment (turgalium).
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ssl-conda

# Load virtual environment (ngpu.ugr).
# export PATH="/opt/anaconda/anaconda3/bin:$PATH"
# export PATH="/opt/anaconda/bin:$PATH"
# eval "$(conda shell.bash hook)"
# conda activate /mnt/homeGPU/asanchez/ssl-conda
# export TFHUB_CACHE_DIR=.

# Define the general settings.
# input_data="/p/project/prcoe12"
# input_data="~/GitHub/datasets/"
dataset_name="Sentinel2GlobalLULC_SSL"
save_every=5
eval_every=50
batch_size=512                      # 128
ini_weights="random"
seed=42

# Run experiment (--standalone).
# $SLURM_GPUS_PER_TASK $SLURM_NTASKS $SLURM_CPUS_PER_TASK
# --input_data=$input_data \
# --partially_frozen \
command="torchrun --standalone \
--nnodes=$SLURM_JOB_NUM_NODES \
--nproc_per_node=$SLURM_NTASKS \
--rdzv_id=$RANDOM \
--rdzv_backend=c10d \
--rdzv_endpoint=$head_node_ip:29500 \
ssl_pretraining.py $model \
--backbone_name=$backbone_name \
--dataset_name=$dataset_name \
--dataset_ratio=$dataset_ratio \
--epochs=$epochs \
--save_every=$save_every \
--eval_every=$eval_every \
--batch_size=$batch_size \
--num_workers=4 \
--ini_weights=$ini_weights \
--seed=$seed \
${more_options}
"

# Show the chosen options.
echo "---------------------"
echo "Command executed: >> srun $command"
echo "---------------------"

# Run.
mail -s "Sbatch $model began" sfandres@unex.es <<< "Starting..."
srun $command
mail -s "Sbatch $model ended" sfandres@unex.es <<< "Completed!"
