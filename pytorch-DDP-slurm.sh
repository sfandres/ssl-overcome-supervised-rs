#!/bin/bash


#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=ddp
#SBATCH --output=ddp_%j.out


function show_help {
    echo "Usage: $0 [OPTION] [MODEL] [BACKBONE]"
    echo "  -t, --training           Runs normal training."
    echo "  -r, --resume-training    Resumes the training from a previous saved checkpoint."
    echo "  -h, --help               Display the help message."
}


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

# Catch the arguments.
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

# Load virtual environment.
source /p/project/joaiml/hetgrad/anaconda3/etc/profile.d/conda.sh
conda activate lulc2-conda

# Define settings for the experiments.
model=$2
backbone_name=$3
input_data="/p/project/prcoe12"
dataset_name="Sentinel2GlobalLULC_SSL"
dataset_ratio="(0.900,0.0250,0.0750)"
epochs=5
save_every=3
if [ "${backbone_name}" == "resnet50" ]; then
    batch_size=32  ##128
else
    batch_size=64  ##512
fi
num_workers=1
ini_weights="random"

srun torchrun \
--nnodes 3 \
--nproc_per_node 1 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:29500 \
pytorch-DDP-Sentinel-2_SSL_pretraining.py $model \
--backbone_name $backbone_name \
--input_data $input_data \
--dataset_name $dataset_name \
--dataset_ratio $dataset_ratio \
--epochs $epochs \
--save_every $save_every \
--batch_size $batch_size \
--num_workers $num_workers \
--ini_weights $ini_weights \
--distributed \
$exp_options

# srun torchrun \
# --standalone \
# --nproc_per_node 2 \
