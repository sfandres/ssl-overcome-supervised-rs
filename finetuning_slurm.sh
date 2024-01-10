#!/bin/bash


# General resource requests.
#--------------------------------------------
#---> COMMON OPTIONS
#--------------------------------------------
#SBATCH --time=48:00:00                             # Job duration (72h is the limit).
#SBATCH --ntasks=1                                  # Number of tasks.
#   #SBATCH --mem=0                                     # Real memory required per node.
#SBATCH --gres=gpu:1                                # The specified resources will be allocated to the job on each node.
#   #SBATCH --cpus-per-task=4                           # Number of cpu-cores per task (>1 if multi-threaded tasks).

#--------------------------------------------
#---> TURGALIUM
#--------------------------------------------
#SBATCH --partition=volta                           # Request specific partition.
#   #SBATCH --exclude=acp[02],aap[01-04]             # Explicitly exclude certain nodes from the resources granted to the job.

#--------------------------------------------
#---> NGPU.URG
#--------------------------------------------
#   #SBATCH --partition=dios                            # Request specific partition (dios, dgx).

#--------------------------------------------
#---> UNUSED OPTIONS
#--------------------------------------------
#   #SBATCH --nodes=1                                   # Number of nodes.
#   #SBATCH --gpus-per-node=2                           # Specify the number of GPUs required for the job on each node.
#   #SBATCH --exclusive                                 # The job can not share nodes with other running jobs.


# Current exp configuration --> Imbalanced/Balanced
#--------------------------------------------
# --> INFO: Specific configurations for the experiments.
#--------------------------------------------
# * RayTune:            --ntasks=1, --gpus-per-node=2/4, --exclusive
# * DDP-4GPUs:          --ntasks=4, --gpus-per-node=4,   --exclusive
# * Imbalanced (1-GPU): --ntasks=1, --gpus-per-node=2/4,
# * Balanced (1-GPU):   --ntasks=1, --gpus-per-node=2/4
#--------------------------------------------


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
conda activate ssl-bsu-conda

# Load virtual environment (ngpu.ugr).
# export PATH="/opt/anaconda/anaconda3/bin:$PATH"
# export PATH="/opt/anaconda/bin:$PATH"
# eval "$(conda shell.bash hook)"
# conda activate /mnt/homeGPU/asanchez/ssl-conda
# export TFHUB_CACHE_DIR=.

# Define the general settings.
command="./finetuning_run_localhost.sh"

# Show the chosen options.
echo "---------------------"
echo "Command executed: >> srun $command"
echo "---------------------"

# Run.
mail -s "Sbatch $model began" sfandres@unex.es <<< "Starting..."
srun $command
mail -s "Sbatch $model ended" sfandres@unex.es <<< "Completed!"
