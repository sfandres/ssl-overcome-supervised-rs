#!/bin/bash
## Shebang.

## Resource request.
#SBATCH --nodes=1                                   ## Number of nodes.
#SBATCH --partition=volta                           ## Request specific partition.
#SBATCH --wait-all-nodes=1                          ## Controls when the execution begins.
#SBATCH --time=00:30:00                             ## Job duration (Sergio: 24:00:00).
#SBATCH --gpus-per-node=4                           ## Number of GPUs on each node (Sergio: 4).
#SBATCH --job-name=uexssl_%A_%a                     ## Name of the job.
#SBATCH --output=uexssl_%A_%a.out                   ## Output file.
#SBATCH --mail-type=ALL				                ## Type of notification via email.
#SBATCH --mail-user=sfandres@unex.es                ## User to receive the email notification.
#SBATCH --array=0-2:1                               ## Run job arrays.

# Array of models.
array_models=("simsiam" "simclr" "barlowtwins")
model=${array_models[${SLURM_ARRAY_TASK_ID}]}

## Send email when job begin.
cat mail.txt | /usr/bin/mail -s "Sbatch job array_task_id=${SLURM_ARRAY_TASK_ID} model=${model} began" sfandres@unex.es

## Load the Python module.
## module load cuda/11.0.1

## Load virtual environment.
source ~/lulc/lulc-venv/bin/activate

## Execute the Python script and pass the arguments.
## srun python3 script.py ${model}
srun python3 03_1-PyTorch-Sentinel-2_SSL_pretraining.py \
${model} \
--dataset Sentinel2GlobalLULC \
--balanced_dataset False \
--epochs 1 \
--batch_size 128 \
--ini_weights random \
--show_fig False \
--cluster True

## Send email when job ends.
cat uexssl_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out | /usr/bin/mail -s "Sbatch job array_task_id=${SLURM_ARRAY_TASK_ID} model=${model} ended" sfandres@unex.es
