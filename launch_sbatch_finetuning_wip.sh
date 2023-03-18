#!/bin/bash
## Shebang.

## Resource request.
#SBATCH --nodes=1                                   ## Number of nodes.
#SBATCH --partition=volta                           ## Request specific partition.
#SBATCH --wait-all-nodes=1                          ## Controls when the execution begins.
#SBATCH --time=3-24:00                              ## Job duration (Sergio: 24:00:00).
#SBATCH --gpus-per-node=2                           ## Number of GPUs on each node (Sergio: 4).
#SBATCH --job-name=uexsslfn_%A_%a                   ## Name of the job.
#SBATCH --output=uexsslfn_%A_%a.out                 ## Output file.
#SBATCH --mail-type=ALL                             ## (not working) Type of notification via email.
#SBATCH --mail-user=sfandres@unex.es                ## (not working) User to receive the email notification.
#SBATCH --array=0-2:1                               ## Run job arrays.

## Catch Slurm environment variables.
job_id=${SLURM_ARRAY_JOB_ID}
task_id=${SLURM_ARRAY_TASK_ID}

## Array of models.
array_models=("scratch" "imagenet" "ssl")
model=${array_models[${task_id}]}

## Create a string for email subject.
email_info="job_id=${job_id} task_id=${task_id} ssl_model=${model}"

## Send email when job begin (two options).
## cat email_body.txt | /usr/bin/mail -s "Sbatch ${email_info} began" sfandres@unex.es
echo " " | /usr/bin/mail -s "Sbatch ${email_info} began" sfandres@unex.es

## Load the Python module (not necessary).
## module load cuda/11.0.1

## Load virtual environment.
## source ~/lulc/lulc-venv/bin/activate
source ~/anaconda3/etc/profile.d/conda.sh
conda activate lulc-conda

## Execute the Python script and pass the arguments.
srun python3 03_2-PyTorch-Backbone_classifier.py \
${model} \
--dataset Sentinel2AndaluciaLULC \
--epochs 25 \
--batch_size 128 \
--show_fig False \
--cluster True
## srun python3 testing_GPU_PyTorch.py

## Send email when job ends.
## cat uexsslfn_${job_id}_${task_id}.out | /usr/bin/mail -s "Sbatch ${email_info} ended" sfandres@unex.es
/usr/bin/mail -a uexsslfn_${job_id}_${task_id}.out -s "Sbatch ${email_info} ended" sfandres@unex.es
