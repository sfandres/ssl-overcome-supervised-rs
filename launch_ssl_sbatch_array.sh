#!/bin/bash
## Shebang.

## Resource request.
#SBATCH --nodes=1                                   ## Number of nodes.
#SBATCH --partition=volta                           ## Request specific partition.
#SBATCH --mem=128GB                                 ## Real memory required per node.
#SBATCH --wait-all-nodes=1                          ## Controls when the execution begins.
#SBATCH --time=72:00:00                             ## Job duration.
#SBATCH --gpus-per-node=2                           ## Number of GPUs on each node (Sergio: 4).
#SBATCH --job-name=uexssl_%A_%a                     ## Name of the job.
#SBATCH --output=uexssl_%A_%a.out                   ## Output file.
#SBATCH --mail-type=ALL                             ## (not working) Type of notification via email.
#SBATCH --mail-user=sfandres@unex.es                ## (not working) User to receive the email notification.
#SBATCH --array=0-4:1                               ## Run job arrays.

## Catch Slurm environment variables.
job_id=${SLURM_ARRAY_JOB_ID}
task_id=${SLURM_ARRAY_TASK_ID}

## Array of models.
array_models=("SimSiam" "SimCLR" "SimCLRv2" "BarlowTwins" "MoCov2")
model=${array_models[${task_id}]}

## Create a string for email subject.
email_info="job_id=${job_id} task_id=${task_id} ssl_model=${model}"

## Send email when job begin (two options).
## cat email_body.txt | /usr/bin/mail -s "Sbatch ${email_info} began" sfandres@unex.es
echo " " | /usr/bin/mail -s "Sbatch ${email_info} began" sfandres@unex.es

## Load virtual environment or module (not necessary).
## module load cuda/11.0.1
## source ~/lulc/lulc-venv/bin/activate
source ~/anaconda3/etc/profile.d/conda.sh
conda activate lulc2-conda

## Execute the Python script and pass the arguments.
srun python3 03_1-PyTorch-Sentinel-2_SSL_pretraining.py \
${model} \
--backbone_name=resnet18 \
--dataset_name=Sentinel2GlobalLULC_SSL \
--dataset_ratio=\(0.900,0.0250,0.0750\) \
--epochs=10 \
--batch_size=512 \
--ini_weights=random \
--cluster \
## --resume_training \
--reduced_dataset \
--ray_tune \
## --load_best_hyperparameters \
--num_samples_trials=1

## Send email when job ends.
## cat uexssl_${job_id}_${task_id}.out | /usr/bin/mail -s "Sbatch ${email_info} ended" sfandres@unex.es
/usr/bin/mail -a uexssl_${job_id}_${task_id}.out -s "Sbatch ${email_info} ended" sfandres@unex.es
