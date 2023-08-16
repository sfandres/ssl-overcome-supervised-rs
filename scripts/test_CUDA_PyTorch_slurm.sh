#!/bin/bash
## Shebang.

## Resource request.
#SBATCH --nodes=1                                   ## Number of nodes.
#SBATCH --partition=volta                           ## Request specific partition.
#SBATCH --mem=16GB                                  ## Real memory required per node.
#SBATCH --wait-all-nodes=1                          ## Controls when the execution begins.
#SBATCH --time=00:10:00                             ## Job duration.
#SBATCH --gpus-per-node=2                           ## Number of GPUs on each node (Sergio: 4).
#SBATCH --job-name=test_cuda_pytorch_%A             ## Name of the job.
#SBATCH --output=test_cuda_pytorch_%A.out           ## Output file.
#SBATCH --mail-type=ALL                             ## (not working) Type of notification via email.
#SBATCH --mail-user=sfandres@unex.es                ## (not working) User to receive the email notification.

## Catch Slurm environment variables.
job_id=${SLURM_JOB_ID}

## Create a string for email subject.
email_info="job_id=${job_id} test CUDA PyTorch"

## Send email when job begin (two options).
echo " " | /usr/bin/mail -s "Sbatch ${email_info} began" sfandres@unex.es

## Load virtual environment or module (not necessary).
## module load cuda/11.0.1
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ssl-conda

## Execute the Python script and pass the arguments.
srun python3 test_CUDA_PyTorch.py

## Send email when job ends.
cat test_cuda_pytorch_${job_id}.out | /usr/bin/mail -s "Sbatch ${email_info} ended" sfandres@unex.es
