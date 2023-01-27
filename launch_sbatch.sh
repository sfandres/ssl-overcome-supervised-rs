#!/bin/bash
## Shebang.

## Resource request.
#SBATCH --nodes=1                                   ## Number of nodes.
#SBATCH --partition=volta                           ## Request specific partition.
#SBATCH --wait-all-nodes=1                          ## Controls when the execution begins.
#SBATCH --time=00:10:00                             ## Job duration (Sergio: 24:00:00).
#SBATCH --gpus-per-node=2                           ## Number of GPUs on each node (Sergio: 4).
#SBATCH --job-name=uexssl                           ## Name of the job.
#SBATCH --output=uexssl.out                         ## Output file.
#SBATCH --mail-type=ALL				    ## Type of notification via email.
#SBATCH --mail-user=sfandres@unex.es                ## User to receive the email notification.

## Send email when job begin.
cat mail.txt | /usr/bin/mail -s "Sbatch job began" sfandres@unex.es

## Load the Python module.
## module load cuda/11.0.1

## Init virtual environment.
source ~/lulc/lulc-venv/bin/activate

## Execute the Python script and pass the arguments.
srun python3 script.py 10

## Send email when job ends.
cat uexssl.out | /usr/bin/mail -s "Sbatch job ended" sfandres@unex.es
