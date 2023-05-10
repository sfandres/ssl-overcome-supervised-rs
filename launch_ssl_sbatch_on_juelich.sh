#!/bin/bash
## Shebang.


## Resource request.
#SBATCH --nodes=1                                   ## Number of nodes.
#SBATCH --ntasks=1                                  ## Number of tasks.
#SBATCH --gpus-per-node=1                           ## Number of GPUs on each node (Sergio: 4).
#SBATCH --job-name=uexssl_%A                        ## Name of the job.
#SBATCH --output=uexssl_%A.out                      ## Output file.
#SBATCH --mail-type=ALL                             ## (not working) Type of notification via email.
#SBATCH --mail-user=sfandres@unex.es                ## (not working) User to receive the email notification.


function show_help {
    echo "Usage: $0 [OPTION] [BACKBONE] [MODEL]"
    echo "  -t, --training           Runs normal training."
    echo "  -r, --resume-training    Resumes the training from a previous saved checkpoint."
    echo "  -h, --help               Display the help message."
}

## Define settings for the experiments.
backbone_name=$2
model=$3
dataset_name="Sentinel2GlobalLULC_SSL"
dataset_ratio="(0.900,0.0250,0.0750)"
epochs=100
if [ "${backbone_name}" == "resnet50" ]; then
    batch_size=128
else
    batch_size=512
fi
ini_weights="random"

## Catch the arguments.
if [[ "$1" == "-t" ]] || [[ "$1" == "--training" ]]; then
    echo "You chose normal training"
    exp_options="--epochs=${epochs}"

elif [[ "$1" == "-r" ]] || [[ "$1" == "--resume-training" ]]; then
    echo "You chose resume training"
    exp_options="--epochs=${epochs} --resume_training"

elif [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    show_help
    exit 0

else
    echo "Invalid option. Use -h or --help to display available options."
    exit 1
fi

## Show the chosen options.
echo "Specific options of the current experiment: ${backbone_name} ${exp_options}"

## Catch Slurm environment variables.
job_id=${SLURM_JOB_ID}

## Create a string for email subject.
email_info="job_id=${job_id} ssl_backbone=${backbone_name} ssl_model=${model}"

## Send email when job begin (two options).
echo " " | /usr/bin/mail -s "Sbatch ${email_info} began" sfandres@unex.es

## Load virtual environment.
source /p/project/joaiml/hetgrad/anaconda3/etc/profile.d/conda.sh
conda activate lulc2-conda

## Execute the Python script and pass the arguments.
srun python3 03_1-PyTorch-Sentinel-2_SSL_pretraining.py \
${model} \
--backbone_name=${backbone_name} \
--dataset_name=${dataset_name} \
--dataset_ratio=${dataset_ratio} \
--batch_size=${batch_size} \
--ini_weights=${ini_weights} \
--cluster \
${exp_options}

## Send email when job ends.
## cat uexssl_${job_id}_${task_id}.out | /usr/bin/mail -s "Sbatch ${email_info} ended" sfandres@unex.es
/usr/bin/mail -a uexssl_${job_id}.out -s "Sbatch ${email_info} ended" sfandres@unex.es
