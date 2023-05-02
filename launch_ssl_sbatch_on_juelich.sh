#!/bin/bash
## Shebang.


## Resource request.
#SBATCH --nodes=4                                   ## Number of nodes.
#SBATCH --partition=dp-dam                          ## Request specific partition.
#SBATCH --mem=128GB                                 ## Real memory required per node.
#SBATCH --time=1100                                 ## Job duration.
#SBATCH --gpus-per-node=2                           ## Number of GPUs on each node (Sergio: 4).
#SBATCH --job-name=uexssl_%A_%a                     ## Name of the job.
#SBATCH --output=uexssl_%A_%a.out                   ## Output file.
#SBATCH --mail-type=ALL                             ## (not working) Type of notification via email.
#SBATCH --mail-user=sfandres@unex.es                ## (not working) User to receive the email notification.
##SBATCH --wait-all-nodes=1                         ## Controls when the execution begins.
#SBATCH --array=0-4:1                               ## Run job arrays.


function show_help {
    echo "Usage: $0 [OPTION]"
    echo "  -t, --training           Runs normal training."
    echo "  -r, --resume-training    Resumes the training from a previous saved checkpoint."
    echo "  -g, --gridsearch         Enables Ray Tune with tune.gridsearch to tune all the hyperparamenters."
    echo "  -l, --loguniform         Enables Ray Tune with tune.loguniform to tune the learning rate."
    echo "  -h, --help               Display the help message."
}

## Define settings for the experiments.
model_names=("SimSiam" "SimCLR" "SimCLRv2" "BarlowTwins" "MoCov2")
backbone_name="resnet18"  ## "resnet50"
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

elif [[ "$1" == "-g" ]] || [[ "$1" == "--gridsearch" ]]; then
    echo "You chose tune.gridsearch"
    epochs=5
    exp_options="--epochs=${epochs} --reduced_dataset --ray_tune=gridsearch --num_samples_trials=2"

elif [[ "$1" == "-l" ]] || [[ "$1" == "--loguniform" ]]; then
    echo "You chose tune.loguniform"
    epochs=10
    exp_options="--epochs=${epochs} --reduced_dataset --ray_tune=loguniform --num_samples_trials=10"

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
job_id=${SLURM_ARRAY_JOB_ID}
task_id=${SLURM_ARRAY_TASK_ID}

## Array of models.
model=${model_names[${task_id}]}

## Create a string for email subject.
email_info="job_id=${job_id} task_id=${task_id} ssl_model=${model}"

## Send email when job begin (two options).
echo " " | /usr/bin/mail -s "Sbatch ${email_info} began" sfandres@unex.es

## Load virtual environment.
source ~/anaconda3/etc/profile.d/conda.sh
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
/usr/bin/mail -a uexssl_${job_id}_${task_id}.out -s "Sbatch ${email_info} ended" sfandres@unex.es
