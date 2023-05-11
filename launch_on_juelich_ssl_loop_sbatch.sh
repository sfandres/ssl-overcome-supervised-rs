#!/bin/bash


function show_help {
    echo "Usage: $0 [OPTION]"
    echo "  -t, --training           Runs normal training."
    echo "  -r, --resume-training    Resumes the training from a previous saved checkpoint."
    echo "  -h, --help               Display the help message."
}


## Catch the arguments.
if [[ "$1" == "-t" ]] || [[ "$1" == "--training" ]]; then
    echo "You chose normal training using $1"

elif [[ "$1" == "-r" ]] || [[ "$1" == "--resume-training" ]]; then
    echo "You chose resume training using $1"

elif [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    show_help
    exit 0

else
    echo "Invalid option. Use -h or --help to display available options."
    exit 1
fi

## Define the variables.
models=("SimSiam" "SimCLR" "SimCLRv2" "BarlowTwins" "MoCov2")
backbones=("resnet18" "resnet50")
node=60

## Loop over the sbatch commands.
for m in "${models[@]}"; do
    for b in "${backbones[@]}"; do
        echo sbatch -p dp-esb -w dp-esb[${node}] -t 1080 -J ${b}_${m} -o ${b}_${m}.out launch_on_juelich_ssl_sbatch.sh $1 ${b} ${m}
        sbatch -p dp-esb -w dp-esb[${node}] -t 1080 -J ${b}_${m} -o ${b}_${m}.out launch_on_juelich_ssl_sbatch.sh $1 ${b} ${m}
        (( node++ ))
    done
done