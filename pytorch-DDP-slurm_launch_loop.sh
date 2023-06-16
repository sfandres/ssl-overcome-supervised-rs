#!/bin/bash

# Help function.
function display_help {
    echo "Usage: $0 EXPERIMENT"
    echo "Arguments:"
    echo "  EXPERIMENT    Type of experiment to carry out ('RayTune', 'DDP', or 'Balanced')"
    echo "  -h, --help    Display this help message"
    exit 0
}

# Parse and check the first argument.
experiment=$1

if [[ $1 == "-h" || $1 == "--help" ]]; then
    display_help

elif [[ -z $experiment ]]; then
    echo "Error: Selecting a type of experiment is required."
    display_help

elif [[ $experiment != "RayTune" && $experiment != "DDP" && $experiment != "Balanced" ]]; then
    echo "Error: Invalid type of experiment. Supported ones are 'RayTune', 'DDP', and 'Balanced'."
    display_help

else
    echo "Experiment: $experiment"
fi

# Define the variables.
models=("BarlowTwins" "MoCov2" "SimCLR" "SimCLRv2" "SimSiam")
backbones=("resnet18")  # "resnet50"

# Loop over the sbatch commands.
for m in "${models[@]}"; do
    for b in "${backbones[@]}"; do

        # Run.
        command="sbatch -J ${m}_${b}_${experiment} -o out_${m}_${b}_${experiment}.out pytorch-DDP-slurm.sh ${m} ${b} ${experiment}"
        echo ${command}
        ${command}

        # Check if continue.
        read -p "Continue? (yes/enter to continue, no to exit): " input
        while [[ -n $input && $input != "yes" && $input != "no" ]]; do
            read -p "Invalid input. Please type 'yes'/enter or 'no': " input
        done
        if [[ $input == "no" ]]; then
            break
        fi

    done

    if [[ $input == "no" ]]; then
        break
    fi

done