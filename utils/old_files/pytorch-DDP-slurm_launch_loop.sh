#!/bin/bash

# Help function.
function display_help {
    echo "Usage: $0 NODE"
    echo "Arguments:"
    echo "  NODE          Starting target node (a number between 1 and 75)"
    echo "  -h, --help    Display this help message"
    exit 0
}

# Function to include a leading 0 if num<10.
format_number() {
    local num=$1
    local formatted_num

    # Check if num is less than 10
    if ((num < 10)); then
        # Add a leading zero to the number and store it in a string
        formatted_num="0$num"
    else
        # Number is 10 or greater, no leading zero required
        formatted_num="$num"
    fi

    echo "$formatted_num"
}

# Parse arguments.
node=$1
echo "Starting node (4-GPUs): $node"

if [[ $1 == "-h" || $1 == "--help" ]]; then
    display_help

elif [[ -z $node ]]; then
    echo "Error: Starting node is required."
    display_help
elif [[ ! $node -ge 1 || ! $node -le 75 ]]; then
    echo "Error: Not a number or a value in the rage 1-75."
    display_help
fi

# Define the variables.
models=("BarlowTwins" "MoCov2" "SimCLR" "SimCLRv2" "SimSiam")
backbones=("resnet18" "resnet50")

# Loop over the sbatch commands.
for m in "${models[@]}"; do
    for b in "${backbones[@]}"; do

        # Range of target nodes.
        target_nodes="dp-esb[$(format_number $node)-$(format_number $(($node + 3)))]"

        # Run.
        echo sbatch -p dp-esb -w $target_nodes -t 1080 -J ${m}_${b} -o out_${m}_${b}.out pytorch-DDP-slurm.sh ${m} ${b}
        sbatch -p dp-esb -w $target_nodes -t 1080 -J ${m}_${b} -o out_${m}_${b}.out pytorch-DDP-slurm.sh ${m} ${b}

        # Check if continue (drain/alloc nodes in between).
        read -p "Continue? (yes/enter to continue, no to exit): " input
        while [[ -n $input && $input != "yes" && $input != "no" ]]; do
            read -p "Invalid input. Please type 'yes'/enter or 'no': " input
        done
        if [[ $input == "no" ]]; then
            break
        fi

        # Increment the starting node by 4.
        ((node += 4))
    done

    if [[ $input == "no" ]]; then
        break
    fi

done