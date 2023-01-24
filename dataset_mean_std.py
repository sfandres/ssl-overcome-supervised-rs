"""Writes to a txt the mean and std of each dataset
   so that the data can be normalized afterwards.

Usage:
    ./dataset_mean_std.py <input_dir> --batch_size <batch_size>

Author:
    A.J. Sanchez-Fernandez - 24/01/2023
"""


import torch
import torchvision
import utils
import argparse
import sys
import os


def create_arg_parser():
    """Creates and returns the ArgumentParser object."""

    # Parser creation and description.
    parser = argparse.ArgumentParser(
        description=('Python script that writes to a txt the mean and std of each '
                     'dataset so that the data can be normalized afterwards.')
    )

    parser.add_argument(
        'input_dir',
        type=str,
        help='Path to the input directory where the datasets are stored.'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size (default = 64) for the PyTorch dataloader to compute the mean and std.'
    )

    return parser


# Parser (get arguments).
if __name__ == "__main__":

    # Create parser and get arguments.
    parser = create_arg_parser()
    args = parser.parse_args(sys.argv[1:])

# Show info.
datasets_dir = args.input_dir
print(f'\nPath to the input datasets: {datasets_dir}')

batch_size = args.batch_size
print(f'\nPyTorch dataloader batch size: {batch_size}')

# Reproducibility.
exp = utils.Experiment()
exp.reproducibility()
print(f'\nSeed used: {exp.seed}')

# Get the subsets with full path.
data_dirs = utils.listdir_fullpath(datasets_dir)

# Leave out unwanted subsets (0_Raw).
# data_dirs = data_dirs[1:]
print('\nInput directories:')
for dirs in data_dirs:
    print(f'- {dirs}')

# Initialization.
splits = ['train', 'val', 'test']
filename = 'dataset_mean_std.txt'

# Loop over the datasets (except raw and clothing).
for data_dir in data_dirs:

    # Create path to the txt file.
    filepath = os.path.join(data_dir, filename)

    # Removing the old txt file if exists.
    print(f'\n{data_dir}:')
    if os.path.exists(filepath):
        os.remove(filepath)
        print('Existing txt file found and removed. New empty txt file created.')
    else:
        print('New empty txt file created.')

    # Creating/opening the file.
    f = open(filepath, 'w')

    # Loading the datasets into a dic.
    datasets = {x: torchvision.datasets.ImageFolder(
        os.path.join(data_dir, x),
        transform=torchvision.transforms.ToTensor()
    ) for x in splits}

    # Creating the dataloaders into a dic.
    dataloaders = {x: torch.utils.data.DataLoader(
        datasets[x],
        batch_size=batch_size,
        worker_init_fn=exp.seed_worker,
        generator=exp.g
    ) for x in splits}

    # Loop over the train, val, and test datasets.
    for x in splits:

        # Computation.
        print(f'{x}/')
        print(f'Samples to be processed: '
              f'{len(dataloaders[x].dataset)}')
        mean, std = utils.get_mean_std_dataloader(dataloaders[x])
        print(mean)
        print(std)

        # Write to file.
        f.write(f'{x}\n')
        f.write(f'{mean}\n')
        f.write(f'{std}\n')

    # Close file and print an empty line.
    f.close()
    print('')