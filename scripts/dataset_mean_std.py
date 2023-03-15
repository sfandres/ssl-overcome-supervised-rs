"""Writes to a txt the mean and std values of each dataset
   so that the data can be normalized afterwards. The aim is
   to calculate the normalization values of each division
   (ratios) automatically all at once.

Usage: dataset_mean_std.py [-h] [--batch_size BATCH_SIZE] input_path

Python script that writes to a txt the mean and std of each dataset so that the data can be normalized afterwards.

positional arguments:
  input_path            path to the input directory where the datasets are stored (parent directory).

options:
  -h, --help               show this help message and exit
  --batch_size BATCH_SIZE  batch size (default = 64) for the PyTorch dataloader to compute the mean and std values.

Author:
    A.J. Sanchez-Fernandez - 15/03/2023
"""


import sys
import os
import argparse
import torch
import torchvision
sys.path.append("../")
from utils.computation import Experiment
from utils.dataset import (list_subdirs_fullpath,
                           get_mean_std_dataloader)


def create_arg_parser():
    """Creates and returns the ArgumentParser object."""

    # Parser creation and description.
    parser = argparse.ArgumentParser(
        description=('Python script that writes to a txt the mean and std of each '
                     'dataset so that the data can be normalized afterwards.')
    )

    parser.add_argument(
        'input_path',
        type=str,
        help='path to the input directory where the datasets are stored (parent directory).'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='batch size (default = 64) for the PyTorch dataloader to compute the mean and std values.'
    )

    return parser

def main():
    """Main function."""

    # Create parser and get arguments.
    parser = create_arg_parser()
    args = parser.parse_args(sys.argv[1:])

    # Show info.
    datasets_dir = args.input_path
    print(f'\nPath to the input datasets:\t{datasets_dir}')

    batch_size = args.batch_size
    print(f'PyTorch dataloader batch size:\t{batch_size}')

    # Reproducibility.
    exp = Experiment()
    exp.reproducibility()
    print(f'\nSeed used: {exp.seed}')

    # Get the subsets with full path.
    data_dirs = list_subdirs_fullpath(datasets_dir)
    print('\nInput directories (full paths):')
    for i, dirs in enumerate(data_dirs):
        print(f'[{i}] --> {dirs}')

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

        # Loading the datasets into a dict.
        datasets = {x: torchvision.datasets.ImageFolder(
            os.path.join(data_dir, x),
            transform=torchvision.transforms.ToTensor()
        ) for x in splits}

        # Creating the dataloaders into a dict.
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
            mean, std = get_mean_std_dataloader(dataloaders[x])
            print(mean)
            print(std)

            # Write to file.
            f.write(f'{x}\n')
            f.write(f'{mean}\n')
            f.write(f'{std}\n')

        # Close file and print an empty line.
        f.close()
        print()
    
    return 0


# Call the main function to execute the program.
if __name__ == '__main__':
    sys.exit(main())
