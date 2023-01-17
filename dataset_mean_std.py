"""
Script to write to a txt the mean and std of each 
dataset so that the data can be normalized afterwards.
"""

import os
import torch
import torchvision
import utils


# Reproducibility.
exp = utils.Experiment()
exp.reproducibility()
print(f'\nSeed used: {exp.seed}')

# List of trained models.
datasets_dir = 'datasets/'

# Get the subsets with full path.
data_dirs = utils.listdir_fullpath(datasets_dir)

# Leave out unwanted subsets (0_Raw).
data_dirs = data_dirs[1:]
for dirs in data_dirs:
    print(dirs)

# Initialization.
splits = ['train', 'val', 'test']
filename = 'dataset_mean_std.txt'

# Loop over the datasets (except raw and clothing).
for data_dir in data_dirs:

    # Create path to the txt file.
    filepath = os.path.join(data_dir, filename)

    # Removing the old txt file if exists.
    if os.path.exists(filepath):
        os.remove(filepath)
        print(f'\n{data_dir}: Existing txt file found and removed.')
        print('New empty txt file created.')
    else:
        print(f'\n{data_dir}: New empty txt file created.')

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
        batch_size=128,
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

    # Close file and print a space.
    f.close()
    print('')