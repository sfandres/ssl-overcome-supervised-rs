#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script launches the training of the self-supervised learning models.

Usage: This module does not have arguments.

Author: Andres J. Sanchez-Fernandez
Email: sfandres@unex.es
Date: 2023-04-26
"""


import sys
import os
import argparse


def main():
    """"Main function."""

    # Enable arguments.
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--training", help="Runs normal training.", action="store_true")
    parser.add_argument("-r", "--resume-training", help="Resumes the training from a previous saved checkpoint.", action="store_true")
    parser.add_argument("-g", "--gridsearch", help="Enables Ray Tune with tune.gridsearch to tune all the hyperparamenters.", action="store_true")
    parser.add_argument("-l", "--loguniform", help="Enables Ray Tune with tune.loguniform to tune the learning rate.", action="store_true")

    args = parser.parse_args()

    # Define settings for the experiments.
    model_names = ['SimSiam', 'SimCLR', 'SimCLRv2', 'BarlowTwins', 'MoCov2']
    backbone_name = 'resnet18'  # 'resnet50'
    dataset_name = 'Sentinel2GlobalLULC_SSL'
    dataset_ratio = '\(0.900,0.0250,0.0750\)'
    epochs = 500
    if backbone_name == 'resnet50':
        batch_size = 16
    else:
        batch_size = 64
    ini_weights = 'random'

    # Catch the arguments.
    if args.training:
        print("You chose normal training")
        exp_options = f"--epochs={epochs}"
    elif args.resume_training:
        print("You chose resume training")
        exp_options = f"--epochs={epochs} --resume_training"
    elif args.gridsearch:
        print("You chose tune.gridsearch")
        epochs = 10
        exp_options = f"--epochs={epochs} --reduced_dataset --ray_tune=gridsearch --num_samples_trials=1"
    elif args.loguniform:
        print("You chose tune.loguniform")
        epochs = 10
        exp_options = f"--epochs={epochs} --reduced_dataset --ray_tune=loguniform --num_samples_trials=10"
    else:
        print("Invalid option. Use -h or --help to display available options.")
        exit(1)

    # Show the chosen options.
    print(f"Specific options of the current experiment: {backbone_name} {exp_options}")

    # for s in range(0, times):
    # print (f'Experiments: {times}')

    for model in model_names:
        print('\n----------------------------------------------------------')
        print(model)
        print('----------------------------------------------------------')
        os.system(
            'python3 03_1-PyTorch-Sentinel-2_SSL_pretraining.py '
            f'{model} '
            f'--backbone_name={backbone_name} '
            f'--dataset_name={dataset_name} '
            f'--dataset_ratio={dataset_ratio} '
            f'--batch_size={batch_size} '
            f'--ini_weights={ini_weights} '
            f'--cluster '
            f'{exp_options}'
        )

    return 0


# Call the main function to execute the program.
if __name__ == '__main__':
    sys.exit(main())

