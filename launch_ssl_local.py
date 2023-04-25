#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script launches the training of the self-supervised learning models.

Usage: This module does not have arguments.

Author: Andres J. Sanchez-Fernandez
Email: sfandres@unex.es
Date: 2023-04-18
"""


import sys
import os


def main():
    """"Main function."""

    model_names = ['SimSiam', 'SimCLR', 'SimCLRv2', 'BarlowTwins', 'MoCov2']
    backbone_name = ['resnet18', 'resnet50']
    dataset_name = 'Sentinel2GlobalLULC_SSL'
    dataset_ratio = '\(0.900,0.0250,0.0750\)'
    epochs = 500
    batch_size = 32
    ini_weights = 'random'

    ray_tune = ['gridsearch', 'loguniform']
    num_samples_trials = 10
    tune_options = f'--reduced_dataset --ray_tune={ray_tune[0]} --num_samples_trials={num_samples_trials}'
    resume_training = '--resume_training'

    more_options = tune_options  ## '' / tune_options / resume_training

    # for s in range(0, times):
    # print (f'Experiments: {times}')

    for model in model_names:
        print('\n----------------------------------------------------------')
        print(model)
        print('----------------------------------------------------------')
        os.system(
            'python3 03_1-PyTorch-Sentinel-2_SSL_pretraining.py '
            f'{model} '
            f'--backbone_name={backbone_name[1]} '
            f'--dataset_name={dataset_name} '
            f'--dataset_ratio={dataset_ratio} '
            f'--epochs={epochs} '
            f'--batch_size={batch_size} '
            f'--ini_weights={ini_weights} '
            f'--cluster '
            f'{more_options}'
        )

    return 0


# Call the main function to execute the program.
if __name__ == '__main__':
    sys.exit(main())

