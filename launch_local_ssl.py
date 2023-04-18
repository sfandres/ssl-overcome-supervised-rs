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
    backbone = 'resnet18'
    dataset_name = 'Sentinel2GlobalLULC_SSL'
    dataset_ratio = '(0.020,0.0196,0.9604)'
    epochs = 25
    batch_size = 64

    # for s in range(0, times):
    # print (f'Experiments: {times}')

    for model in model_names:
        print('\n----------------------------------------------------------')
        print(model)
        print('----------------------------------------------------------')
        os.system(
            'python3 03_1-PyTorch-Sentinel-2_SSL_pretraining.py '
            f'{model} '
            f'--backbone={backbone} '
            f'--dataset_name={dataset_name} '
            f'--dataset_ratio={dataset_ratio} '
            f'--epochs={epochs} '
            f'--batch_size={batch_size} '
            f'--cluster '
            f'--ray_tune'
        )

    return 0


# Call the main function to execute the program.
if __name__ == '__main__':
    sys.exit(main())
    