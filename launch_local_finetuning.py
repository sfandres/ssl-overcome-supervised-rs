#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script launches the training of the self-supervised learning models.

Usage: This module does not have arguments.

Author: Andres J. Sanchez-Fernandez
Email: sfandres@unex.es
Date: 2023-04-11
"""


import sys
import os


def main():
    """"Main function."""

    model_names = ['SimSiam', 'SimCLR', 'SimCLRv2', 'BarlowTwins']

    tasks = ['multiclass', 'multilabel']
    modes = ['scratch', 'imagenet', 'ssl']
    dataset_name = 'Sentinel2AndaluciaLULC'
    dataset_level = 'Level_N2'
    dataset_train_pcs = [.025, .05, .075, .1, .25, .5, .75, 1.]
    epochs = 4
    batch_size = 64

    # for s in range(0, times):
    # print (f'Experiments: {times}')

    for dataset_train_pc in dataset_train_pcs:
        print('\n----------------------------------------------------------')
        print(dataset_train_pc)
        print('----------------------------------------------------------')
        for task in tasks:
            print(f'\n------------> {task}')
            for mode in modes:
                print(f'\n------------> {mode}\n')
                os.system(
                    'python3 03_2-PyTorch-Backbone_classifier.py '
                    f'{mode} '
                    f'{task} '
                    f'--dataset_name={dataset_name} '
                    f'--dataset_level={dataset_level} '
                    f'--dataset_train_pc={dataset_train_pc} '
                    f'--epochs={epochs} '
                    f'--batch_size={batch_size} '
                    f'--cluster'
                )

    return 0


# Call the main function to execute the program.
if __name__ == '__main__':
    sys.exit(main())
    