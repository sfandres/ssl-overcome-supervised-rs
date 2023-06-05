#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script launches the finetuning of the random, imagenet, and self-supervised learning models.

Usage: This module does not have arguments.

Author: Andres J. Sanchez-Fernandez
Email: sfandres@unex.es
Date: 2023-04-12
"""


import sys
import os


def main():
    """"Main function."""

    model_names = ['Random', 'Imagenet', 'SimSiam', 'SimCLR', 'SimCLRv2', 'BarlowTwins', 'MoCov2']
    tasks = ['multiclass', 'multilabel']
    dataset_name = 'Sentinel2AndaluciaLULC'
    dataset_level = 'Level_N2'
    dataset_train_pcs = [.01, .025, .05, .075, .1, .25, .5, .75, 1.]
    epochs = 100
    batch_size = 64

    # for s in range(0, times):
    # print (f'Experiments: {times}')

    for dataset_train_pc in dataset_train_pcs:
        print('\n----------------------------------------------------------')
        print(dataset_train_pc)
        print('----------------------------------------------------------')
        for task in tasks:
            print(f'\n------------> {task}')
            for model in model_names:
                print(f'\n------------> {model}\n')
                os.system(
                    'python3 03_2-PyTorch-Backbone_classifier.py '
                    f'{model} '
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
    