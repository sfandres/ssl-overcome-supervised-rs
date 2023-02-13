"""Useful functions used to manage the datasets.

Usage:
    -

Author:
    A.J. Sanchez-Fernandez - 13/02/2023
"""


import os
import ast


def list_subdirs_fullpath(input_path):
    """
    Takes the "input_path" and creates a list of
    the subdirectories with the root path included.

    Args:
        input_path: the target directory.

    Returns:
        A sorted list of the subdirectories with the full path. 
    """

    return sorted([os.path.join(input_path, x)
                   for x in os.listdir(input_path)])


def load_mean_std_values(input_path):
    """
    Takes the "input_path" and loads the mean
    and std values from the txt file.

    Args:
        input_path: path to the directory where the dataset is saved.

    Returns:
        mean: dictionary holding the mean of the samples
            per dimension and split.
        std: dictionary holding the standard deviation of
            the samples per dimension and split.
    """

    # Initialization.
    splits = ['train', 'val', 'test']
    filename = 'dataset_mean_std.txt'
    mean, std = {}, {}

    # Create path to the txt file.
    filepath = os.path.join(input_path, filename)

    # Read the values by transforming from str to list
    # and reading only the target characters.
    with open(filepath) as f:
        lines = f.readlines()
        mean['train'] = ast.literal_eval(lines[1][7:-2])
        std['train'] = ast.literal_eval(lines[2][7:-2])
        mean['val'] = ast.literal_eval(lines[4][7:-2])
        std['val'] = ast.literal_eval(lines[5][7:-2])
        mean['test'] = ast.literal_eval(lines[7][7:-2])
        std['test'] = ast.literal_eval(lines[8][7:-2])

    return mean, std


def load_dataset_based_on_ratio(input_path, name, ratio):
    """
    Takes the "input_path", as well as the target "name"
    and "ratio" of the dataset and returns the full path,
    mean and std values.

    Args:
        input_path: directory where the datasets are stored.
        name: name of the target dataset (directory's name as well).
        ratio: ratio of the target dataset according to train, val, and test.

    Returns:
        split_path: full path to the target split.
        mean: dictionary holding the mean of the samples.
        std: dictionary holding the standard deviation of the samples.
    """

    # Get target dataset directory.
    dataset_path = os.path.join(input_path, name)
    print(f'Path to dataset folder: {dataset_path}')

    # Get the splits in the dataset directory.
    splits_path = list_subdirs_fullpath(dataset_path)

    # Iterate over the splits.
    for split_path in splits_path:
        
        # Get the current ratio.
        split_ratio = split_path[
            split_path.index("("):split_path.index(")")+1
        ]

        # Compare both ratios.
        if split_ratio == ratio:
            found = True
            break

    # Handling errors: finding the path to the target split.
    try:
        print(f'Dataset found based on the given ratio: {found}')
        print(f'\nPath to split folder: {split_path}')
    except:
        raise Exception('Dataset split not found based on the given ratio. '
                        'Please check the ratio.')

    # Handling errors: loading mean and std from file.
    try:
        mean, std = load_mean_std_values(split_path)
        print(f'Mean loaded from .txt: {mean}')
        print(f'Std loaded from .txt: {std}')
    except:
        raise Exception('Error loading the mean and std files for the target dataset. '
                        'Please check the files.')

    return split_path, mean, std
