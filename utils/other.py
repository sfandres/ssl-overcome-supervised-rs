#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module provides functions with different objectives.

The module contains two functions: 'is_notebook' to check whether
the Python environment runs inside a notebook and 'build_paths' to
create all the paths necessary for the script to run into a dict.

Author: Andres J. Sanchez-Fernandez
Email: sfandres@unex.es
Date: 2023-03-22
"""


import os


def is_notebook() -> bool:
    """
    Checks whether the current Python environment is running inside a
    Jupyter Notebook or a Python script.

    Returns:
        True if the environment is a Jupyter Notebook, False otherwise.
    """

    try:
        shell = get_ipython().__class__.__name__

        # Jupyter notebook or qtconsole.
        if shell == 'ZMQInteractiveShell':
            return True

        # Terminal running IPython.
        elif shell == 'TerminalInteractiveShell':
            return False

        # Other type (?).
        else:
            return False

    # Probably standard Python interpreter.
    except NameError:
        return False


def build_paths(cwd: str, model_name: str) -> dict:
    """
    Buils all the necessary paths into a dictionary.

    Args:
        cwd (str): path to the current working directory.
        model_name (str): name of the target SSL model.

    Returns:
        All the generated paths in a dictionary.
    """

    # Create main paths.
    input_path = os.path.join(cwd, 'input')
    output_path = os.path.join(cwd, 'output')

    # Create directories if they don't exist.
    if not os.path.exists(input_path):
        os.makedirs(input_path)
        print(f'Dir created: {input_path}')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f'Dir created: {output_path}')

    # Second level paths within input.
    datasets_path = os.path.join(input_path, 'datasets')

    # Create directories if they don't exist.
    if not os.path.exists(datasets_path):
        os.makedirs(datasets_path)
        print(f'Dir created: {datasets_path}')

    # Second level paths within output.
    checkpoints_path = os.path.join(output_path, 'checkpoints')
    images_path = os.path.join(output_path, 'images')
    runs_path = os.path.join(output_path, 'runs')

    # Create directories if they don't exist.
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)
        print(f'Dir created: {checkpoints_path}')
    if not os.path.exists(images_path):
        os.makedirs(images_path)
        print(f'Dir created: {images_path}')
    if not os.path.exists(runs_path):
        os.makedirs(runs_path)
        print(f'Dir created: {runs_path}')

    # Third level paths within checkpoints.
    checkpoints_logs_path = os.path.join(checkpoints_path, '0_history_logs')
    if not os.path.exists(checkpoints_logs_path):
        os.makedirs(checkpoints_logs_path)
        print(f'Dir created: {checkpoints_logs_path}')

    checkpoints_model_path = os.path.join(checkpoints_path, model_name)
    if not os.path.exists(checkpoints_model_path):
        os.makedirs(checkpoints_model_path)
        print(f'Dir created: {checkpoints_model_path}')

    # Third level paths within images.
    images_logs_path = os.path.join(images_path, '0_history_logs')
    if not os.path.exists(images_logs_path):
        os.makedirs(images_logs_path)
        print(f'Dir created: {images_logs_path}')

    images_model_path = os.path.join(images_path, model_name)
    if not os.path.exists(images_model_path):
        os.makedirs(images_model_path)
        print(f'Dir created: {images_model_path}')

    # Create dictionary.
    paths_dict = {
        'input_path': input_path,
        'datasets_path': datasets_path,
        'output_path': output_path,
        'runs_path': runs_path,
        'checkpoints_model_path': checkpoints_model_path,
        'images_model_path': images_model_path        
    }
            
    return paths_dict
