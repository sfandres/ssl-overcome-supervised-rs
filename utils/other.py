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


def build_paths(cwd: str, script_name: str) -> dict:
    """
    Buils all the necessary paths into a dictionary.

    Args:
        cwd (str): path to the current working directory.
        script_name (str): name of the running script.

    Returns:
        paths (dict): all the generated paths in a dictionary.
    """

    paths = {}

    try:
        # Create main paths.
        input_path = os.path.join(cwd, 'input')
        output_path = os.path.join(
            cwd,
            os.path.join('output', script_name)
        )

        # Create directories if they don't exist.
        os.makedirs(input_path, exist_ok=True)
        os.makedirs(output_path, exist_ok=True)

        # INPUT.
        # Second level paths within input.
        datasets_path = os.path.join(input_path, 'datasets')
        best_configs_path = os.path.join(input_path, 'best_configs')
    
        # Create directories if they don't exist.
        os.makedirs(datasets_path, exist_ok=True)
        os.makedirs(best_configs_path, exist_ok=True)

        # OUTPUT.
        # Second level paths within output.
        csv_results_path = os.path.join(output_path, 'csv_results')
        images_path = os.path.join(output_path, 'images')
        checkpoints_path = os.path.join(output_path, 'ckpts')
        ray_tune_path = os.path.join(output_path, 'ray_tune')
        runs_path = os.path.join(output_path, 'runs')
        snapshots_path = os.path.join(output_path, 'snapshots')

        # Create directories if they don't exist.
        os.makedirs(csv_results_path, exist_ok=True)
        os.makedirs(images_path, exist_ok=True)
        os.makedirs(checkpoints_path, exist_ok=True)
        os.makedirs(ray_tune_path, exist_ok=True)
        os.makedirs(runs_path, exist_ok=True)
        os.makedirs(snapshots_path, exist_ok=True)

        # CHECKPOINTS.
        # Third level paths within checkpoints.

        # Create dictionary.
        paths = {
            'input': input_path,
            'best_configs': best_configs_path,
            'datasets': datasets_path,
            'output': output_path,
            'csv_results': csv_results_path,
            'images': images_path,
            'checkpoints': checkpoints_path,
            'ray_tune': ray_tune_path,
            'runs': runs_path,
            'snapshots': snapshots_path
        }

    except OSError as e:
        print(f"Error occurred while creating directories: {e}")

    return paths
