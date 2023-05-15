#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module provides functions for seeding.

The module contains two functions: 'set_seed' to seed the
main modules and 'seed_worker' to seed the PyTorch workers.

Author: Andres J. Sanchez-Fernandez
Email: sfandres@unex.es
Date: 2023-03-22
"""


import os
import numpy as np
import random
import torch


def set_seed(seed: int) -> torch._C.Generator:
    """
    Seeds the numpy, random, and torch modules.

    Args:
        seed (int): target seed for each RNG.

    Returns:
        PyTorch generator for DataLoaders.
    """

    # Seed torch and numpy.
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Seed the RNG for all devices (both CPU and CUDA).
    torch.manual_seed(seed)

    # Enable CUDNN deterministic mode.
    torch.backends.cudnn.benchmark = False

    # PyTorch to use deterministic algorithms.
    torch.backends.cudnn.deterministic = True

    # Issues a warning if it is not met.
    torch.use_deterministic_algorithms(True)

    # Set PyTorch dataloader generator seed.
    g = torch.Generator()
    g.manual_seed(seed)

    # Enable deterministic behavior using external GPU.
    # %env CUBLAS_WORKSPACE_CONFIG=:4096:8
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

    return g


def seed_worker(worker_id: int):
    """
    Seeds the workers of the PyTorch DataLoaders.

    Args:
        worker_id (int): id of the current worker.
    """

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
