#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module provides is used to check the versions of Python and PyTorch being used.

Usage: This module does not have arguments.

Author: Andres J. Sanchez-Fernandez
Email: sfandres@unex.es
Date: 2023-04-10
"""


# Import the PyTorch library.
import torch


# Print the version of Python being used.
print('\n------------------------------------------------------------------------------------')
print(f"{'Python3 version:'.ljust(33)}"
      f"{__import__('sys').version}")

# Print the version of PyTorch being used.
print(f"{'PyTorch version:'.ljust(33)}"
      f"{torch.__version__}")

# Print the version of CUDA being used by PyTorch.
print(f"{'Cuda version used by PyTorch:'.ljust(33)}"
      f"{torch.version.cuda}")

# Check if a GPU is available for use with PyTorch.
print(f"{'torch.cuda.is_available():'.ljust(33)}"
      f"{torch.cuda.is_available()}")

# Count the number of available GPUs for use with PyTorch.
print(f"{'torch.cuda.device_count():'.ljust(33)}"
      f"{torch.cuda.device_count()}")

# Get the index of the current device being used by PyTorch.
idx = torch.cuda.current_device()
print(f"{'torch.cuda.current_device():'.ljust(33)}"
      f"{idx}")

# Get the device object for a specified GPU.
string1 = f'torch.cuda.device({idx}):'
print(f"{string1.ljust(33)}"
      f"{torch.cuda.device(idx)}")

# Get the name of the specified GPU.
string2 = f'torch.cuda.get_device_name({idx}):'
print(f"{string2.ljust(33)}"
      f"{torch.cuda.get_device_name(idx)}")
print('------------------------------------------------------------------------------------\n')
