#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script provides an example for timing CPU-GPU transfers.

Usage:

Author: Andres J. Sanchez-Fernandez
Email: sfandres@unex.es
Date: 2023-04-02
"""


import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from torchvision import transforms
import time
import numpy as np


# Set the random seed for PyTorch.
torch.manual_seed(0)

# Set the random seed for NumPy.
np.random.seed(0)

# Define transform for data preprocessing.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR10 dataset.
dataset = CIFAR10(root='./input/datasets/', train=True, download=True, transform=transform)

# Split dataset into train and validation sets.
train_dataset, val_dataset = random_split(dataset, [40000, 10000])

# Create dataloaders for train and validation sets.
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=1)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=1)

# Define a model.
model = torch.nn.Sequential(
    torch.nn.Conv2d(3, 6, kernel_size=5),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2),
    torch.nn.Conv2d(6, 16, kernel_size=5),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2),
    torch.nn.Flatten(),
    torch.nn.Linear(16 * 5 * 5, 120),
    torch.nn.ReLU(),
    torch.nn.Linear(120, 84),
    torch.nn.ReLU(),
    torch.nn.Linear(84, 10)
)

# Move model to GPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Time the CPU-GPU transfers of the train dataloader.
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

transfer_times = []
start_event.record()
for i, (images, labels) in enumerate(train_dataloader):
    images = images.to(device)
    labels = labels.to(device)
    end_event.record()
    torch.cuda.synchronize()  # synchronize CPU and GPU execution.
    transfer_time = start_event.elapsed_time(end_event) / 1000.0
    transfer_times.append(transfer_time)
    start_event.record()

# Compute average and show results.
avg_transfer_time = sum(transfer_times) / len(transfer_times)
print(f"Average time for transferring one batch of data from CPU to GPU: {avg_transfer_time:.5f} seconds")