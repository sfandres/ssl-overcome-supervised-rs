#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module provides the base class for all models.

Author: Andres J. Sanchez-Fernandez
Email: sfandres@unex.es
Date: 2023-04-04
"""


import torch
import torch.nn as nn
from datetime import datetime
import os


class BaseModel(nn.Module):
    """
    Base class for all methods that inherits from nn.Module.
    """
    def __init__(self):
        """Constructor of the class."""
        super().__init__()

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): input tensor to the model.

        Returns:
            torch.Tensor: output tensor of the model.
        """
        pass

    def training_step(self, batch, batch_idx):
        """
        Defines the training step for the model.

        Args:
            batch (torch.Tensor): input batch of data for training.
            batch_idx (int): the index of the current batch.

        Returns:
            torch.Tensor: the loss value for the current batch.
        """
        pass

    def save(
        self,
        backbone_name: str = None,
        epoch: int = None,
        train_loss: float = None,
        dataset_ratio: str = None,
        balanced_dataset: bool = None,
        path: str = None,
        collapse_level: float = 0.
    ) -> None:
        """
        Saves the model to a file with a custom name.

        Args:
            backbone_name (str): the name of the backbone model.
            epoch (int): the epoch number for which the model is being saved.
            train_loss (float): the loss value for the model at the given epoch.
            path (str): the path where the checkpoint will be saved.
            dataset_ratio (str): the current ratio of the dataset.
            balanced_dataset (bool): whether the dataset is balanced.
            path (str): the path where the model is saved.
            collapse_level (optional, float): collapse level for non-contrastive SSL models.
        """

        # Save parameters.
        self.backbone_name = backbone_name
        self.epoch = epoch
        self.train_loss = train_loss
        self.dataset_ratio = dataset_ratio
        self.balanced_dataset = balanced_dataset
        self.collapse_level = collapse_level
        self.time = datetime.now()

        # Save the weights.
        torch.save(self.backbone.state_dict(),
                   os.path.join(path, self.__str__()))

    def __str__(self) -> str:
        """Overwriting the string representation of the class."""

        # Get the class name and filename.
        class_name = self.__class__.__name__

        # Filename with stats.
        filename = f'{class_name}' \
                   f'-{self.backbone_name}' \
                   f'-epoch={self.epoch:03}' \
                   f'-train_loss={self.train_loss:.3f}' \
                   f'-ratio={self.dataset_ratio}' \
                   f'-balanced={self.balanced_dataset}' \
                   f'-coll={self.collapse_level:.3f}(0)' \
                   f'-time={self.time:%Y_%m_%d_%H_%M_%S}'

        return filename