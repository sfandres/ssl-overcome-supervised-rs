#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module provides the SimCLR v2 class.

https://paperswithcode.com/method/simclrv2

Author: Andres J. Sanchez-Fernandez
Email: sfandres@unex.es
Date: 2023-03-28
"""


import torch
import torch.nn as nn
import torchvision.models.resnet as resnet
from lightly.loss import NTXentLoss
import os
from datetime import datetime


class SimCLRv2(nn.Module):
    """
    SimCLR model with a ResNet backbone and a projection head.

    Attributes:
        criterion (NTXentLoss): Contrastive Cross Entropy Loss.
        backbone (nn.Module): A ResNet backbone network.
        projection_head (nn.Sequential): A projection head network that generates feature embeddings.

    Methods:
        forward(x): Computes the forward pass of the SimCLR model.
        training_step(batch): Computes the loss of the current batch (tuple of tensors).
        save(loss, epoch, path): Saves the SimCLR model to a file with a custom name.
    """

    def __init__(
        self,
        backbone: nn.Sequential,
        input_dim: int = 512,
        hidden_dim: int = 512,
        output_dim: int = 128
    ):
        """
        Initializes a new SimCLR model.

        Args:
            backbone (nn.Sequential): A ResNet backbone network.
            input_dim (int): Number of input features to the fc layer.
            feature_dim (int): The dimensionality of the feature embeddings
            produced by the projection head network.
        """

        # Call the nn.Module superclass constructor.
        super(SimCLRv2, self).__init__()

        # Loss criterion (memory bank = 1 for MoCo).
        self.criterion = NTXentLoss(temperature=0.5, memory_bank_size=0)

        # Include the backbone.
        self.backbone = backbone

        # Projector network to generate feature embeddings.
        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the forward pass of the SimCLR model.

        Args:
            x (torch.Tensor): A batch of input images.

        Returns:
            torch.Tensor: A tensor of feature embeddings produced by the projection head network.
        """

        # Feature extraction using the ResNet backbone.
        z = self.backbone(x).flatten(start_dim=1)

        # Feature embeddings.
        h = self.projection_head(z)

        return h

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> float:
        """
        Performs a single training step on a batch of transformed images.

        Args:
            batch (tuple): A tuple of two batches of transformed images, where each batch is a tensor of size (batch_size, C, H, W).

        Returns:
            Float: The loss value for the current batch.
        """

        # Two batches of transformed images.
        x0, x1 = batch

        # Output projections of both transformed batches.
        z0 = self.forward(x0)
        z1 = self.forward(x1)

        # Contrastive Cross Entropy Loss.
        loss = self.criterion(z0, z1)

        return loss

    def save(
        self,
        backbone_name: str = None,
        epoch: int = None,
        train_loss: float = None,
        dataset_ratio: str = None,
        balanced_dataset: bool = None,
        path: str = None
    ) -> None:
        """
        Saves the SimCLR model to a file with a custom name.

        Args:
            epoch (int): The epoch number for which the model is being saved.
            loss (float): The loss value for the model at the given epoch.
            path (str): The path where the checkpoint will be saved.
        """

        # Save parameters.
        self.backbone_name = backbone_name
        self.epoch = epoch
        self.train_loss = train_loss
        self.dataset_ratio = dataset_ratio
        self.balanced_dataset = balanced_dataset
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
                   f'-time={self.time:%Y_%m_%d_%H_%M_%S}'

        return filename