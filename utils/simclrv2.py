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
        backbone: str = 'resnet18',
        weights = None,
        feature_dim: int = 128
    ):
        """
        Initializes a new SimCLR model.

        Args:
            backbone (nn.Module): A ResNet backbone network.
            weights (ResNet18/50_Weights): The pretrained weights of the network.
            feature_dim (int): The dimensionality of the feature embeddings
            produced by the projection head network.
        """

        # Call the nn.Module superclass constructor.
        super(SimCLRv2, self).__init__()

        # Loss criterion (memory bank = 1 for MoCo).
        self.criterion = NTXentLoss(temperature=0.5, memory_bank_size=0)

        # Choose the backbone architecture based on the argument provided.
        if backbone == 'resnet18':
            self.backbone = resnet.resnet18(weights=weights)
            num_ftrs = self.backbone.fc.in_features
        elif backbone == 'resnet50':
            self.backbone = resnet.resnet50(weights=weights)
            num_ftrs = self.backbone.fc.in_features
        else:
            raise ValueError("Invalid backbone architecture")

        # Replace the last fully connected layer with an identity function.
        self.backbone.fc = nn.Identity()

        # Projector network to generate feature embeddings.
        hidden_dim = num_ftrs
        self.projection_head = nn.Sequential(
            nn.Linear(num_ftrs, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim)
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
        z = self.backbone(x)

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
        epoch: int,
        loss: float,
        path: str
    ) -> None:
        """
        Saves the SimCLR model to a file with a custom name.

        Args:
            epoch (int): The epoch number for which the model is being saved.
            loss (float): The loss value for the model at the given epoch.
            path (str): The path where the checkpoint will be saved.
        """

        # Get the class name and filename.
        class_name = self.__class__.__name__
        filename = f'{class_name}_epoch={epoch:03}_loss={loss:.4f}'

        # Save the weights.
        torch.save(self.state_dict(),
                   os.path.join(path, filename))
