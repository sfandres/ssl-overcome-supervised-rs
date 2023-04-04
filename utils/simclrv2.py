#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module provides the SimCLRv2 class.

https://paperswithcode.com/method/simclrv2

Author: Andres J. Sanchez-Fernandez
Email: sfandres@unex.es
Date: 2023-04-04
"""


import torch
import torch.nn as nn
from lightly.loss import NTXentLoss
from .base_model import BaseModel


class SimCLRv2(BaseModel):
    """
    SimCLRv2 self-supervised learning model.

    Attributes:
        backbone (nn.Module): Backbone model.
        projection_head (nn.Sequential): Projection head network that generates feature embeddings.
        criterion (NTXentLoss): Contrastive Cross Entropy Loss.

    Methods:
        forward(x): Computes the forward pass of the SimCLR model.
        training_step(batch): Computes the loss of the current batch (tuple of tensors).
    """

    def __init__(
        self,
        backbone: nn.Sequential,
        input_dim: int = 512,
        hidden_dim: int = 512,
        output_dim: int = 512
    ):
        """
        Initializes a new SimCLR model.

        Args:
            backbone (nn.Sequential): Backbone model.
            input_dim (int): Number of input features to the fc layer.
            hidden_dim (int): Dimension of the hidden layers.
            output_dim (int): Dimensionality of the feature embeddings
            produced by the projection head network.
        """

        # Call the BaseModel superclass constructor.
        super().__init__()

        # Include the backbone.
        self.backbone = backbone

        # Projector network to generate feature embeddings.
        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

        # Loss criterion (memory bank = 1 for MoCo).
        self.criterion = NTXentLoss(temperature=0.5, memory_bank_size=0)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the forward pass.

        Args:
            x (torch.Tensor): Batch of input images.

        Returns:
            torch.Tensor: Tensor of feature embeddings produced by the projection head network.
        """

        # Feature extraction using the ResNet backbone.
        z = self.backbone(x).flatten(start_dim=1)

        # Feature embeddings.
        h = self.projection_head(z)

        return h

    def training_step(
        self,
        two_batches: tuple[torch.Tensor, torch.Tensor],
    ) -> float:
        """
        Performs a single training step on a batch of transformed images.

        Args:
            two_batches (tuple): Tuple of two batches of transformed images,
            where each batch is a tensor of size (batch_size, C, H, W).

        Returns:
            float: The loss value for the current batch.
        """

        # Two batches of transformed images.
        x0, x1 = two_batches

        # Output projections of both transformed batches.
        z0 = self.forward(x0)
        z1 = self.forward(x1)

        # Contrastive Cross Entropy Loss.
        loss = self.criterion(z0, z1)

        return loss
