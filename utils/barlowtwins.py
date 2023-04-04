#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module provides the BarlowTwins class.

https://github.com/facebookresearch/barlowtwins/blob/8e8d284ca0bc02f88b92328e53f9b901e86b4a3c/main.py#L187

Author: Andres J. Sanchez-Fernandez
Email: sfandres@unex.es
Date: 2023-04-04
"""


import torch
import torch.nn as nn
from lightly.models.modules import BarlowTwinsProjectionHead
from lightly.loss import BarlowTwinsLoss
from .base_model import BaseModel


class BarlowTwins(BaseModel):
    """
    BarlowTwins self-supervised learning model.

    Attributes:
        backbone (nn.Module): Backbone model.
        projection_head (nn.Sequential): Projection head network that generates feature embeddings.
        criterion (BarlowTwinsLoss): Loss function.

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
        Initializes a new BarlowTwins model.

        Args:
            backbone (nn.Sequential): Backbone model.
            input_dim (int): Number of input features to the fc layer.
            hidden_dim (int): Dimension of the hidden layers.
            output_dim (int): Dimensionality of the feature embeddings
            produced by the projection head network.
        """

        # Call the BaseModel superclass constructor.
        super().__init__()

        # Blackbone model.
        self.backbone = backbone

        # Projection head.
        self.projection_head = BarlowTwinsProjectionHead(input_dim=input_dim,
                                                         hidden_dim=hidden_dim,
                                                         output_dim=output_dim)

        # Loss criterion.
        self.criterion = BarlowTwinsLoss(lambda_param=0.005)

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
        x = self.backbone(x).flatten(start_dim=1)

        # Feature embeddings.
        z = self.projection_head(x)

        return z

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

        # Contrastive BarlowTwinsLoss.
        loss = self.criterion(z0, z1)

        return loss
