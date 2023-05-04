#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module provides the SimCLR(v2) class.

https://paperswithcode.com/method/simclr
https://github.com/lightly-ai/lightly/blob/master/lightly/models/modules/heads.py

Author: Andres J. Sanchez-Fernandez
Email: sfandres@unex.es
Date: 2023-05-04
"""


import torch
import torch.nn as nn
from lightly.models.modules.heads import SimCLRProjectionHead
from lightly.loss import NTXentLoss
from .base_model import BaseModel


class SimCLR(BaseModel):
    """
    SimCLR(v2) self-supervised learning model.

    Attributes:
        backbone (nn.Module): Backbone model.
        projection_head (nn.Sequential): Projection head network that generates feature embeddings.
        criterion (NTXentLoss): Contrastive Cross Entropy Loss.

    Methods:
        forward(x): Computes the forward pass of the SimCLR(v2) model.
        training_step(batch): Computes the loss of the current batch (tuple of tensors).
    """

    def __init__(
        self,
        backbone: nn.Sequential,
        input_dim: int = 512,
        hidden_dim: int = 512,
        output_dim: int = 512,
        num_layers: int = 2,
        memory_bank_size: int = 0
    ):
        """
        Initializes a new SimCLR(v2) model.

        Args:
            backbone (nn.Sequential): Backbone model.
            input_dim (int): Number of input features to the fc layer.
            hidden_dim (int): Dimension of the hidden layers.
            output_dim (int): Dimensionality of the feature embeddings
            produced by the projection head network.
            num_layers (int): Number of hidden layers (2 for v1, 3+ for v2).
            memory_bank_size (int): Number of negative samples to store in the memory bank.
            Use 0 for SimCLRv1 and 65536 for v2. Memory bank > 0 for MoCo, we typically use numbers like 4096 or 65536.
        """

        # Call the BaseModel superclass constructor.
        super().__init__()

        # Blackbone model.
        self.backbone = backbone

        # Projection head.
        # num_layers: Number of hidden layers (2 for v1, 3+ for v2).
        # batch_norm: Whether or not to use batch norms.
        self.projection_head = SimCLRProjectionHead(input_dim=input_dim,
                                                    hidden_dim=hidden_dim,
                                                    output_dim=output_dim,
                                                    num_layers=num_layers,
                                                    batch_norm=True)

        # Loss criterion.
        self.criterion = NTXentLoss(temperature=0.5, memory_bank_size=memory_bank_size)

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
        **kwargs
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
