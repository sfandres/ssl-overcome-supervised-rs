#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module provides the MoCov2 class.

Author: Andres J. Sanchez-Fernandez
Email: sfandres@unex.es
Date: 2023-04-10
"""


import torch
import torch.nn as nn
import copy
from lightly.loss import NTXentLoss
from lightly.models.modules import MoCoProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from .base_model import BaseModel


class MoCov2(BaseModel):
    """
    MoCov2 self-supervised learning model.

    Attributes:
        backbone (nn.Module): Backbone model.
        backbone_momentum (nn.Module): Backbone model.
        projection_head (nn.Sequential): Projection head network that generates feature embeddings.
        projection_head_momentum (nn.Sequential): Projection head network that generates feature embeddings.
        criterion (NTXentLoss): Contrastive Cross Entropy Loss with bank memory enabled.

    Methods:
        forward(x): Computes the forward pass of the MoCov2 model.
        forward_momentum(x): Computes the forward pass of the MoCov2 model.
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
        self.projection_head = MoCoProjectionHead(input_dim,
                                                  hidden_dim,
                                                  output_dim)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        # Loss criterion (memory bank > 0 for MoCo).
        self.criterion = NTXentLoss(temperature=0.5, memory_bank_size=4096)

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
        query = self.backbone(x).flatten(start_dim=1)

        # Feature embeddings.
        query = self.projection_head(query)

        return query

    def forward_momentum(
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
        key = self.backbone_momentum(x).flatten(start_dim=1)

        # Feature embeddings.
        key = self.projection_head_momentum(query)

        return key
    
    
    def training_step(
        self,
        two_batches: tuple[torch.Tensor, torch.Tensor],
        momentum_val: float = None
    ) -> float:
        """
        Performs a single training step on a batch of transformed images.

        Args:
            two_batches (tuple): Tuple of two batches of transformed images,
            where each batch is a tensor of size (batch_size, C, H, W).
            momentum_val (float): Momentum value according to the current epoch.

        Returns:
            float: The loss value for the current batch.
        """

        update_momentum(self.backbone,
                        self.backbone_momentum,
                        m=momentum_val)

        update_momentum(self.projection_head,
                        self.projection_head_momentum,
                        m=momentum_val)

        # Two batches of transformed images.
        x_query, x_key = two_batches

        # Output projections of both transformed batches.
        query = self.forward(x_query)
        key = self.forward_momentum(x_key)

        # Contrastive Cross Entropy Loss.
        loss = self.criterion(query, key)

        return loss
