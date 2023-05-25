#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module provides the MoCov2 class.

https://docs.lightly.ai/self-supervised-learning/examples/moco.html

Author: Andres J. Sanchez-Fernandez
Email: sfandres@unex.es
Date: 2023-05-25
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
        x_q: torch.Tensor,
        x_k: torch.Tensor,
        momentum_val: float = 0.999
    ) -> torch.Tensor:
        """
        Computes the forward pass and returns loss.

        Args:
            x0 (torch.Tensor): Batch of input images.
            x1 (torch.Tensor): Batch of input images.
            momentum_val (float, optional): Momentum value according to the current epoch.

        Returns:
            torch.Tensor: Loss computed after the forward pass in the projection head.
        """

        # Check if momentum is present in kwargs.
        # if 'momentum_val' in kwargs:
        #     momentum_val = kwargs['momentum_val']
        # else:
        #     momentum_val = None

        # Update momentum.
        update_momentum(self.backbone,
                        self.backbone_momentum,
                        m=momentum_val)

        update_momentum(self.projection_head,
                        self.projection_head_momentum,
                        m=momentum_val)

        # Query.
        # Feature extraction using the ResNet backbone.
        q = self.backbone(x_q).flatten(start_dim=1)
        # Feature embeddings.
        z_q = self.projection_head(q)

        # Key.
        # Feature extraction using the ResNet backbone.
        k = self.backbone_momentum(x_k).flatten(start_dim=1)
        # Feature embeddings.
        z_k = self.projection_head_momentum(k).detach()

        # Contrastive Cross Entropy Loss.
        loss = self.criterion(z_q, z_k)

        return loss
