#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module provides the SimSiam class.

https://github.com/facebookresearch/simsiam/blob/main/simsiam/builder.py

Author: Andres J. Sanchez-Fernandez
Email: sfandres@unex.es
Date: 2023-04-04
"""


import torch
import torch.nn as nn
from lightly.models.modules.heads import SimSiamPredictionHead, SimSiamProjectionHead
from lightly.loss import NegativeCosineSimilarity
from .base_model import BaseModel


class SimSiam(BaseModel):
    """
    SimSiam self-supervised learning model.

    Attributes:
        backbone (torch.nn.Module): Backbone model.
        projection_head (SimSiamProjectionHead): Projection head of the model.
        prediction_head (SimSiamPredictionHead): Prediction head of the model (bottleneck architecture).
        criterion (NegativeCosineSimilarity): Loss criterion used for training.
        p0 (torch.Tensor): Output projection of the first transformed batch of images x0 (check collapse).
        avg_loss (float): Average loss during training.
        avg_output_std (float): Average output standard deviation during training.

    Methods:
        forward(x): Computes the forward pass of the SimCLR model.
        training_step(batch): Computes the loss of the current batch (tuple of tensors).
        check_collapse(p0, loss): Checks the collapse of the model (non-contrastive SSL).
    """
    def __init__(
        self,
        backbone: nn.Sequential,
        input_dim: int = 512,
        proj_hidden_dim: int = 512,
        pred_hidden_dim: int = 128,
        output_dim: int = 512
    ):
        """
        Initializes a new SimSiam model.

        Args:
            backbone (nn.Sequential): Backbone model.
            input_dim (int): Number of input features to the fc layer.
            proj_hidden_dim (int): Dimension of the hidden layers of the projection head.
            pred_hidden_dim (int): Dimension of the hidden layers of the prediction head.
            output_dim (int): Dimensionality of the feature embeddings.
        """

        # Inheritance.
        super().__init__()

        # Blackbone model.
        self.backbone = backbone

        # Projection head (lightly).
        self.projection_head = SimSiamProjectionHead(input_dim=input_dim,
                                                     hidden_dim=proj_hidden_dim,
                                                     output_dim=output_dim)

        # Prediction head (lightly).
        self.prediction_head = SimSiamPredictionHead(input_dim=output_dim,
                                                     hidden_dim=pred_hidden_dim,
                                                     output_dim=output_dim)

        # Loss criterion.
        self.criterion = NegativeCosineSimilarity()
        
        # Setup (check collapse).
        # self.avg_loss = 0.
        self.avg_output_std = 0.

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

        # Get representations.
        f = self.backbone(x).flatten(start_dim=1)

        # Get projections.
        z = self.projection_head(f)

        # Get predictions.
        p = self.prediction_head(z)

        # Stop gradient.
        z = z.detach()

        return z, p

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
        z0, p0 = self.forward(x0)
        z1, p1 = self.forward(x1)

        # Negative Cosine Similarity.
        loss = .5 * (self.criterion(z0, p1) + self.criterion(z1, p0))

        # For checking collapsing later.
        self.p0 = p0

        return loss
    
    def check_collapse(
        self,
        loss
    ) -> None:
        """
        Checks the collapse of the model (non-contrastive SSL).

        Calculates the per-dimension standard deviation of the model's output (prediction head),
        and uses moving averages to track the loss and standard deviation. This method can be
        used to detect whether the embeddings produced by the model are collapsing.

        Args:
            loss (torch.Tensor): The loss incurred by the model.

        Returns:
            None.
        """

        # Calculate the per-dimension standard deviation of the outputs.
        # We can use this later to check whether the embeddings are collapsing.
        output = self.p0.detach()
        output = torch.nn.functional.normalize(output, dim=1)

        output_std = torch.std(output, 0)
        output_std = output_std.mean()

        # Use moving averages to track the loss and standard deviation.
        w = 0.9
        # self.avg_loss = w * self.avg_loss + (1 - w) * loss.item()
        self.avg_output_std = w * self.avg_output_std + (1 - w) * output_std.item()
