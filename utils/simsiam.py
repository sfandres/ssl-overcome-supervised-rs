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
from lightly.utils.debug import std_of_l2_normalized
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

    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the forward pass and returns loss.

        Args:
            x0 (torch.Tensor): Batch of input images.
            x1 (torch.Tensor): Batch of input images.

        Returns:
            torch.Tensor: Loss computed after the forward pass in the projection head.
        """

        # Output projections of both transformed batches.
        # Get representations.
        f0 = self.backbone(x0).flatten(start_dim=1)
        f1 = self.backbone(x1).flatten(start_dim=1)

        # Get projections.
        z0 = self.projection_head(f0)
        z1 = self.projection_head(f1)

        # Get predictions.
        p0 = self.prediction_head(z0)
        p1 = self.prediction_head(z1)

        # Stop gradient for projections.
        z0 = z0.detach()
        z1 = z1.detach()

        # Negative Cosine Similarity.
        loss = .5 * (self.criterion(z0, p1) + self.criterion(z1, p0))

        # Used to check if the representations are collapsing later.
        # We only take one for simplicity.
        self.embedding = p0.detach()

        return loss

    def check_collapse(
        self,
    ) -> float:
        """
        Checks the collapse of the model (non-contrastive SSL).

        https://docs.lightly.ai/self-supervised-learning/getting_started/advanced.html

        Representation collapse can happen during unstable training and results in the model's
        output (prediction head) predicting the same, or very similar, representations for all images.
        This is of course disastrous for model training as we want to the representations to be
        as different as possible between images!
        
        A value close to 0 indicates that the representations have collapsed. A value close to
        1/sqrt(dimensions), where dimensions are the number of representation dimensions, indicates
        that the representations are stable. Below we show model training outputs from a run where
        the representations collapse and one where they don't collapse.

        Args:
            None.
        
        Returns:
            float: the level of collapse.
        """

        return std_of_l2_normalized(self.embedding)
