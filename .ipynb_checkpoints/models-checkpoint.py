import torch
from lightly.models.modules.heads import SimSiamPredictionHead,SimSiamProjectionHead
from lightly.models.modules.heads import SimCLRProjectionHead
from lightly.loss import NTXentLoss,NegativeCosineSimilarity
from datetime import datetime


class SimSiam(torch.nn.Module):
    """
    SimSiam self-supervised learning model.

    Attributes:
        backbone: Architecture of the CNN model.
        num_ftrs: Dimension of the embeddings.
        proj_hidden_dim: Dimension of the output of the prediction and projection heads.
        pred_hidden_dim: Dimension of the prediction head.
        out_dim: Dimension of the output of the prediction and projection heads.
    """

    def __init__(self, backbone, num_ftrs, proj_hidden_dim,
                 pred_hidden_dim, out_dim):
        """Constructor of the class."""

        # Inheritance.
        super().__init__()

        # Blackbone model.
        self.backbone = backbone

        # Projection head (lightly).
        self.projection_head = SimSiamProjectionHead(
            num_ftrs, proj_hidden_dim, out_dim
        )

        # Prediction head (lightly).
        self.prediction_head = SimSiamPredictionHead(
            out_dim, pred_hidden_dim, out_dim
        )

        # Loss criterion.
        self.criterion = NegativeCosineSimilarity()

    def forward(self, x):
        """How your model runs from input to output."""

        # Get representations.
        f = self.backbone(x).flatten(start_dim=1)

        # Get projections.
        z = self.projection_head(f)

        # Get predictions.
        p = self.prediction_head(z)

        # Stop gradient.
        z = z.detach()

        return z, p

    def checkpoint_filename(self, epoch, train_loss, val_loss,
                            avg_rep_collapse):
        """Creates the name of the checkpoint file."""

        # Filename with stats.
        filename = f'simsiam_bb_resnet18' \
                   f'-epoch={epoch:03}' \
                   f'-train_loss={train_loss:.4f}' \
                   f'-val_loss={val_loss:.4f}' \
                   f'-coll={avg_rep_collapse:.4f}(0)' \
                   f'-time={datetime.now():%Y_%m_%d_%H_%M_%S}'

        return filename


class SimCLRModel(torch.nn.Module):
    """
    SimCLR self-supervised learning model.

    Attributes:
        backbone: Architecture of the CNN model.
        hidden_dim: Dimension of the output of the projection head.
    """

    def __init__(self, backbone, hidden_dim):
        """Constructor of the class."""

        # Inheritance.
        super().__init__()

        # Blackbone model.
        self.backbone = backbone

        # Projection head.
        self.projection_head = SimCLRProjectionHead(
            hidden_dim, hidden_dim, 128
        )

        # Loss criterion.
        self.criterion = NTXentLoss()

    def forward(self, x):
        """How your model runs from input to output."""

        # Get representations.
        h = self.backbone(x).flatten(start_dim=1)

        # Get projections.
        z = self.projection_head(h)

        return z

    def checkpoint_filename(self, epoch, train_loss, val_loss,
                             handle_imb_classes, ratio):
        """Creates the name of the checkpoint file."""

        # Filename with stats.
        filename = f'simclr_bb_resnet18' \
                   f'-epoch={epoch:03}' \
                   f'-train_loss={train_loss:.4f}' \
                   f'-val_loss={val_loss:.4f}' \
                   f'-balanced={handle_imb_classes}' \
                   f'-ratio={ratio}' \
                   f'-time={datetime.now():%Y_%m_%d_%H_%M_%S}'

        return filename