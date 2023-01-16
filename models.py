import torch
from lightly.models.modules.heads import SimSiamPredictionHead, SimSiamProjectionHead
from lightly.loss import NegativeCosineSimilarity

from lightly.models.modules.heads import SimCLRProjectionHead
from lightly.loss import NTXentLoss

from lightly.models.modules import BarlowTwinsProjectionHead
from lightly.loss import BarlowTwinsLoss

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
        
        # Setup (check collapse).
        self.avg_loss = 0.
        self.avg_output_std = 0.

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
    
    def check_collapse(self, p0, loss):
        """Check the collapse of the model (non-contrastive SSL)."""

        # Calculate the per-dimension standard deviation of the outputs.
        # We can use this later to check whether the embeddings are collapsing.
        output = p0.detach()
        output = torch.nn.functional.normalize(output, dim=1)

        output_std = torch.std(output, 0)
        output_std = output_std.mean()

        # Use moving averages to track the loss and standard deviation.
        w = 0.9
        self.avg_loss = w * self.avg_loss + (1 - w) * loss.item()
        self.avg_output_std = w * self.avg_output_std + (1 - w) * output_std.item()

    def evaluation(self, dataloader_val):
        """Evaluation process, returns the loss."""

        pass

    def save(self, epoch, train_loss, handle_imb_classes, ratio, output_dir_model, collapse_level=None):
        """Saving the model."""
        
        # Save parameters.
        self.epoch = epoch
        self.train_loss = train_loss
        self.handle_imb_classes = handle_imb_classes
        self.ratio = ratio
        self.collapse_level = collapse_level
        self.time = datetime.now()

        torch.save(self.backbone.state_dict(),
                   output_dir_model + 'simsiam/' + self.__str__())

    def __str__(self):
        """Overwriting the string representation of the class."""

        # Filename with stats.
        filename = f'simsiam_bb_resnet18' \
                   f'-epoch={self.epoch:03}' \
                   f'-train_loss={self.train_loss:.4f}' \
                   f'-coll={self.collapse_level:.4f}(0)' \
                   f'-balanced={self.handle_imb_classes}' \
                   f'-ratio={self.ratio}' \
                   f'-time={self.time:%Y_%m_%d_%H_%M_%S}'

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

        # Get representations and projections.
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)

        return z

    def training_step(self, x0, x1):
        """Calculate loss."""

        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)

        return loss

    def evaluation(self, dataloader_val):
        """Evaluation process, returns the loss."""

        pass

    def save(self, epoch, train_loss, handle_imb_classes, ratio, output_dir_model, collapse_level=None):
        """Saving the model."""

        # Save parameters.
        self.epoch = epoch
        self.train_loss = train_loss
        self.handle_imb_classes = handle_imb_classes
        self.ratio = ratio
        self.time = datetime.now()

        # Save weights and biases.
        torch.save(self.backbone.state_dict(),
                   output_dir_model + 'simclr/' + self.__str__())

    def __str__(self):
        """Overwriting the string representation of the class."""

        # Filename with stats (no avg_rep_collapse).
        filename = f'simclr_bb_resnet18' \
                   f'-epoch={self.epoch:03}' \
                   f'-train_loss={self.train_loss:.4f}' \
                   f'-balanced={self.handle_imb_classes}' \
                   f'-ratio={self.ratio}' \
                   f'-time={self.time:%Y_%m_%d_%H_%M_%S}'

        return filename


class BarlowTwins(torch.nn.Module):
    """
    BarlowTwins self-supervised learning model.

    Attributes:
        backbone: Architecture of the CNN model.
    """

    def __init__(self, backbone):
        """Constructor of the class."""

        # Inheritance.
        super().__init__()

        # Blackbone model.
        self.backbone = backbone

        # Projection head.
        self.projection_head = BarlowTwinsProjectionHead(512, 2048, 2048)

        # Loss criterion.
        self.criterion = BarlowTwinsLoss()

    def forward(self, x):
        """How your model runs from input to output."""

        # Get representations and projections.
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)

        return z

    def training_step(self, x0, x1):
        """Calculate loss."""

        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)

        return loss

    def evaluation(self, dataloader_val):
        """Evaluation process, returns the loss."""

        pass

    def save(self, epoch, train_loss, handle_imb_classes, ratio, output_dir_model, collapse_level=None):
        """Saving the model."""

        # Save parameters.
        self.epoch = epoch
        self.train_loss = train_loss
        self.handle_imb_classes = handle_imb_classes
        self.ratio = ratio
        self.time = datetime.now()

        # Save weights and biases.
        torch.save(self.backbone.state_dict(),
                   output_dir_model + 'barlowtwins/' + self.__str__())

    def __str__(self):
        """Overwriting the string representation of the class."""

        # Filename with stats (no avg_rep_collapse).
        filename = f'barlowtwins_bb_resnet18' \
                   f'-epoch={self.epoch:03}' \
                   f'-train_loss={self.train_loss:.4f}' \
                   f'-balanced={self.handle_imb_classes}' \
                   f'-ratio={self.ratio}' \
                   f'-time={self.time:%Y_%m_%d_%H_%M_%S}'

        return filename
