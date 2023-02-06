import torch
from lightly.models.modules.heads import SimSiamPredictionHead, SimSiamProjectionHead
from lightly.loss import NegativeCosineSimilarity

from lightly.models.modules.heads import SimCLRProjectionHead
from lightly.loss import NTXentLoss

from lightly.models.modules import BarlowTwinsProjectionHead
from lightly.loss import BarlowTwinsLoss

from datetime import datetime
import os

from flash.core.optimizers import LARS as FLASH_LARS


class SimSiam(torch.nn.Module):
    """
    SimSiam self-supervised learning model.

    Attributes:
        backbone: Architecture of the CNN model.
        num_ftrs: Dimension of the embeddings.
        proj_hidden_dim: Dimension of the output of the prediction and projection heads.
        pred_hidden_dim: Dimension of the prediction head.
        out_dim: Dimension of the output of the prediction and projection heads.
        
    From the original SimSiam paper:
        Optimizer. We use SGD for pre-training. Our method
        does not require a large-batch optimizer such as LARS
        [36] (unlike [8, 15, 7]). We use a learning rate of
        lr×BatchSize/256 (linear scaling [14]), with a base lr =
        0.05. The learning rate has a cosine decay schedule
        [26, 8]. The weight decay is 0.0001 and the SGD momentum is 0.9.
        The batch size is 512 by default, [...]
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

    def set_up_optimizer_and_scheduler(self, epochs, batch_size=512,
                                       lr=0.05, momentum=0.9, weight_decay=1e-4):
        """Set up the optimizer and scheduler."""

        # Infer learning rate.
        self.init_lr = lr * batch_size / 256

        # Optimizer.
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            self.init_lr,
            momentum=momentum,
            weight_decay=weight_decay
        )

        # Scheduler.
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
            verbose=True
        )

    def save(self, epoch, train_loss, val_loss, handle_imb_classes, ratio, output_dir_model, collapse_level=None):
        """Saving the model."""
        
        # Save parameters.
        self.epoch = epoch
        self.train_loss = train_loss
        self.val_loss = val_loss
        self.handle_imb_classes = handle_imb_classes
        self.ratio = ratio
        self.collapse_level = collapse_level
        self.time = datetime.now()

        torch.save(self.backbone.state_dict(),
                   os.path.join(output_dir_model, self.__str__()))

    def __str__(self):
        """Overwriting the string representation of the class."""

        # Filename with stats.
        filename = f'simsiam' \
                   f'-ratio={self.ratio}' \
                   f'-val_loss={self.val_loss:.3f}' \
                   f'-coll={self.collapse_level:.3f}(0)' \
                   f'-train_loss={self.train_loss:.3f}' \
                   f'-epoch={self.epoch:03}' \
                   f'-balanced={self.handle_imb_classes}' \
                   f'-bb=resnet18' \
                   f'-time={self.time:%Y_%m_%d_%H_%M_%S}'

        return filename



class SimCLRModel(torch.nn.Module):
    """
    SimCLR self-supervised learning model.

    Attributes:
        backbone: Architecture of the CNN model.
        hidden_dim: Dimension of the output of the projection head.

    From the original SimCLR paper:
        Default setting. Unless otherwise specified, for data augmentation
        we use random crop and resize (with random flip), color distortions,
        and Gaussian blur (for details, see Appendix A). We use ResNet-50 as
        the base encoder network, and a 2-layer MLP projection head to project
        the representation to a 128-dimensional latent space. As the loss, we
        use NT-Xent, optimized using LARS with learning rate of 4.8
        (= 0.3 × BatchSize/256) and weight decay of 10−6. We train at batch size
        4096 for 100 epochs.3 Furthermore, we use linear warmup for the first
        10 epochs, and decay the learning rate with the cosine decay schedule
        without restarts (Loshchilov & Hutter, 2016).
        3 Although max performance is not reached in 100 epochs, reasonable
        results are achieved, allowing fair and efficient ablations [...]
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

    def set_up_optimizer_and_scheduler(self, epochs, batch_size=512,
                                       lr=0.3, weight_decay=1e-6):
        """Set up the optimizer and scheduler."""

        # Infer learning rate.
        self.init_lr = float(lr * batch_size / 256)

        # Optimizer.
        self.optimizer = FLASH_LARS(self.parameters(),
                                    lr=self.init_lr)

        # Linear warmup for the first 10 epochs.
        self.warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda epoch: min(1, epoch / 10),
            verbose=True
        )

        # Cosine decay starting from the 11th epoch.
        self.main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
            verbose=True
        )

    def training_step(self, x0, x1):
        """Calculate loss."""

        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)

        return loss

    def evaluation(self, dataloader_val):
        """Evaluation process, returns the loss."""

        pass

    def save(self, epoch, train_loss, val_loss, handle_imb_classes, ratio, output_dir_model, collapse_level=None):
        """Saving the model."""

        # Save parameters.
        self.epoch = epoch
        self.train_loss = train_loss
        self.val_loss = val_loss
        self.handle_imb_classes = handle_imb_classes
        self.ratio = ratio
        self.time = datetime.now()

        # Save weights and biases.
        torch.save(self.backbone.state_dict(),
                   os.path.join(output_dir_model, self.__str__()))

    def __str__(self):
        """Overwriting the string representation of the class."""
    
        # Filename with stats (no avg_rep_collapse).
        filename = f'simclr' \
                   f'-ratio={self.ratio}' \
                   f'-val_loss={self.val_loss:.3f}' \
                   f'-train_loss={self.train_loss:.3f}' \
                   f'-epoch={self.epoch:03}' \
                   f'-balanced={self.handle_imb_classes}' \
                   f'-bb=resnet18' \
                   f'-time={self.time:%Y_%m_%d_%H_%M_%S}'

        return filename


class BarlowTwins(torch.nn.Module):
    """
    BarlowTwins self-supervised learning model.

    Attributes:
        backbone: Architecture of the CNN model.
            
    From the original BarlowTwins paper:
        We use the LARS optimizer (You et al., 2017) and train for 1000 epochs with
        a batch size of 2048. We however emphasize that our model works well with
        batches as small as 256 (see Ablations). We use a learning rate of 0.2 for
        the weights and 0.0048 for the biases and batch normalization parameters. We
        multiply the learning rate by the batch size and divide it by 256. We use a
        learning rate warm-up period of 10 epochs, after which we reduce the learning
        rate by a factor of 1000 using a cosine decay schedule (Loshchilov & Hutter,
        2016). We ran a search for the trade-off parameter λ of the loss function and
        found the best results for λ = 5 · 10−3. We use a weight decay parameter of
        1.5 · 10−6. The biases and batch normalization parameters are excluded from
        LARS adaptation and weight decay. [...]
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
        self.criterion = BarlowTwinsLoss(lambda_param=0.005)

    def forward(self, x):
        """How your model runs from input to output."""

        # Get representations and projections.
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)

        return z
    
    def configure_optimizer(self, batch_size, weight_decay=1.5e-6):
        """Configures the optimizer."""

        # Creates two groups of params.
        param_weights = []
        param_biases = []
        for param in self.parameters():
            if param.ndim == 1:
                param_biases.append(param)
            else:
                param_weights.append(param)
        # parameters = [{'params': param_weights}, {'params': param_biases}]
        self.parameters = [{'params': param_weights, 'lr': 0.2}, {'params': param_biases, 'lr': 0.0048}]

        # Infer learning rate.
        base_lr = float(batch_size / 256)
        print(f'Base lr: {base_lr}')

        # Optimizer.
        self.optimizer = LARS(self.parameters,
                              lr=base_lr,
                              weight_decay=weight_decay,
                              weight_decay_filter=True,
                              lars_adaptation_filter=True)

    def configure_scheduler(self, epochs, batch_size=512,
                            lr=0.2, weight_decay=1e-6):
        """Set up the optimizer and scheduler."""

        # Linear warmup for the first 10 epochs.
        self.warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda epoch: min(1, epoch / 10),
            verbose=True
        )

        # Cosine decay starting from the 11th epoch.
        self.main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
            verbose=True
        )

    def training_step(self, x0, x1):
        """Calculate loss."""

        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)

        return loss

    def evaluation(self, dataloader_val):
        """Evaluation process, returns the loss."""

        pass

    def save(self, epoch, train_loss, val_loss, handle_imb_classes, ratio, output_dir_model, collapse_level=None):
        """Saving the model."""

        # Save parameters.
        self.epoch = epoch
        self.train_loss = train_loss
        self.val_loss = val_loss
        self.handle_imb_classes = handle_imb_classes
        self.ratio = ratio
        self.time = datetime.now()

        # Save weights and biases.
        torch.save(self.backbone.state_dict(),
                   os.path.join(output_dir_model, self.__str__()))

    def __str__(self):
        """Overwriting the string representation of the class."""

        # Filename with stats (no avg_rep_collapse).
        filename = f'barlowtwins' \
                   f'-ratio={self.ratio}' \
                   f'-val_loss={self.val_loss:.3f}' \
                   f'-train_loss={self.train_loss:.3f}' \
                   f'-epoch={self.epoch:03}' \
                   f'-balanced={self.handle_imb_classes}' \
                   f'-bb=resnet18' \
                   f'-time={self.time:%Y_%m_%d_%H_%M_%S}'

        return filename


class LARS(torch.optim.Optimizer):
    """
    LARS (Layer-wise Adaptive Rate Scaling) is an optimization algorithm designed
    for large-batch training published by You, Gitman, and Ginsburg, which calculates
    the local learning rate per layer at each optimization step.

    From Barlow Twins: Self-Supervised Learning via Redundancy Reduction.
    https://github.com/facebookresearch/barlowtwins -> main.py
    """
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=False, lars_adaptation_filter=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])
