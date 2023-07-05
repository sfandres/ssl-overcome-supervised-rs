import torch
import os
import time
import csv
import numpy as np

from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from ray import tune
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    f1_score
)

NUM_DECIMALS = 3


# ===================================
# ACCURACY METRICS
# ===================================

def accuracy(
    model: torch.nn.Module,
    dataloader: DataLoader,
    task_name: str,
    device: int
) -> dict:
    """Calculates the accuracy of a model on a given dataset.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        dataloader (DataLoader): The DataLoader object that provides the dataset.
        task_name (str): The name of the downstream task being evaluated ('multiclass' or 'multilabel').
        device (int): The device to use for computations.

    Returns:
        dict: A dictionary containing the accuracy of the model on the dataset.

    Example:
        model = MyModel()
        data_loader = DataLoader(dataset)
        accuracy_dict = accuracy(model, data_loader, "multiclass", device=0)
    """

    # Initialize the probabilities, predictions and labels lists.
    y_prob = []
    y_pred = []
    y_true = []

    # Calculate probabilities and predictions.
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:

            inputs = inputs.to(device)                              # Inputs and labels from the dataset.
            labels = labels.to(device)

            outputs = model(inputs)                                 # Forward pass.

            if task_name == 'multiclass':
                probs = torch.softmax(outputs, dim=1)               # Convert logits to probabilities using a softmax function.
                preds = torch.argmax(probs, dim=1)                  # Take the argmax of the probabilities to obtain the predicted class labels.

            elif task_name == 'multilabel':
                probs = torch.sigmoid(outputs)                      # Convert logits to probabilities using a sigmoid function.
                preds_sum = probs.sum(dim=1, keepdim=True)          # Scale predicted abundances to sum to 1 across all the classes.
                preds = probs / preds_sum

            y_prob.append(probs)                                    # Append true and predicted labels to the lists.
            y_pred.append(preds)
            y_true.append(labels)

    # Accuracy metrics.
    if task_name == 'multiclass':                                   # Downstream task --> multiclass.

        y_prob_cpu = torch.cat(y_prob).to('cpu')                    # Concatenate the lists into tensors.
        y_pred_cpu = torch.cat(y_pred).to('cpu')
        y_true_cpu = torch.cat(y_true).to('cpu')

        top1_accuracy = torch.sum(torch.eq(y_pred_cpu, y_true_cpu)).item() / len(y_true_cpu)
        top5_accuracy = torch.sum(torch.topk(y_prob_cpu, k=5, dim=1)[1] == y_true_cpu.view(-1, 1)).item() / len(y_true_cpu)

        f1_micro = f1_score(y_true_cpu, y_pred_cpu, average='micro')
        f1_macro = f1_score(y_true_cpu, y_pred_cpu, average='macro')
        f1_weighted = f1_score(y_true_cpu, y_pred_cpu, average='weighted')
        f1_per_class = f1_score(y_true_cpu, y_pred_cpu, average=None)

        acc_dict = {
            'top1': top1_accuracy,
            'top5': top5_accuracy,
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'f1_per_class': list(np.round(f1_per_class, NUM_DECIMALS))
        }

    elif task_name == 'multilabel':                                 # Downstream task --> multilabel.

        y_prob_cpu = torch.cat(y_prob).to('cpu')                    # Concatenate the lists into tensors.
        y_pred_cpu = torch.cat(y_pred).to('cpu')
        y_true_cpu = torch.cat(y_true).to('cpu')

        rmse = mean_squared_error(y_true_cpu, y_pred_cpu, squared=False)
        mae = mean_absolute_error(y_true_cpu, y_pred_cpu)

        rmse_per_class = mean_squared_error(y_true_cpu, y_pred_cpu, multioutput='raw_values', squared=False)
        mae_per_class = mean_absolute_error(y_true_cpu, y_pred_cpu, multioutput='raw_values')

        acc_dict = {
            'rmse': rmse,
            'mae': mae,
            'rmse_per_class': list(np.round(rmse_per_class, NUM_DECIMALS)),
            'mae_per_class': list(np.round(mae_per_class, NUM_DECIMALS))
        }

    return acc_dict


# ===================================
# CUSTOM TRAINER  (tune.Trainable)
# ===================================

class Trainer():
    """
    Trainer class for training and evaluation.

    Attributes:
        model (torch.nn.Module): The model to be trained.
        dataloader (DataLoader): Data loader object for training and validation data.
        loss_fn (torch.nn.modules.loss): Loss function for training.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        save_every (int): Frequency of saving snapshots during training.
        snapshot_path (str): Path to save the snapshots.
        csv_path (str): Path to save the CSV file for tracking metrics.
        distributed (bool, optional): Flag indicating distributed training. Defaults to False.
        lightly_train (bool, optional): Flag indicating if lightly supervised training is enabled. Defaults to False.
        ray_tune (bool, optional): Flag indicating if the Ray Tune tool is enabled. Defaults to False.
        ignore_ckpts (bool, optional): Flag indicating whether to ignore checkpoints. Defaults to False.
        
        local_rank (int): The local rank of the current process retrieved from the environment variables (torchrun).
        global_rank (int): The global rank of the current process retrieved from the environment variables (torchrun).
        epochs_run (int): The number of epochs run so far. Initialized as 0.
        batch_size (int): The batch size of the training dataset.

    Private methods:
        _load_snapshot(): Load a snapshot from the provided path.
        _save_snapshot(epoch: int): Save a snapshot of the model and optimizer state.
        _run_evaluation(): Run evaluation on the validation dataset and return the validation loss.
        _run_batch(source: torch.Tensor, targets: torch.Tensor): Run a single batch during training and compute the loss.
        _run_epoch(epoch: int): Run a single epoch of training and compute the training and validation losses.
        _save_to_csv(data: list): Save data to a CSV file.
        _adjust_optimizer_for_ft(config: dict): Change from LP to FT (transfer learning to fine-tuning).
        _initial_optimizer_setup(config: dict): Adjust the optimizer according to the provided hyperparameters (Ray Tune).

    Public methods:
        train(config: dict = None): Main training loop.
    """

    # ===================================================
    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        loss_fn: torch.nn.modules.loss,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        snapshot_path: str,
        csv_path: str,
        distributed: bool = False,
        lightly_train: bool = False,
        ray_tune: bool = False,
        ignore_ckpts: bool = False
    ) -> None:
        """ Initialize Trainer object with the provided parameters. """

        super().__init__()                                                          # Assign instance attributes.
        self.model = model
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.save_every = save_every
        self.snapshot_path = snapshot_path
        self.csv_path = csv_path
        self.distributed = distributed
        self.lightly_train = lightly_train
        self.ray_tune = ray_tune
        self.ignore_ckpts = ignore_ckpts

        self.local_rank = int(os.environ["LOCAL_RANK"])                             # Retrieve environment variables.
        self.global_rank = int(os.environ["RANK"])

        self.model = model.to(self.local_rank)                                      # Move the model to the local rank device.

        self.batch_size = len(next(iter(self.dataloader['train']))[0])              # Retrieve the batch size and initialize current epoch.
        self.epochs_run = 0

        if os.path.exists(snapshot_path) and not ignore_ckpts and not ray_tune:
            self._load_snapshot()                                                   # Load snapshot if it exists and not other flags.

        if distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank])              # Distributed training with DDP.


    # ===================================================
    def _load_snapshot(
        self
    ) -> None:
        """ Load a snapshot from the provided path. """

        print('\nLoading snapshot...')
        loc = f'cuda:{self.local_rank}'                         # Specify the device location for loading the snapshot.
        snapshot = torch.load(self.snapshot_path,               # Load the snapshot from the specified path.
                              map_location=loc)
        self.model.load_state_dict(snapshot['MODEL_STATE'])     # Load the model's state dictionary from the snapshot.
        self.epochs_run = snapshot['EPOCHS_RUN']                # Set the number of epochs run from the snapshot.
        self.optimizer.load_state_dict(snapshot['OPTIMIZER'])   # Load the optimizer's state dictionary from the snapshot.
        print(f"Resuming training from snapshot at Epoch {self.epochs_run} <-- {self.snapshot_path.rsplit('/', 1)[-1]}")


    # ===================================================
    def _save_snapshot(
        self,
        epoch: int
    ) -> None:
        """
        Save a snapshot of the model and optimizer state.

        Args:
            epoch (int): Current epoch number.
        """

        print('Saving snapshot...')
        if self.distributed:
            model_state = self.model.module.state_dict()    # Get the state dictionary of the model (distributed training).
        else:
            model_state = self.model.state_dict()           # Get the state dictionary of the model (non-distributed training).
        snapshot = {
            'MODEL_STATE': model_state,                     # Save the model state dictionary.
            'EPOCHS_RUN': epoch + 1,                        # Save the number of epochs run (+1 for resuming at the same point).
            'OPTIMIZER': self.optimizer.state_dict()        # Save the optimizer state dictionary.
        }
        torch.save(snapshot, self.snapshot_path)            # Save the snapshot to the specified path.
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path.rsplit('/', 1)[-1]}")


    # ===================================================
    def _run_evaluation(
        self
    ) -> torch.Tensor:
        """
        Run evaluation on the validation dataset.

        Returns:
            torch.Tensor: The validation loss.
        """

        running_val_loss = 0.                                           # Initialize the running loss.
        self.model.eval()                                               # Set the model to evaluation mode (e.g., Dropout and BatchNorm layers).

        with torch.no_grad():                                           # Disable gradient computation (lighter computation).
            for source, targets in self.dataloader['val']:
                source = source.to(self.local_rank)                     # Move source and targets to the specified device.
                targets = targets.to(self.local_rank)
                output = self.model(source)                             # Compute model output.
                loss = self.loss_fn(output, targets)                    # Compute loss.
                running_val_loss += loss.detach() * self.batch_size     # Update the running loss (no gradient required).

        epoch_val_loss = running_val_loss / len(self.dataloader['val'].sampler)     # Compute the average validation loss per sample.

        return epoch_val_loss


    # ===================================================
    def _run_batch(
        self,
        source: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Run a single batch during training and compute the loss.

        Args:
            source (torch.Tensor): Input data for the batch.
            targets (torch.Tensor): Target data for the batch.

        Returns:
            torch.Tensor: The computed loss.
        """

        source = source.to(self.local_rank)         # Move the source and targets tensors to the device.
        targets = targets.to(self.local_rank)

        self.optimizer.zero_grad()                  # Clear optimizer gradients.
        output = self.model(source)                 # Compute model output.
        loss = self.loss_fn(output, targets)        # Compute loss.
        loss.backward()                             # Compute gradients (backpropagation).
        self.optimizer.step()                       # Update model parameters.

        return loss.detach()                        # Detach loss from computation graph (no gradient required).


    # ===================================================
    def _run_epoch(
        self,
        epoch: int
    ) -> tuple:
        """
        Run a single epoch of training and compute the training and validation losses.

        Args:
            epoch (int): Current epoch number.

        Returns:
            tuple: A tuple containing the training loss and validation loss.
        """

        running_train_loss = 0.                                         # Track the running training loss.
        self.model.train()                                              # Set the model to training mode.

        if self.distributed:
            self.dataloader['train'].sampler.set_epoch(epoch)           # Set the epoch for distributed training (if enabled).

        t0 = time.time()                                                # Record the starting time of the epoch.

        if not self.lightly_train:                                      # Check if lightly supervised training is disabled.
            for source, targets in self.dataloader['train']:            # Iterate over the training data batches.
                loss = self._run_batch(source, targets)                 # Run a single training batch and compute the loss.
                running_train_loss += loss * self.batch_size            # Accumulate the training loss.
        else:
            for (source, _), targets, _ in self.dataloader['train']:    # Iterate over the lightly supervised training data batches.
                loss = self._run_batch(source, targets)
                running_train_loss += loss * self.batch_size

        epoch_train_loss = (running_train_loss /
                            len(self.dataloader['train'].sampler))      # Compute the average training loss for the epoch.

        epoch_val_loss = self._run_evaluation()                         # Run the evaluation for the epoch.

        print(f"[GPU{self.global_rank}] | [Epoch: {epoch}] | Train loss: {epoch_train_loss:.4f} | "
              f"Steps: {len(self.dataloader['train'])} | Val loss: {epoch_val_loss:.4f} | "
              f"Batch size: {self.batch_size} | lr: {self.optimizer.param_groups[0]['lr']} | "
              f"Duration: {(time.time()-t0):.2f}s")

        return round(float(epoch_train_loss), NUM_DECIMALS), round(float(epoch_val_loss), NUM_DECIMALS)


    # ===================================================
    def _save_to_csv(
        self,
        data: list
    ) -> None:
        """
        Saves the given data to a CSV file.

        Args:
            data (list): Data to be saved in the CSV file.
        """

        with open(self.csv_path, 'a', newline='') as file:      # Open the CSV file in append mode.
            csv_writer = csv.writer(file)                       # Create a CSV writer object.
            csv_writer.writerow(data)                           # Write the data as a row in the CSV file.


    # ===================================================
    def _adjust_optimizer_for_ft(
        self,
        config: dict
    ) -> None:
        """
        Changes to FT, which means a new optimizer configuration and unfrozen weights (when LP+FT).

        Args:
            config (dict): Provided configuration for the experiment.
        """

        print('\nUnfreezing the weights and updating the optimizer configuration (LP --> FT)...')
        for param in self.model.parameters():               # Iterate over the model parameters.
            param.requires_grad = True                      # Enable gradient computation for the parameters.
        # self.optimizer = torch.optim.SGD(                   # Create a new optimizer with a smaller learning rate.
        #     self.model.parameters(),
        #     lr=lr/10,
        #     momentum=0.9
        # )
        self.optimizer.param_groups[0]['lr'] = config['lr2']
        self.optimizer.param_groups[0]['momentum'] = config['momentum2']
        self.optimizer.param_groups[0]['weight_decay'] = config['weight_decay2']
        print('Configuration completed!')
        print(f'New optimizer parameters:\n{self.optimizer}')

    # ===================================================
    def _initial_optimizer_setup(
        self,
        config: dict
    ) -> None:
        """
        Sets up the optimizer according to the provided configuration of hyperparameters (for Ray Tune).

        Args:
            config (dict): Provided configuration for the experiment.
        """

        print('\nAdjusting optimizer according to the provided configuration...')
        self.optimizer.param_groups[0]['lr'] = config['lr']
        self.optimizer.param_groups[0]['momentum'] = config['momentum']
        self.optimizer.param_groups[0]['weight_decay'] = config['weight_decay']
        print('Configuration completed!')
        print(f'New optimizer parameters:\n{self.optimizer}')

    # ===================================================
    def train(
        self,
        config: dict = None
    ) -> None:
        """
        Trains the model based on the provided configuration.

        Args:
            config (dict, optional): Configuration for training. Defaults to None.
        """

        args = config['args']                                                           # Retrieve the arguments from the configuration.
        print(f"Dataloader to compute accuracy: {config['accuracy']}")

        if self.ray_tune or args.load_best_hyperparameters:
            self._initial_optimizer_setup(config)                                       # Adjust optimizer according to the provided configuration.

        for epoch in range(self.epochs_run, config['epochs']):                          # Iterate over the epochs.

            if epoch == config['epochs'] // 2 and args.transfer_learning == 'LP+FT':
                self._adjust_optimizer_for_ft(config)                                   # Adjust the optimizer hyperparameters for fine-tuning.

            print()
            epoch_train_loss, epoch_val_loss = self._run_epoch(epoch)                   # Run the epoch and get the train and validation loss.

            if ((self.global_rank == 0 and not self.ignore_ckpts and not self.ray_tune)
                and (epoch % self.save_every == 0 or epoch == config['epochs'] - 1)):
                self._save_snapshot(epoch)                                              # Save a snapshot of the model.

            if config['accuracy']:                                                      # Compute accuracy on the target dataloader.
                acc_results = accuracy(self.model,
                                       self.dataloader[config['accuracy']],
                                       args.task_name,
                                       self.local_rank)
                for metric in acc_results:
                    print(f'{f"{metric}:".ljust(5)} {acc_results[metric]}')             # Print the accuracy results.

            if self.ray_tune:                                                           # Ray Tune reporting stage.
                if args.task_name == 'multiclass':
                    tune.report(
                        loss=epoch_train_loss,
                        f1_macro=round(acc_results['f1_macro'], NUM_DECIMALS)
                    )
                elif args.task_name == 'multilabel':
                    tune.report(
                        loss=epoch_train_loss,
                        rmse=round(acc_results['rmse'], NUM_DECIMALS)
                    )

            if config['save_csv'] and not self.ray_tune:

                if epoch == 0:
                    header = ['epoch', 'train_loss', 'val_loss'] + list(acc_results.keys())
                    with open(self.csv_path, 'w', newline='') as file:
                        csv_writer = csv.writer(file)
                        csv_writer.writerow(header)                                 # Write the header row in the CSV file (if first epoch).

                data = [epoch_train_loss, epoch_val_loss] + list(acc_results.values())

                data_rounded = [format(elem, f'.{NUM_DECIMALS}f')
                                if not isinstance(elem, list) else elem
                                for elem in data]
                self._save_to_csv([f"{epoch:02d}"]+data_rounded)                    # Save epoch, train loss, val loss, and other metrics to CSV.
