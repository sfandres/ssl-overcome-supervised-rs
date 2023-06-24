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

    This function takes a trained model, a data loader, the task name, and the device to perform
    computations on. It calculates the accuracy of the model on the provided dataset according to
    the downstream task and returns the result as a dictionary.

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
        for data in dataloader:

            # Dataset.
            inputs, labels = data[0].to(device), data[1].to(device)

            # Forward pass.
            outputs = model(inputs)

            # Get model predictions.
            if task_name == 'multiclass':
                # Convert logits to probabilities using a softmax function.
                probs = torch.softmax(outputs, dim=1)
                # Take the argmax of the probabilities to obtain the predicted class labels.
                preds = torch.argmax(probs, dim=1)

            elif task_name == 'multilabel':
                # Convert logits to probabilities using a sigmoid function.
                probs = torch.sigmoid(outputs)
                # Scale predicted abundances to sum to 1 across all the classes for each sample.
                preds_sum = probs.sum(dim=1, keepdim=True)
                preds = probs / preds_sum

            # Append true and predicted labels to the lists (option 2).
            y_prob.append(probs)
            y_pred.append(preds)
            y_true.append(labels)

    # Accuracy metrics for multi-class downstream task.
    if task_name == 'multiclass':

        # Concatenate the lists into tensors (option 2).
        y_prob_cpu = torch.cat(y_prob).to('cpu')
        y_pred_cpu = torch.cat(y_pred).to('cpu')
        y_true_cpu = torch.cat(y_true).to('cpu')

        # Compute top1 and top5 accuracy (option 2).
        top1_accuracy = torch.sum(torch.eq(y_pred_cpu, y_true_cpu)).item() / len(y_true_cpu)
        top5_accuracy = torch.sum(torch.topk(y_prob_cpu, k=5, dim=1)[1] == y_true_cpu.view(-1, 1)).item() / len(y_true_cpu)

        # F1 metrics.
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

    # Accuracy metrics for multi-label (abundances) downstream task.
    elif task_name == 'multilabel':

        # Concatenate the lists into tensors.
        y_prob_cpu = torch.cat(y_prob).to('cpu')
        y_pred_cpu = torch.cat(y_pred).to('cpu')
        y_true_cpu = torch.cat(y_true).to('cpu')

        # Compute global RMSE and MAE.
        rmse = mean_squared_error(y_true_cpu, y_pred_cpu, squared=False)
        mae = mean_absolute_error(y_true_cpu, y_pred_cpu)

        # Compute RMSE and MAE per class.
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
        ignore_ckpts (bool, optional): Flag indicating whether to ignore checkpoints. Defaults to False.

    Methods:
        _load_snapshot(snapshot_path: str): Load a snapshot from the provided path.
        _save_snapshot(epoch: int): Save a snapshot of the model and optimizer state.
        _run_evaluation(): Run evaluation on the validation dataset and calculate the validation loss.
        _run_batch(source, targets): Run a single batch during training and compute the loss.
        _run_epoch(epoch: int): Run a single epoch of training and compute the training and validation losses.
        _save_to_csv(csv_file, data): Save data to a CSV file.
        _change_from_lp_to_ft(lr: float): Change from LP to FT (transfer learning to fine-tuning).
        train(config: dict = None): Main training loop.
    """

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
        ignore_ckpts: bool = False
    ) -> None:

        # Initialize Trainer object with the provided parameters.
        super().__init__()

        # Assign instance attributes.
        self.model = model
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.save_every = save_every
        self.snapshot_path = snapshot_path
        self.csv_path = csv_path
        self.distributed = distributed
        self.lightly_train = lightly_train
        self.ignore_ckpts = ignore_ckpts

        # Retrieve environment variables.
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])

        # Move the model to the local rank device.
        self.model = model.to(self.local_rank)
        self.epochs_run = 0

        # Load snapshot if it exists and ignore_ckpts is False.
        if os.path.exists(snapshot_path) and not ignore_ckpts:
            print("\nLoading snapshot...")
            self._load_snapshot(snapshot_path)

        # Distributed training with DDP.
        if distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank])

    def _load_snapshot(self, snapshot_path: str):
        loc = f"cuda:{self.local_rank}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        self.optimizer.load_state_dict(snapshot['OPTIMIZER'])
        print(f"Resuming training from snapshot at Epoch {self.epochs_run} <-- {snapshot_path.rsplit('/', 1)[-1]}")

    def _save_snapshot(self, epoch: int):
        if self.distributed:
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        snapshot = {
            "MODEL_STATE": model_state,
            "EPOCHS_RUN": epoch + 1,                    # +1 so that the training resumes at the same point.
            "OPTIMIZER": self.optimizer.state_dict()
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def _run_evaluation(self):
        batch_size = len(next(iter(self.dataloader['val']))[0])
        running_loss = 0.
        self.model.eval()
        with torch.no_grad():
            for source, targets in self.dataloader['val']:
                source = source.to(self.local_rank)
                targets = targets.to(self.local_rank)
                output = self.model(source)
                loss = self.loss_fn(output, targets)
                running_loss += loss * batch_size
        epoch_val_loss = running_loss / len(self.dataloader['val'].sampler)
        return epoch_val_loss

    def _run_batch(self, source, targets):
        source = source.to(self.local_rank)
        targets = targets.to(self.local_rank)
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = self.loss_fn(output, targets)
        loss.backward()
        self.optimizer.step()
        return loss.detach()

    def _run_epoch(self, epoch: int):
        batch_size = len(next(iter(self.dataloader['train']))[0])
        running_loss = 0.
        t0 = time.time()
        if self.distributed:
            self.dataloader['train'].sampler.set_epoch(epoch)
        self.model.train()
        if not self.lightly_train:
            for source, targets in self.dataloader['train']:
                loss = self._run_batch(source, targets)
                running_loss += loss * batch_size
        else:
            for (source, _), targets, _ in self.dataloader['train']:
                loss = self._run_batch(source, targets)
                running_loss += loss * batch_size
        epoch_train_loss = running_loss / len(self.dataloader['train'].sampler)
        epoch_val_loss = self._run_evaluation()
        print(f"[GPU{self.global_rank}] | [Epoch: {epoch}] | Train loss: {epoch_train_loss:.4f} | "
              f"Steps: {len(self.dataloader['train'])} | Val loss: {epoch_val_loss:.4f} | "
              f"Batch size: {batch_size} | lr: {self.optimizer.param_groups[0]['lr']} | "
              f"Duration: {(time.time()-t0):.2f}s")
        return round(float(epoch_train_loss), NUM_DECIMALS), round(float(epoch_val_loss), NUM_DECIMALS)

    def _save_to_csv(self, csv_file, data):
        # Open the file in the append mode.
        with open(csv_file, 'a', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(data)

    def _change_from_lp_to_ft(self, lr: float):
        for param in self.model.parameters():
            param.requires_grad = True
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr/10, momentum=0.9)   # TEN TIMES SMALLER
        print('Changed from LP to FT w/ lr 10 times smaller')

    def train(self, config: dict = None):

        args = config['args']
        max_epochs = args.epochs
        test = config['test']
        save_csv = config['save_csv']

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config['lr'], momentum=0.9)

        for epoch in range(self.epochs_run, max_epochs):

            if epoch == max_epochs // 2 and args.transfer_learning == 'LP+FT':
                self._change_from_lp_to_ft(args.learning_rate)

            print()
            epoch_train_loss, epoch_val_loss = self._run_epoch(epoch)
            if self.global_rank == 0 and not self.ignore_ckpts and (epoch % self.save_every == 0 or epoch == max_epochs - 1):
                self._save_snapshot(epoch)

            if test:
                acc_results = accuracy(self.model, self.dataloader['test'], args.task_name, self.local_rank)
                for metric in acc_results:
                    print(f'{f"{metric}:".ljust(5)} {acc_results[metric]}')

            if save_csv:

                if epoch == 0:
                    header = ['epoch', 'train_loss', 'val_loss'] + list(acc_results.keys())
                    with open(self.csv_path, 'w', newline='') as file:
                        csv_writer = csv.writer(file)
                        csv_writer.writerow(header)

                data = [epoch_train_loss, epoch_val_loss] + list(acc_results.values())

                data_rounded = [format(elem, f'.{NUM_DECIMALS}f') if not isinstance(elem, list) else elem for elem in data]
                self._save_to_csv(self.csv_path, [f"{epoch:02d}"]+data_rounded)
