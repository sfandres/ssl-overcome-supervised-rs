import torch
from torch.utils.data import DataLoader
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import csv
import numpy as np

NUM_DECIMALS = 4


def accuracy(model, dataloader, task_name, device):

    # Initialize the probabilities, predictions and labels lists.
    y_prob = []
    y_pred = []
    y_true = []

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

    if task_name == 'multiclass':

        # Concatenate the lists into tensors (option 2).
        y_prob_cpu = torch.cat(y_prob).to('cpu')
        y_pred_cpu = torch.cat(y_pred).to('cpu')
        y_true_cpu = torch.cat(y_true).to('cpu')

        # Compute top1 and top5 accuracy (option 2).
        top1_accuracy = torch.sum(torch.eq(y_pred_cpu, y_true_cpu)).item() / len(y_true_cpu)
        top5_accuracy = torch.sum(torch.topk(y_prob_cpu, k=5, dim=1)[1] == y_true_cpu.view(-1, 1)).item() / len(y_true_cpu)

        acc_dict = {
            'top1': top1_accuracy,
            'top5': top5_accuracy
        }

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
            'rmse_per_class': list(rmse_per_class),
            'mae_per_class': list(mae_per_class)
        }

    return acc_dict


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        loss_fn: torch.nn.modules.loss,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        snapshot_path: str,
    ) -> None:
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.model = model.to(self.local_rank)
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        if os.path.exists(snapshot_path):
            print("\nLoading snapshot")
            self._load_snapshot(snapshot_path)

        # self.model = DDP(self.model, device_ids=[self.local_rank])

    def _load_snapshot(self, snapshot_path: str):
        loc = f"cuda:{self.local_rank}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, targets):
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
        # self.dataloader['train'].sampler.set_epoch(epoch)
        for source, targets in self.dataloader['train']:
            source = source.to(self.local_rank)
            targets = targets.to(self.local_rank)
            loss = self._run_batch(source, targets)
            running_loss += loss * batch_size
        epoch_loss = running_loss / len(self.dataloader['train'].sampler)
        print(f"[GPU{self.global_rank}] | [Epoch: {epoch}] | Loss: {epoch_loss:.4f} | "
              f"Batch size: {batch_size} | Steps: {len(self.dataloader['train'])} | "
              f"Duration: {(time.time()-t0):.2f}s")
        return round(float(epoch_loss), NUM_DECIMALS)

    def _save_snapshot(self, epoch: int):
        snapshot = {
            "MODEL_STATE": self.model.state_dict(),  # self.model.module.state_dict(),
            "EPOCHS_RUN": epoch + 1,                 # +1 so that the training resumes at the same point.
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def _save_to_csv(self, csv_file, data):
        # Open the file in the append mode.
        with open(csv_file, 'a', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(data)

    def train(self, max_epochs: int, args, test: bool = False, save_csv: bool = True):

        for epoch in range(self.epochs_run, max_epochs):

            print()
            epoch_loss = self._run_epoch(epoch)
            if self.global_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)

            if test:
                acc_results = accuracy(self.model, self.dataloader['test'], args.task_name, self.local_rank)
                for metric in acc_results:
                    print(f'{f"{metric}:".ljust(5)} {acc_results[metric]}')

            if save_csv:

                csv_file = os.path.join(
                    os.getcwd(),
                    f'{args.task_name}_pctrain_{args.dataset_train_pc:.3f}_{args.model_name}.csv'
                )

                if epoch == 0:
                    header = ['epoch', 'loss'] + list(acc_results.keys())
                    with open(csv_file, 'w', newline='') as file:
                        csv_writer = csv.writer(file)
                        csv_writer.writerow(header)

                data = [epoch_loss] + list(acc_results.values())

                data_rounded = [format(elem, f'.{NUM_DECIMALS}f') if not isinstance(elem, list) else elem for elem in data]
                self._save_to_csv(csv_file, [f"{epoch:02d}"]+data_rounded)
