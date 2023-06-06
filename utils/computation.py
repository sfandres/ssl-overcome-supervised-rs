"""Useful functions to process data.

Usage:
    -

Author:
    A.J. Sanchez-Fernandez - 14/02/2023
"""


import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import math
import torch
import random
import numpy as np
import plotly.express as px
import ast


class Experiment:
    """
    Experimental setup.

    Attributes:
        batch_size: Size of the batch of samples to train.
        epochs: Number of epochs to train the model.
        input_size: Size of the input images.
        num_workers: Number of threads for the dataloader.
        seed: Seed for reproducibility purposes.
    """
    def __init__(self, batch_size=128, epochs=10,
                 input_size=224, num_workers=0, seed=42):
        """Inits Experiment with default parameters."""
        self.batch_size = batch_size
        self.epochs = epochs
        self.input_size = input_size
        self.num_workers = num_workers
        self.seed = seed
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def reproducibility(self):
        """Makes the experiments reproducible."""
        # Seed torch and numpy.
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Enable CUDNN deterministic mode.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Issues a warning if it is not met.
        torch.use_deterministic_algorithms(True)
        
        self.g = torch.Generator()
        self.g.manual_seed(self.seed)
        
        # Enable deterministic behavior using external GPU.
        # %env CUBLAS_WORKSPACE_CONFIG=:4096:8
        # %env CUBLAS_WORKSPACE_CONFIG=:16:8
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
        
    def seed_worker(self, worker_id):
        """Seed for the workers of the dataloaders (reproducibility)."""
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)


def create_confusion_matrix(model, dataloader, device, class_names):
    """
    Takes the "model" and "dataloader" and computes
    the confusion matrix required to calculate and
    plot the F1-score using the "device".

    Args:
        model: trained model.
        dataloader: PyTorch DataLoader.
        device: 'cpu' or 'cuda' for PyTorch.
        class_names: names of the classes.

    Returns:
        conf_mat: resulting confusion matrix.
        class_accuracy: accuracy results per class.
    """

    # Initialize the prediction and label lists (tensors).
    pred_list = torch.zeros(0, dtype=torch.long, device=device)
    label_list = torch.zeros(0, dtype=torch.long, device=device)

    # Since we're not training, we don't need to calculate
    # the gradients for our outputs with torch.no_grad():
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader):

            # Dataset.
            inputs, labels = data[0].to(device), data[1].to(device)

            # Predict labels.
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            # Append batch prediction results.
            pred_list = torch.cat([pred_list, predicted.view(-1)])
            label_list = torch.cat([label_list, labels.view(-1)])

    # Copy back to cpu.
    pred_list = pred_list.to('cpu')
    label_list = label_list.to('cpu')

    # Confusion matrix and per-class accuracy.
    conf_mat = confusion_matrix(label_list.numpy(), pred_list.numpy())
    class_accuracy = 100 * conf_mat.diagonal() / conf_mat.sum(1)
    
    # Create dataframe from confusion matrix.
    df_conf_mat = pd.DataFrame(conf_mat,
                               index=class_names,
                               columns=class_names).astype(int)

    # Plot seaborn confusion matrix.
    fig = plt.figure(figsize=(22, 12))
    sns_plot = sns.heatmap(df_conf_mat,
                           annot=True,
                           xticklabels=class_names,
                           yticklabels=class_names,
                           fmt="d")
    plt.show()

    # Print the classification report.
    print(classification_report(pred_list,
                                label_list,
                                target_names=class_names,
                                zero_division=0))

    return conf_mat, class_accuracy
