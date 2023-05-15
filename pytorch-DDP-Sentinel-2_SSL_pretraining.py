# Custom modules.
from utils.other import is_notebook, build_paths
from utils.reproducibility import set_seed, seed_worker
from utils.dataset import load_dataset_based_on_ratio, GaussianBlur
from utils.computation import pca_computation, tsne_computation
from utils.simsiam import SimSiam
from utils.simclr import SimCLR
from utils.mocov2 import MoCov2
from utils.barlowtwins import BarlowTwins
from utils.graphs import simple_bar_plot

# Arguments and paths.
import os
import sys
import argparse

# PyTorch.
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import SubsetRandomSampler, DataLoader
import torchvision
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
    resnet50,
    ResNet50_Weights
)
from torchvision import transforms
from torchinfo import summary

# For resizing images to thumbnails.
import torchvision.transforms.functional as functional

# Data management.
import numpy as np
import random
import pandas as pd

# SSL library.
import lightly
from lightly.utils.scheduler import cosine_schedule

# Training checks.
from datetime import datetime
import time
import math

# Hyperparameter tunning.
from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

# For plotting.
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.offsetbox as osb
from matplotlib import rcParams as rcp
import seaborn as sns
import plotly.express as px

# For clustering and 2d representations.
from sklearn import random_projection

AVAIL_SSL_MODELS = ['SimSiam', 'SimCLR', 'SimCLRv2', 'BarlowTwins', 'MoCov2']
SEED = 42


def main(args):

    # Enable reproducibility.
    print(f"\n{'torch initial seed:'.ljust(20)} {torch.initial_seed()}")
    g = set_seed(SEED)
    print(f"{'torch current seed:'.ljust(20)} {torch.initial_seed()}")

    # Check torch CUDA
    print(f"\n{'torch.cuda.is_available():'.ljust(32)}"
      f"{torch.cuda.is_available()}")
    print(f"{'torch.cuda.device_count():'.ljust(32)}"
        f"{torch.cuda.device_count()}")
    print(f"{'torch.cuda.current_device():'.ljust(32)}"
        f"{torch.cuda.current_device()}")
    print(f"{'torch.cuda.device(0):'.ljust(32)}"
        f"{torch.cuda.device(0)}")
    print(f"{'torch.cuda.get_device_name(0):'.ljust(32)}"
        f"{torch.cuda.get_device_name(0)}")
    print(f"{'torch.backends.cudnn.benchmark:'.ljust(32)}"
        f"{torch.backends.cudnn.benchmark}")

    # Convert the parsed arguments into a dictionary and declare
    # variables with the same name as the arguments.
    args_dict = vars(args)
    for arg_name in args_dict:
        globals()[arg_name] = args_dict[arg_name]

    # Iterate over the keys of the dictionary and check whether
    # the corresponding variables have been declared.
    print()
    for arg_name in args_dict:
        if arg_name in globals():
            arg_name_col = f'{arg_name}:'
            print(f'{arg_name_col.ljust(20)} {globals()[arg_name]}')
        else:
            print(f'{arg_name} has not been declared')

    # Avoiding the runtimeError: "Too many open files.
    # Communication with the workers is no longer possible."
    if is_notebook() or cluster:
        print(' - Torch sharing strategy set to file_descriptor (default)')
        torch.multiprocessing.set_sharing_strategy('file_descriptor')
    else:
        print(' - Torch sharing strategy set to file_system (less memory)')
        torch.multiprocessing.set_sharing_strategy('file_system')

    # Setting the device.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"{'Device:'.ljust(23)} {device}\n")

    # Build paths.
    cwd = os.getcwd()
    paths = build_paths(cwd, model_name)

    # Show built paths.
    for path in paths:
        path_name_col = f'{path}:'
        print(f'{path_name_col.ljust(20)} {paths[path]}')

    # Size of the images.
    input_size = 224

    # Format of the saved images.
    fig_format = '.png'

    # ======================
    # DATASET.
    # ======================

    #--------------------------
    # Load normalization values.
    #--------------------------
    # Retrieve the path, mean and std values of each split from
    # a .txt file previously generated using a custom script.
    paths[dataset_name], mean, std = load_dataset_based_on_ratio(
        paths['datasets'],
        dataset_name,
        dataset_ratio
    )

    #--------------------------
    # Custom transforms.
    #--------------------------
    splits = ['train', 'val', 'test']

    # Normalization transform (val and test).
    transform = {x: transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean[x],
                            std=std[x])
    ]) for x in splits[1:]}

    # Normalization transform (train).
    # from https://github.com/facebookresearch/simsiam/blob/main/main_simsiam.py
    # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    transform['train'] = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(.4, .4, .4, .1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean['train'],
                            std['train'])
    ])

    for t in transform:
        print(f'\n{t}: {transform[t]}')

    #--------------------------
    # ImageFolder.
    #--------------------------
    # Loading the three datasets with ImageFolder.
    dataset = {x: torchvision.datasets.ImageFolder(
        os.path.join(paths[dataset_name], x)) for x in splits}

    # for d in dataset:
    #     print(f'\n{d}: {dataset[d]}')

    #--------------------------
    # Dealing with imbalanced data (option).
    #--------------------------
    if balanced_dataset:

        # Creating a list of labels of samples.
        train_sample_labels = dataset['train'].targets

        # Calculating the number of samples per label/class.
        class_sample_count = np.unique(train_sample_labels,
                                    return_counts=True)[1]
        print(class_sample_count)

        # Weight per sample not per class.
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in train_sample_labels])

        # Casting.
        samples_weight = torch.from_numpy(samples_weight)
        samples_weigth = samples_weight.double()

        # Sampler, imbalanced data.
        sampler = torch.utils.data.WeightedRandomSampler(
            samples_weight,
            len(samples_weight)
        )
        shuffle = False

    else:
        sampler = None
        shuffle = True

    #--------------------------
    # Creating a reduced subset (option).
    #--------------------------
    if reduced_dataset:

        # Get the number of samples in the full dataset.
        num_samples = len(dataset['train'])

        # Get the labels.
        labels = dataset['train'].targets

        # Get the unique labels and their corresponding counts in the dataset.
        unique_labels, label_counts = np.unique(labels, return_counts=True)

        # Set the percentage of samples you want to keep.
        percent_keep = 0.05

        # Calculate the number of samples to keep for each label.
        num_keep = np.ceil(percent_keep * label_counts).astype(int)

        # Create a list of indices for the samples to keep.
        keep_indices = []
        for i in range(len(unique_labels)):
            label_indices_i = np.where(labels == unique_labels[i])[0]
            np.random.shuffle(label_indices_i)
            keep_indices_i = label_indices_i[:num_keep[i]]
            keep_indices.extend(keep_indices_i)

        # Create a SubsetRandomSampler using the keep indices.
        sampler = SubsetRandomSampler(keep_indices)
        shuffle = False

    else:
        sampler = None
        shuffle = True

    #--------------------------
    # Cast to Lightly dataset.
    #--------------------------
    # Builds a LightlyDataset from a PyTorch (or torchvision) dataset.
    # Returns a tuple (sample, target, fname) when accessed using __getitem__.
    lightly_dataset = {x: lightly.data.LightlyDataset.from_torch_dataset(
        dataset[x]) for x in splits}

    # print()
    # for d in lightly_dataset:
    #     print(f'{d}:\t{lightly_dataset[d]}')

    #--------------------------
    # Collate functions.
    #--------------------------
    # Base class for other collate implementations.
    # This allows training.
    collate_fn = {x: lightly.data.collate.BaseCollateFunction(
        transform[x]) for x in splits}

    # print()
    # for c in collate_fn:
    #     print(f'{c}:\t{collate_fn[c]}')

    #--------------------------
    # PyTorch dataloaders.
    #--------------------------




if __name__ == "__main__":

   # Get arguments.
    parser = argparse.ArgumentParser(
        description="Script for training the self-supervised learning models."
    )

    parser.add_argument('model_name', type=str,
                        choices=AVAIL_SSL_MODELS,
                        help="target SSL model.")

    parser.add_argument('--backbone_name', '-bn', type=str, default='resnet18',
                        choices=['resnet18', 'resnet50'],
                        help="backbone model name (default: resnet18).")

    parser.add_argument('--dataset_name', '-dn', type=str,
                        default='Sentinel2GlobalLULC_SSL',
                        help='dataset name for training '
                            '(default: Sentinel2GlobalLULC_SSL).')

    parser.add_argument('--dataset_ratio', '-dr', type=str,
                        default='(0.900,0.0250,0.0750)',
                        help='dataset ratio for evaluation '
                            '(default: (0.900,0.0250,0.0750)).')

    parser.add_argument('--epochs', '-e', type=int, default=25,
                        help='number of epochs for training (default: 25).')

    parser.add_argument('--batch_size', '-bs', type=int, default=64,
                        help='number of images in a batch during training '
                            '(default: 64).')

    parser.add_argument('--ini_weights', '-iw', type=str, default='random',
                        choices=['random', 'imagenet'],
                        help="initial weights (default: random).")

    parser.add_argument('--show', '-s', action='store_true',
                        help='the images should appear.')

    parser.add_argument('--balanced_dataset', '-bd', action='store_true',
                        help='whether the dataset should be balanced.')

    parser.add_argument('--reduced_dataset', '-rd', action='store_true',
                        help='whether the dataset should be reduced.')

    parser.add_argument('--cluster', '-c', action='store_true',
                        help='the script runs on a cluster (large mem. space).')

    parser.add_argument('--distributed', '-ddp', action='store_true',
                        help='enables multi-node training using Pytorch DDP.')

    parser.add_argument('--torch_compile', '-tc', action='store_true',
                        help='PyTorch 2.0 compile enabled.')

    parser.add_argument('--resume_training', '-r', action='store_true',
                        help='training is resumed from the latest checkpoint.')

    parser.add_argument('--ray_tune', '-rt', type=str,
                        choices=['gridsearch', 'loguniform'],
                        help='enables Ray Tune (tunes everything or only lr).')

    parser.add_argument('--grace_period', '-gp', type=int, default=5,
                        help='only stop trials at least this old in time.')

    parser.add_argument('--num_samples_trials', '-nst', type=int, default=10,
                        help='number of samples to tune the hyperparameters.')

    args = parser.parse_args(sys.argv[1:])

    main(args)