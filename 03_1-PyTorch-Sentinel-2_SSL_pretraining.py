#!/usr/bin/env python
# coding: utf-8

# **TRAINING SSL MODELS USING THE ENTIRE SENTINEL-2 DATASET**

# Reference: https://docs.lightly.ai/tutorials/package/tutorial_simsiam_esa.html

# ***

# ***

# # Initial configuration

# ## Libraries and modules

# In[ ]:


from utils.other import is_notebook, build_paths

# Load notebook extensions.
if is_notebook():
    get_ipython().run_line_magic('load_ext', 'tensorboard')
    get_ipython().run_line_magic('load_ext', 'pycodestyle_magic')
    get_ipython().run_line_magic('pycodestyle_on', '')
    get_ipython().run_line_magic('env', 'RAY_PICKLE_VERBOSE_DEBUG=1')


# In[ ]:


# Custom modules.
from utils.reproducibility import set_seed, seed_worker
from utils.dataset import load_dataset_based_on_ratio, GaussianBlur
from utils.computation import pca_computation, tsne_computation
from utils.simsiam import SimSiam
from utils.simclr import SimCLR
from utils.simclrv2 import SimCLRv2
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

# Showing images in the notebook.
# from IPython.display import Image
# from IPython.core.display import HTML

# Other imports.
# import copy
# from lightly.utils.debug import std_of_l2_normalized
# import matplotlib.font_manager


# In[ ]:


AVAIL_SSL_MODELS = ['SimSiam', 'SimCLR', 'SimCLRv2', 'BarlowTwins', 'MoCov2']
SEED = 42


# ## Enable reproducibility

# Reference: https://pytorch.org/docs/stable/notes/randomness.html

# In[ ]:


print(f"\n{'torch initial seed:'.ljust(20)} {torch.initial_seed()}")
g = set_seed(SEED)
print(f"{'torch current seed:'.ljust(20)} {torch.initial_seed()}")


# ## Check torch CUDA

# In[ ]:


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


# ## Command line arguments

# In[ ]:


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

parser.add_argument('--cluster', '-c', action='store_true',
                    help='the script runs on a cluster (large mem. space).')

parser.add_argument('--torch_compile', '-tc', action='store_true',
                    help='PyTorch 2.0 compile enabled.')

parser.add_argument('--resume_training', '-r', action='store_true',
                    help='training is resumed from the latest checkpoint.')

parser.add_argument('--ray_tune', '-t', action='store_true',
                    help='hyperparameter tuning with Ray Tune.')

print()


# ## Simulate and get input arguments

# In[ ]:


# Input arguments.
if is_notebook():
    args = parser.parse_args(
        args=[
            'SimSiam',
            '--backbone=resnet18',
            '--dataset_name=Sentinel2GlobalLULC_SSL',
            '--dataset_ratio=(0.020,0.0196,0.9604)',
            '--epochs=25',
            '--batch_size=64',
            '--show',
            '--ray_tune',
            # '--resume_training'
        ]
    )
else:
    args = parser.parse_args(sys.argv[1:])


# In[ ]:


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
print(f"{'Device:'.ljust(20)} {device}")


# In[ ]:


# Setting the model and initial weights.
if backbone_name == 'resnet18':
    if ini_weights == 'imagenet':
        resnet = resnet18(
            weights=ResNet18_Weights.DEFAULT,
            # zero_init_residual=True
        )
    elif ini_weights == 'random':
        resnet = resnet18(
            weights=None,
            # zero_init_residual=True
        )
elif backbone_name == 'resnet50':
    if ini_weights == 'imagenet':
        resnet = resnet50(
            weights=ResNet50_Weights.DEFAULT,
            # zero_init_residual=True
        )
    elif ini_weights == 'random':
        resnet = resnet50(
            weights=None,
            # zero_init_residual=True
        )


# ## Build paths

# In[ ]:


# Get current directory.
cwd = os.getcwd()

# Build paths.
paths = build_paths(cwd, model_name)

# Show built paths.
print()
for path in paths:
    path_name_col = f'{path}:'
    print(f'{path_name_col.ljust(20)} {paths[path]}')


# ## Settings and options

# In[ ]:


# Size of the images.
input_size = 224

# Format of the saved images.
fig_format = '.png'


# ## Load two pretrained models (ignore)

# In[ ]:


# print('\nModel with pretrained weights using SSL')
# resnet18 = torchvision.models.resnet18(weights=None)

# # Only backbone.
# pt_backbone = torch.nn.Sequential(*list(resnet18.children())[:-1])

# # List of trained models.
# model_list = []
# print()
# for root, dirs, files in os.walk(input_dir_models):
#     for i, filename in enumerate(sorted(files, reverse=True)):
#         model_list.append(os.path.join(root, filename))
#         print(f'{i:02} --> {filename}')

# # Loading model.
# idx = 0
# print(f'\nLoaded: {model_list[idx]}')
# pt_backbone.load_state_dict(torch.load(model_list[idx]))

# # Adding a linear layer on top of the model (linear classifier).
# model = torch.nn.Sequential(
#     pt_backbone,
#     torch.nn.Flatten(),
#     torch.nn.Linear(in_features=512, out_features=10, bias=True),
#     # torch.nn.Softmax(dim=1)
# )

# print()
# print(model[0][0])
# print(model[0][0].weight[3])

# # Loading model.
# idx = 5
# print(f'\nLoaded: {model_list[idx]}')
# pt_backbone.load_state_dict(torch.load(model_list[idx]))

# # Adding a linear layer on top of the model (linear classifier).
# model = torch.nn.Sequential(
#     pt_backbone,
#     torch.nn.Flatten(),
#     torch.nn.Linear(in_features=512, out_features=10, bias=True),
#     # torch.nn.Softmax(dim=1)
# )

# print()
# print(model[0][0])
# print(model[0][0].weight[3])


# ***

# ***

# # Dataset

# ## Load normalization values

# In[ ]:


# Retrieve the path, mean and std values of each split from
# a .txt file previously generated using a custom script.
paths[dataset_name], mean, std = load_dataset_based_on_ratio(
    paths['datasets'],
    dataset_name,
    dataset_ratio
)


# ## Custom transforms

# Define the augmentations for self-supervised learning.

# In[ ]:


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


# ## ImageFolder

# In[ ]:


# Loading the three datasets with ImageFolder.
dataset = {x: torchvision.datasets.ImageFolder(
    os.path.join(paths[dataset_name], x)) for x in splits}

for d in dataset:
    print(f'\n{d}: {dataset[d]}')


# ## Dealing with imbalanced data

# In[ ]:


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

print(f'\nSampler:  {sampler}')
print(f'Shuffle:  {shuffle}')


# ## Cast to Lightly dataset

# In[ ]:


# Builds a LightlyDataset from a PyTorch (or torchvision) dataset.
# Returns a tuple (sample, target, fname) when accessed using __getitem__.
lightly_dataset = {x: lightly.data.LightlyDataset.from_torch_dataset(
    dataset[x]) for x in splits}

print()
for d in lightly_dataset:
    print(f'{d}:\t{lightly_dataset[d]}')

# test_data_lightly = lightly.data.LightlyDataset.from_torch_dataset(
#     test_data,
#     transform=test_transform
# )


# ## Collate functions

# PyTorch uses a Collate Function to combine the data in your batches together.
# 
# BaseCollateFunction (base class) takes a batch of images as input and <b>transforms each image into two different augmentations</b> with the help of random transforms. The images are then concatenated such that the output batch is exactly twice the length of the input batch.

# In[ ]:


# Base class for other collate implementations.
# This allows training.
collate_fn = {x: lightly.data.collate.BaseCollateFunction(
    transform[x]) for x in splits}

print()
for c in collate_fn:
    print(f'{c}:\t{collate_fn[c]}')


# **Important note:** These functions could be removed if I implement a custom load dataset with a get_item that gets and tranforms two batches of images.

# ## PyTorch dataloaders

# In[ ]:


# Dataloader for validating and testing.
dataloader = {x: torch.utils.data.DataLoader(
    lightly_dataset[x],
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn[x],
    drop_last=False,
    worker_init_fn=seed_worker if not ray_tune else None,
    generator=g if not ray_tune else None
) for x in splits[1:]}

# Dataloader for training.
dataloader['train'] = torch.utils.data.DataLoader(
    lightly_dataset['train'],
    batch_size=batch_size,
    shuffle=shuffle,
    sampler=sampler,
    num_workers=0,
    collate_fn=collate_fn['train'],
    drop_last=False,
    worker_init_fn=seed_worker if not ray_tune else None,
    generator=g if not ray_tune else None
)

# Check if shuffle is enabled.
if isinstance(dataloader['train'].sampler, torch.utils.data.RandomSampler):
    print('\nShuffle enabled in training!')
else:
    print('\nShuffle disabled in training!')

for d in dataloader:
    print(f"\n{d}:\t{vars(dataloader[d])}")


# ## Check the balance and size of the dataset

# In[ ]:


# Check samples per class, total samples and batches of each dataset.
for d in dataset:
    samples = np.unique(dataset[d].targets, return_counts=True)[1]
    print(f'\n{d}:')
    print(f'  - #Samples/class:\n{samples}')
    print(f'  - #Samples: {len(dataset[d].targets)}')
    print(f'  - #Batches: {len(dataloader[d])}')


# ## Check the distribution of samples in the dataloader (lightly dataset)

# In[ ]:


# # List to save the labels.
# labels_list = []

# # Accessing Data and Targets in a PyTorch DataLoader.
# t0 = time.time()
# for i, (images, labels, names) in enumerate(dataloader['train']):
#     labels_list.append(labels)

# # Concatenate list of lists (batches).
# labels_list = torch.cat(labels_list, dim=0).numpy()
# print(f'\nSample distribution computation in train dataset (s): '
#       f'{(time.time()-t0):.2f}')

# # Count number of unique values.
# data_x, data_y = np.unique(labels_list, return_counts=True)

# # New function to plot (suitable for execution in shell).
# fig, ax = plt.subplots(1, 1, figsize=(20, 5))
# simple_bar_plot(ax,
#                 data_x,
#                 'Class',
#                 data_y,
#                 'N samples (dataloader)')

# plt.gcf().subplots_adjust(bottom=0.15)
# plt.gcf().subplots_adjust(left=0.15)
# fig_name_save = (f'sample_distribution'
#                  f'-ratio={dataset_ratio}'
#                  f'-balanced={handle_imb_classes}')
# fig.savefig(os.path.join(paths['images'], fig_name_save+fig_format),
#             bbox_inches='tight')

# plt.show() if show else plt.close()


# ## Look at some training samples (lightly dataset)

# ### Only one sample from the first batch

# In[ ]:


# Accessing Data and Targets in a PyTorch DataLoader.
if show:
    for i, (images, labels, names) in enumerate(dataloader['train']):
        img = images[0][0]
        label = labels[0]
        print(images[0].shape)
        print(labels.shape)
        plt.title("Label: " + str(int(label)))
        plt.imshow(torch.permute(img, (1, 2, 0)))
        plt.show() if show else plt.close()
        if i == 0:
            break  # Only a few batches.


# ### Two batches

# Note: Comment out the normalization augmentation first to view the images below properly.

# In[ ]:


def show_batch(batch, batch_id):
    """
    Shows the images in the batch.

    Attributes:
        batch: Batch of images.
        batch_id: Batch identification number.
    """

    columns = 8
    rows = 2
    width = 30
    height = 5

    fig = plt.figure(figsize=(width, height))
    fig.suptitle(f'Batch {batch_id}')
    for i in range(1, columns * rows + 1):
        if i < batch_size:
            img = batch[i]
            fig.add_subplot(rows, columns, i)
            plt.imshow(torch.permute(img, (1, 2, 0)))

    plt.show() if show else plt.close()


# Train loop.
if show:
    for b, ((x0, x1), _, _) in enumerate(dataloader['train']):

        # Show the images within the first batch.
        show_batch(x0, 0)
        show_batch(x1, 1)
        break


# Each image is augmented differently in the two batches that are loaded at the same time during training. The dataloader from lightly is capable of providing two batches in one iteration.

# ***

# ***

# # Self-supervised models

# Reference: Lightly tutorials

# In[ ]:


# Model's backbone structure.
if show:
    print(summary(
        resnet,
        input_size=(batch_size, 3, input_size, input_size),
        device=device)
    )


# ## Backbone net

# In[ ]:


# # Dimension of the embeddings.
# num_ftrs = 512

# Dimension of the output of the prediction and projection heads.
out_dim = proj_hidden_dim = 512

# The prediction head uses a bottleneck architecture.
pred_hidden_dim = 128


# In[ ]:


# # Removing head from resnet. Embedding.
# input_dim = resnet.fc.in_features
# hidden_dim = input_dim
# backbone = nn.Sequential(*list(resnet.children())[:-1])

# if model_name == 'SimSiam':
#     model = SimSiam(backbone=backbone,
#                     input_dim=input_dim,
#                     proj_hidden_dim=proj_hidden_dim,
#                     pred_hidden_dim=pred_hidden_dim,
#                     output_dim=out_dim)
# else:
#     model = globals()[model_name](backbone=backbone,
#                                   input_dim=input_dim,
#                                   hidden_dim=hidden_dim,
#                                   output_dim=out_dim)


# ## Training

# <p style="color:red"><b>-----------------------------------------------------------------</b></p>
# <p style="color:red"><b>----------> REVISED UP TO THIS POINT -----------</b></p>
# <p style="color:red"><b>-----------------------------------------------------------------</b></p>

# In[ ]:


# # Model's backbone structure.
# if show:
#     print(summary(
#         model.backbone,
#         input_size=(batch_size, 3, input_size, input_size),
#         device=device)
#     )


# ### Loop

# In[ ]:


def train(
    config: dict
):

    # ======================
    # DEFINE MODEL.
    # Removing head from resnet. Embedding.
    input_dim = resnet.fc.in_features
    hidden_dim = input_dim
    backbone = nn.Sequential(*list(resnet.children())[:-1])

    if model_name == 'SimSiam':
        model = SimSiam(backbone=backbone, input_dim=input_dim, proj_hidden_dim=proj_hidden_dim, pred_hidden_dim=pred_hidden_dim, output_dim=out_dim)
    elif model_name == 'SimCLR':
        model = SimCLR(backbone=backbone, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=out_dim)
    elif model_name == 'SimCLRv2':
        model = SimCLRv2(backbone=backbone, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=out_dim)
    elif model_name == 'BarlowTwins':
        model = BarlowTwins(backbone=backbone, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=out_dim)
    elif model_name == 'MoCov2':
        model = MoCov2(backbone=backbone, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=out_dim)

    # ======================
    # ADDING GPU SUPPORT.
    # Compile model (only for PT2.0).
    if torch_compile:
        model = torch.compile(model)
        torch.set_float32_matmul_precision('high')

    # Device used for training.
    print(f'\nUsing {device} device')
    print(f'Model name: {model_name}')
    model.to(device)

    # ======================
    # CONFIGURE OPTIMIZER AND SCHEDULERS.
    # Set the initial learning rate.
    if ray_tune:
        lr_init = config["lr"]
    else:
        lr_init = 0.2

    # Use SGD with momentum and weight decay.
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr_init,
        momentum=0.9,
        weight_decay=5e-4
    )

    # Define the warmup duration.
    warmup_epochs = max(1, int(.05*epochs))
    warmup_epochs = 5

    # Linear warmup for the first defined epochs.
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: min(1, epoch / warmup_epochs),
        verbose=True
    )

    # Cosine decay afterwards.
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs-warmup_epochs,
        verbose=True
    )

    # ======================
    # LOAD CHECKPOINTS (IF ENABLED).
    if resume_training:

        # List of checkpoints.
        ckpt_list = []
        print()
        for root, dirs, files in os.walk(paths['checkpoints']):
            for i, filename in enumerate(sorted(files, reverse=True)):
                if filename[:4] == 'ckpt':
                    ckpt_list.append(os.path.join(root, filename))
                    print(f'{i:02} --> {filename}')

        # Load the best checkpoint.
        print(f'\nLoaded: {ckpt_list[0]}')
        ckpt = torch.load(ckpt_list[0])

        # Load from dict.
        epoch = ckpt['epoch'] + 1
        model.backbone.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        warmup_scheduler.load_state_dict(ckpt['warmup_scheduler_state_dict'])
        cosine_scheduler.load_state_dict(ckpt['cosine_scheduler_state_dict'])

    else:
        epoch = 0  # start training from scratch.

    # ======================
    # INITIAL PARAMETERS.
    save_interval = 2
    total_train_batches = len(dataloader['train'])
    total_val_batches = len(dataloader['val'])
    collapse_level = 0.
    momentum_val = None  # only for moco.
    print(f'optimizer:\n{optimizer}')
    print(f'warmup_scheduler:\n{warmup_scheduler}')
    print(f'cosine_scheduler:\n{cosine_scheduler}')
    print(f"Initial lr: {optimizer.param_groups[0]['lr']}")

    # ======================
    # TRAINING LOOP.
    # Iterating over the epochs.
    print(f'\nBatches in (train, val) datasets: '
          f'({total_train_batches}, {total_val_batches})\n')
    for epoch in range(epoch, epochs):

        if model_name == 'MoCov2':
            momentum_val = cosine_schedule(epoch, epochs, 0.996, 1)
            print(f"Momentum value: {momentum_val}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")

        # Timer added.
        t0 = time.time()

        # ======================
        # TRAINING COMPUTATION.
        # Iterating through the dataloader (lightly dataset is different).
        model.train()
        running_train_loss = 0.
        for b, ((x0, x1), _, _) in enumerate(dataloader['train']):

            # Move images to the GPU (same batch two transformations).
            x0 = x0.to(device)
            x1 = x1.to(device)

            # Zero the parameter gradients.
            optimizer.zero_grad()

            # Forward + backward + optimize: Compute the loss, run
            # backpropagation, and update the parameters of the model.
            loss = model.training_step((x0, x1), momentum_val=momentum_val)
            loss.backward()
            optimizer.step()

            if model_name == 'SimSiam':
                model.check_collapse(loss.item())

            # Print statistics.
            # Averaged loss across all training examples * batch_size.
            running_train_loss += loss.item() * batch_size

            # Show partial stats.
            if b % (total_train_batches//4) == (total_train_batches//4-1):
                print(f'T[{epoch}, {b + 1:5d}] | '
                      f'Running train loss: '
                      f'{running_train_loss/(b*batch_size):.4f}')

        # The level of collapse is large if the standard deviation of
        # the l2 normalized output is much smaller than 1 / sqrt(dim).
        if model_name == 'SimSiam':
            collapse_level = max(
                0., 1 - math.sqrt(out_dim) * model.avg_output_std)

        # ======================
        # TRAINING LOSS.
        # Loss averaged across all training examples for the current epoch.
        epoch_train_loss = (running_train_loss
                            / len(dataloader['train'].sampler))

        # ======================
        # EVALUATION COMPUTATION.
        # The evaluation process was not okey (it's been deleted).
        model.eval()
        running_val_loss = 0.
        # val_loss = 0.0
        # val_steps = 0
        with torch.no_grad():
            for vb, ((x0, x1), _, _) in enumerate(dataloader['val']):

                # Move images to the GPU (same batch two transformations).
                x0 = x0.to(device)
                x1 = x1.to(device)

                # Compute loss.
                loss = model.training_step((x0, x1), momentum_val=momentum_val)

                # Averaged loss across all validation examples * batch_size.
                running_val_loss += loss.item() * batch_size

                # val_loss += loss.cpu().numpy()
                # val_steps += 1

                # Show partial stats.
                if vb % (total_val_batches//4) == (total_val_batches//4-1):
                    print(f'V[{epoch}, {vb + 1:5d}] | '
                          f'Running val loss:   '
                          f'{running_val_loss/(vb*batch_size):.4f}')

        # ======================
        # VALIDATION LOSS.
        # Loss averaged across all training examples for the current epoch.
        epoch_val_loss = (running_val_loss
                          / len(dataloader['val'].sampler))

        # ======================
        # UPDATE LEARNING RATE SCHEDULER.
        # scheduler.step()
        if (epoch < warmup_epochs):
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        # ======================
        # SAVING CHECKPOINT.
        # Move the model to CPU before saving it
        # to make it more platform-independent.
        # Problems with resuming training.
        # model.to('cpu')
        model.save(
            backbone_name,
            epoch,
            epoch_train_loss,
            dataset_ratio,
            balanced_dataset,
            paths['checkpoints'],
            collapse_level=collapse_level if model_name == 'SimSiam' else 0.
        )

        if epoch % save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.backbone.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'warmup_scheduler_state_dict': warmup_scheduler.state_dict(),
                'cosine_scheduler_state_dict': cosine_scheduler.state_dict(),
                'loss': epoch_train_loss
            }, os.path.join(paths['checkpoints'], 'ckpt_' + str(model)))

        # model.to(device)

        # ======================
        # EPOCH STATISTICS.
        # Show some stats per epoch completed.
        print(f'[Epoch {epoch:3d}] | '
              f'Train loss: {epoch_train_loss:.4f} | '
              f'Val loss: {epoch_val_loss:.4f} | '
              f'Duration: {(time.time()-t0):.2f} s | '
              f'Collapse Level (SimSiam only): {collapse_level:.4f}/1.0\n')

        # ======================
        # RAY TUNE.
        if ray_tune:
            tune.report(loss=epoch_val_loss)


# ## Hyperparameter tuning: Ray Tune

# In[ ]:


if ray_tune:

    max_num_epochs = epochs
    num_samples = 10
    gpus_per_trial = 1
    paths['ray_tune'] = os.path.join(paths['output'], 'ray_results')

    config = {
        # "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        # "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "lr": tune.loguniform(1e-4, 1e-1),
        # "batch_size": tune.choice([2, 4, 8, 16])
    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(
        # ``parameter_columns=["l1", "l2", "lr", "batch_size"]``,
        # metric_columns=["loss", "accuracy", "training_iteration"])
        metric_columns=["loss", "training_iteration"])

    result = tune.run(
        partial(train),
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        name=model_name,
        config=config,
        num_samples=num_samples,
        local_dir=paths['ray_tune'],
        scheduler=scheduler,
        verbose=1,
        progress_reporter=reporter)


# In[ ]:


if ray_tune:

    # Get a dataframe for the last reported results of all of the trials.
    df = result.results_df
    df.to_csv(os.path.join(paths['ray_tune'], f'ray_tune_results_df_{model_name}.csv'))

    # Get best results.
    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final val loss: {best_trial.last_result['loss']}")


# In[ ]:


get_ipython().run_line_magic('tensorboard', '--port 6006 --logdir ./output/ray_results/')


# ## Normal training

# Collapse level: the closer to zero the better

# A value close to 0 indicates that the representations have collapsed. A value close to 1/sqrt(dimensions), where dimensions are the number of representation dimensions, indicates that the representations are stable. 

# In[ ]:


if not ray_tune:
    config = None
    train(config)


# ### Checking the weights of the last model

# In[ ]:


# First convolutional layer weights.
# print(model.backbone[0])
# print(model.backbone[0].weight[63])
print(model.backbone.conv1)
print(model.backbone.conv1.weight[63])


# ***

# ***

# # Reduce dimensionality

# ## Calculate embeddings

# In[ ]:


# Empty lists.
embeddings = []
labels = []

# Disable gradients for faster calculations.
# Put the model in evaluation mode.
model.eval()
with torch.no_grad():
    # for i, (x, y, fnames) in enumerate(dataloader_val):
    # Now taking only the first transformed batch.
    for i, ((x, _), y, fnames) in enumerate(dataloader['val']):

        # Move the images to the GPU.
        x = x.to(device)
        y = y.to(device)

        # Embed the images with the pre-trained backbone.
        emb = model.backbone(x).flatten(start_dim=1)

        # Store the embeddings and filenames in lists.
        embeddings.append(emb)
        labels.append(y)

# Concatenate the embeddings and convert to numpy.
embeddings = torch.cat(embeddings, dim=0).to('cpu').numpy()
labels = torch.cat(labels, dim=0).to('cpu').numpy()

# Show shapes.
print(np.shape(embeddings))
print(np.shape(labels))


# # PCA

# In[ ]:


plot = 'all'


# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

# In[ ]:


# PCA computation.
df = pca_computation(embeddings, labels, SEED)

# 2-D plot.
if plot == '2d' or plot == "23d" or plot == 'all':
    fig = plt.figure(figsize=(10, 10))
    sns.scatterplot(
        x='pca_x',
        y='pca_y',
        hue='labels',
        palette=sns.color_palette('hls', 29),
        data=df,
        legend='full',
        alpha=0.9
    )
    fig_name_save = (f'pca_2d-{model}')
    fig.savefig(os.path.join(paths['images'], fig_name_save+fig_format),
                bbox_inches='tight')
    plt.show() if show else plt.close()

# 3-D plot with matplotlib.
if plot == '3d' or plot == "23d" or plot == 'all':
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(
        xs=df['pca_x'],
        ys=df['pca_y'],
        zs=df['pca_z'],
        c=df['labels'],
        cmap='tab10'
    )
    ax.set_xlabel('pca_x')
    ax.set_ylabel('pca_y')
    ax.set_zlabel('pca_z')
    fig_name_save = (f'pca_3d-{model}')
    fig.savefig(os.path.join(paths['images'], fig_name_save+fig_format),
                bbox_inches='tight')
    plt.show() if show else plt.close()

# # 3-D plot with pyplot.
# if (plot == '3d-plotly' or plot == 'all') and show:
#     fig = px.scatter_3d(df, x='pca_x',
#                         y='pca_y', z='pca_z',
#                         color='labels',
#                         width=1000, height=800)  # symbol='labels'

#     fig.update_traces(marker=dict(size=3))

#     # Move colorbar.
#     # fig.update_layout(coloraxis_colorbar=dict(yanchor="top", y=1, x=0,
#     #                                           ticks="outside",
#     #                                           ticksuffix=""))

#     fig.show()


# # t-SNE

# https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html

# In[ ]:


# t-SNE computation for 2-D.
df = tsne_computation(embeddings, labels, SEED, n_components=2)

# 2-D plot.
if plot == '2d' or plot == "23d" or plot == 'all':
    fig = plt.figure(figsize=(10, 10))
    sns.scatterplot(
        x='tsne_x',
        y='tsne_y',
        hue='labels',
        palette=sns.color_palette('hls', 29),
        data=df,
        legend='full',
        alpha=0.9
    )
    fig_name_save = (f'tsne_2d-{model}')
    fig.savefig(os.path.join(paths['images'], fig_name_save+fig_format),
                bbox_inches='tight')
    plt.show() if show else plt.close()

# t-SNE computation for 3-D.
df = tsne_computation(embeddings, labels, SEED, n_components=3)

# 3-D plot with matplotlib.
if plot == '3d' or plot == "23d" or plot == 'all':
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(
        xs=df['tsne_x'],
        ys=df['tsne_y'],
        zs=df['tsne_z'],
        c=df['labels'],
        cmap='tab10'
    )
    ax.set_xlabel('tsne_x')
    ax.set_ylabel('tsne_y')
    ax.set_zlabel('tsne_z')
    fig_name_save = (f'tsne_3d-{model}')
    fig.savefig(os.path.join(paths['images'], fig_name_save+fig_format),
                bbox_inches='tight')
    plt.show() if show else plt.close()

# # 3-D plot with pyplot.
# if (plot == '3d-plotly' or plot == 'all') and show:
#     fig = px.scatter_3d(df, x='tsne_x',
#                         y='tsne_y', z='tsne_z',
#                         color='labels',
#                         width=1000, height=800)  # symbol='labels'

#     fig.update_traces(marker=dict(size=3))

#     # Move colorbar.
#     # fig.update_layout(coloraxis_colorbar=dict(yanchor="top", y=1, x=0,
#     #                                           ticks="outside",
#     #                                           ticksuffix=""))

#     fig.show()


# ***

# ***

# # Check each model's performance/collapse on val data

# ADD SOFTMAX IF NECESSARY!!

# In[ ]:


# def get_scatter_plot_with_thumbnails_axes(ax, title=''):
#     """
#     Creates a scatter plot with image overlays
#     that are plotted in a particular ax position.

#     """

#     # Shuffle images and find out which images to show.
#     shown_images_idx = []
#     shown_images = np.array([[1., 1.]])
#     iterator = [i for i in range(embeddings_2d.shape[0])]
#     np.random.shuffle(iterator)
#     for i in iterator:

#         # Only show image if it is sufficiently far away from the others.
#         dist = np.sum((embeddings_2d[i] - shown_images) ** 2, 1)
#         if np.min(dist) < 2e-3:
#             continue
#         shown_images = np.r_[shown_images, [embeddings_2d[i]]]
#         shown_images_idx.append(i)

#     # Plot image overlays.
#     for idx in shown_images_idx:
#         thumbnail_size = int(rcp['figure.figsize'][0] * 2.5)  # 2.
#         path = os.path.join(data_path_test, filenames[idx])
#         img = Image.open(path)
#         img = functional.resize(img, thumbnail_size)
#         img = np.array(img)
#         img_box = osb.AnnotationBbox(
#             osb.OffsetImage(img, cmap=plt.cm.gray_r),
#             embeddings_2d[idx],
#             pad=0.2,
#         )
#         ax.add_artist(img_box)

#     # Set aspect ratio.
#     ratio = 1. / ax.get_data_ratio()
#     ax.set_aspect(ratio, adjustable='box')
#     ax.title.set_text(title)


# In[ ]:


# # Validation dataset.
# print(f'\nValidating...')
# data_path_test = os.path.join(data_path_target, 'val')
# print(f'Path to dataset: {data_path_test}')

# # List of trained models.
# print('List of model checkpoints:')
# model_list = []
# for root, dirs, files in os.walk(output_path_models):
#     for i, filename in enumerate(sorted(files, reverse=False)):
#         model_list.append(os.path.join(root, filename))
#         print(f'{i:02}: {filename}')

# # Plot setup.
# ncols = 5
# nrows = int(math.ceil(len(model_list) / ncols))

# fig, axes = plt.subplots(nrows=nrows,
#                          ncols=ncols,
#                          figsize=(12*ncols, 12*nrows))

# # Convert the array to 1 dimension.
# axes = axes.ravel()

# # Main loop over the models.
# for model_id, model_name_local in enumerate(model_list):

#     # Load model weights.
#     model.backbone.load_state_dict(torch.load(model_name_local))

#     # Empty lists.
#     embeddings = []
#     filenames = []

#     # Disable gradients for faster calculations.
#     # Put the model in evaluation mode.
#     model.eval()
#     with torch.no_grad():
#         # for i, (x, _, fnames) in enumerate(dataloader_val):
#         for i, ((x, _), _, fnames) in enumerate(dataloader_val_lightly):

#             # Move the images to the GPU.
#             x = x.to(device)

#             # Embed the images with the pre-trained backbone.
#             y = model.backbone(x).flatten(start_dim=1)

#             # Store the embeddings and filenames in lists.
#             embeddings.append(y)
#             filenames = filenames + list(fnames)

#     # Concatenate the embeddings and convert to numpy.
#     embeddings = torch.cat(embeddings, dim=0)
#     embeddings = embeddings.cpu().numpy()

#     # For the scatter plot we want to transform the images to a
#     # 2-D vector space using a random Gaussian projection.
#     projection = random_projection.GaussianRandomProjection(
#         n_components=2,
#         random_state=seed
#     )
#     embeddings_2d = projection.fit_transform(embeddings)

#     # Normalize the embeddings to fit in the [0, 1] square.
#     M = np.max(embeddings_2d, axis=0)
#     m = np.min(embeddings_2d, axis=0)
#     embeddings_2d = (embeddings_2d - m) / (M - m)

#     # Get a scatter plot with thumbnail overlays.
#     start_chr_epoch = model_name_local.find('-epoch') + 1
#     start_chr_time = model_name_local.find('-time')
#     get_scatter_plot_with_thumbnails_axes(
#         axes[model_id],
#         title=model_name_local[start_chr_epoch:start_chr_time]
#     )

#     # Show progress.
#     print(f'Subplot of model-{model_id} done!',
#           end='\r',
#           flush=True)

# # Save figure.
# fig.suptitle(f'{model_name}')
# fig_name_save = (f'knn-{model}')
# fig.savefig(os.path.join(output_path_figs, fig_name_save+fig_format),
#             bbox_inches='tight')
# if show:
#     plt.show()
# else:
#     plt.close()


# ***

# ***

# # Embeddings for the samples of the test dataset (WARNING: custom)

# ## Setup (NOT WORKING PROPERLY)

# In[ ]:


# # Test dataset.
# print('\nTesting...')
# data_path_test = os.path.join(data_path_target, 'test')
# print(f'Path to dataset: {data_path_test}')

# # Load best model weights.
# idx = -1

# # Print model.
# print(f'Target model checkpoint: {model_list[idx]}')
# model.backbone.load_state_dict(torch.load(model_list[idx]))


# ## Compute embeddings

# In[ ]:


# # Empty lists.
# embeddings = []
# filenames = []

# # Disable gradients for faster calculations.
# # Put the model in evaluation mode.
# model.eval()
# with torch.no_grad():
#     for i, (x, _, fnames) in enumerate(dataloader_test):

#         # Move the images to the GPU.
#         x = x.to(device)

#         # Embed the images with the pre-trained backbone.
#         y = model.backbone(x).flatten(start_dim=1)

#         # Store the embeddings and filenames in lists.
#         embeddings.append(y)
#         filenames = filenames + list(fnames)

# # Concatenate the embeddings and convert to numpy.
# embeddings = torch.cat(embeddings, dim=0)
# embeddings = embeddings.cpu().numpy()


# ## Projection to 2D space

# In[ ]:


# # For the scatter plot we want to transform the images to a two-dimensional
# # vector space using a random Gaussian projection.
# projection = random_projection.GaussianRandomProjection(
#     n_components=2,
#     random_state=seed
# )
# embeddings_2d = projection.fit_transform(embeddings)

# # Normalize the embeddings to fit in the [0, 1] square.
# M = np.max(embeddings_2d, axis=0)
# m = np.min(embeddings_2d, axis=0)
# embeddings_2d = (embeddings_2d - m) / (M - m)


# ## Scatter plots

# In[ ]:


# # Initialize empty figure and add subplot.
# fig = plt.figure(figsize=(15, 15))
# ax = fig.add_subplot(1, 1, 1)

# # Get a scatter plot with thumbnail overlays.
# get_scatter_plot_with_thumbnails_axes(
#     ax,
#     title='Scatter plot with samples'
# )

# # Save figure.
# fig_name_save = (f'scatter_samples-{model}')
# fig.savefig(os.path.join(output_path_figs, fig_name_save+fig_format),
#             bbox_inches='tight')
# if show:
#     plt.show()
# else:
#     plt.close()


# ## Nearest Neighbors

# ### Pick up one random sample per class

# In[ ]:


# # List of subdirectories (classes).
# directory_list = []
# for root, dirs, files in os.walk(data_path_test):
#     for dirname in sorted(dirs):
#         directory_list.append(os.path.join(root, dirname))
#         # print(dirname)


# In[ ]:


# # List of files (samples).
# example_images = []
# for classes in directory_list:

#     # Random samples.
#     random_file = np.random.choice(os.listdir(classes))
#     path_to_random_file = classes + '/' + random_file

#     # Only class and filename.
#     start_chr = path_to_random_file.index('test/') + 5

#     # Append filename.
#     example_images.append(path_to_random_file[start_chr:])
#     # print(example_images)


# ### Look for similar images

# In[ ]:


# def get_image_as_np_array(filename: str):
#     """
#     Loads the image with filename and returns it as a numpy array.

#     """
#     img = Image.open(filename)
#     return np.asarray(img)


# def get_image_as_np_array_with_frame(filename: str, w: int = 5):
#     """
#     Returns an image as a numpy array with a black frame of width w.

#     """
#     img = get_image_as_np_array(filename)
#     ny, nx, _ = img.shape

#     # Create an empty image with padding for the frame.
#     framed_img = np.zeros((w + ny + w, w + nx + w, 3))
#     framed_img = framed_img.astype(np.uint8)

#     # Put the original image in the middle of the new one.
#     framed_img[w:-w, w:-w] = img
#     return framed_img


# def plot_nearest_neighbors_nxn(example_image: str, i: int):
#     """
#     Plots the example image and its eight nearest neighbors.

#     """
#     n_subplots = 6

#     # Initialize empty figure.
#     fig = plt.figure(figsize=(10, 10))
#     fig.suptitle(f"Nearest Neighbor Plot Class {i}")

#     # Get indexes.
#     example_idx = filenames.index(example_image)

#     # Get distances to the cluster center.
#     distances = embeddings - embeddings[example_idx]
#     distances = np.power(distances, 2).sum(-1).squeeze()

#     # Sort indices by distance to the center.
#     nearest_neighbors = np.argsort(distances)[:n_subplots]

#     # Show images.
#     for plot_offset, plot_idx in enumerate(nearest_neighbors):
#         ax = fig.add_subplot(3, 3, plot_offset + 1)

#         # Get the corresponding filename.
#         fname = os.path.join(data_path_test, filenames[plot_idx])
#         if plot_offset == 0:
#             ax.set_title(f"Example Image")
#             plt.imshow(get_image_as_np_array_with_frame(fname))
#         else:
#             plt.imshow(get_image_as_np_array(fname))

#         # Let's disable the axis.
#         plt.axis("off")

#     # Save figure.
#     fig_name_save = (f'knn_per_class-c={i:02}-{model}')
#     fig.savefig(os.path.join(output_path_figs, fig_name_save+fig_format),
#                 bbox_inches='tight')
#     if show:
#         pass  # plt.show()
#     else:
#         plt.close()


# In[ ]:


# # Show example images for each cluster.
# for i, example_image in enumerate(example_images):
#     plot_nearest_neighbors_nxn(example_image, i)


# In[ ]:




