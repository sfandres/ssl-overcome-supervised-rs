#!/usr/bin/env python
# coding: utf-8

# **FIRST ATTEMP TO APPLY SSL TO THE SENTINEL-2 DATASET**

# Reference tutorial: https://docs.lightly.ai/tutorials/package/tutorial_simsiam_esa.html

# In[ ]:


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__

        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)

    except NameError:
        return False      # Probably standard Python interpreter


# In[ ]:


if is_notebook():
    get_ipython().run_line_magic('load_ext', 'pycodestyle_magic')


# In[ ]:


if is_notebook():
    get_ipython().run_line_magic('pycodestyle_on', '')


# In[ ]:


# def create_arg_parser():
#     """Creates and returns the ArgumentParser object."""
#     ...
#     return parser


# ***

# ***

# # Imports

# ## Packages and modules

# In[ ]:


# Main.
import utils

# OS module.
import os

# PyTorch.
import torch
import torchvision
from torchinfo import summary

# Data management.
import numpy as np
import pandas as pd

# Lightly.
import lightly

# Training checks.
from datetime import datetime
import time
import copy
import math

from lightly.utils.debug import std_of_l2_normalized

# Showing images in the notebook.
import IPython

# For plotting.
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.offsetbox as osb
from matplotlib import rcParams as rcp
import seaborn as sns

# For resizing images to thumbnails.
import torchvision.transforms.functional as functional

# For clustering and 2d representations.
from sklearn import random_projection

from graphs import simple_bar_plot
from utils import pca_computation, tsne_computation
import plotly.express as px


# ## Parser

# In[ ]:


# Parser (get arguments).
if __name__ == '__main__':
    import argparse
    import sys
    parser = argparse.ArgumentParser(
        description="Script for training the self-supervised learning models."
    )
    parser.add_argument(
        'model',
        type=str,
        choices=['simsiam', 'simclr', 'barlowtwins'],
        help=('SSL model for training: '
              'use "simsiam", "simclr" or "barlowtwins".')
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='Sentinel2GlobalLULC',
        help='Dataset name for training.'
    )
    parser.add_argument(
        '--balanced_dataset',
        type=bool,
        default=False,
        help='Whether the dataset should be balanced.'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='Number of epochs for training.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Number of images in a batch during training.'
    )
    parser.add_argument(
        '--ini_weights',
        type=str,
        default='random',
        choices=['random', 'imagenet'],
        help='Initial weights: use "random" (default) or "imagenet".'
    )
    parser.add_argument(
        '--show_fig',
        type=bool,
        default=True,
        help='Whether the images should appear.'
    )
    parser.add_argument(
        '--cluster',
        type=bool,
        default=False,
        help=('Whether the script runs on a cluster '
              '(large memory space available).')
    )

if is_notebook():
    args = parser.parse_args(args=['simsiam'])  # , '--ini_weights', 'imagenet'
else:
    args = parser.parse_args(sys.argv[1:])


# ## Settings

# In[ ]:


# Target SSL model.
model_name = args.model
print(f'\nTarget model for training: {model_name}')

# Target dataset.
dataset_name = args.dataset
print(f'Target dataset: {dataset_name}')

# Handling class imbalance.
handle_imb_classes = args.balanced_dataset
print(f'Balanced dataset: {handle_imb_classes}')

# Setting number of epochs.
epochs = args.epochs
print(f'Number of epochs: {epochs}')

# Setting batch size.
batch_size = args.batch_size
print(f'Batch size: {batch_size}')

# Setting the initial weights.
if args.ini_weights == 'imagenet':
    weights = torchvision.models.ResNet18_Weights.DEFAULT
else:
    weights = None
print(f'Initial weights: {weights}')

# Show figures.
show = args.show_fig
print(f'Showing figures: {show}')

# Supercomputer?.
cluster = args.cluster
print(f'\nExecution on cluster: {cluster}')

# Avoiding the runtimeError: Too many open files.
# Communication with the workers is no longer possible.
if is_notebook() or cluster:
    print(f'Execution on jupyter or cluster: '
          f'Torch sharing strategy set to file_system (default)')
    torch.multiprocessing.set_sharing_strategy('file_descriptor')
else:
    print(f'Execution on shell with few resources: '
          'Torch sharing strategy set to file_system')
    torch.multiprocessing.set_sharing_strategy('file_system')


# In[ ]:


# Hyperparamenters.
exp = utils.Experiment(epochs=epochs,
                       batch_size=batch_size)
print(f'\nDevice: {exp.device}\n')

# Get current directory.
cwd = os.getcwd()
print(f'Working directory: {cwd}')

# Directory to save the model checkpoint.
output_dir_model = os.path.join(os.path.join(cwd, 'pytorch_models'),
                                model_name)
print(f'Directory to store model checkpoints: {output_dir_model}')

# Folder to save the figures.
output_dir_fig = os.path.join(os.path.join(cwd, 'figures'), model_name)
print(f'Directory to save figures: {output_dir_fig}')

# Directory where the datasets are stored.
datasets_dir = os.path.join(os.path.join(cwd, 'datasets'), dataset_name)
print(f'\nInput directory of datasets: {datasets_dir}')

# Figure format.
fig_format = '.png'  # .pdf


# In[ ]:


# Dimension of the embeddings.
num_ftrs = 512

# Dimension of the output of the prediction and projection heads.
out_dim = proj_hidden_dim = 512

# The prediction head uses a bottleneck architecture.
pred_hidden_dim = 128


# ## Reproducibility

# In[ ]:


exp.reproducibility()


# ***

# ***

# # Loading dataset

# In[ ]:


# Get the subsets with full path.
data_dirs = utils.listdir_fullpath(datasets_dir)

# Leave out unwanted subsets
# data_dirs = data_dirs[2:]
for dirs in data_dirs:
    print(dirs)

# Select the target dataset.
data_dir_target = data_dirs[2]
print('\nSelected: ' + data_dir_target)

# Ratio.
ratio = data_dir_target[
    data_dir_target.index("("):data_dir_target.index(")")+1
]
print(f'Ratio: {ratio}')

# Load mean and std from file.
mean, std = utils.load_mean_std_values(data_dir_target)
print(f'Mean loaded from .txt: {mean}')
print(f'Std loaded from .txt: {std}')


# ## Custom tranforms (w/o normalization)

# Define the augmentations for self-supervised learning.

# In[ ]:


# from https://github.com/facebookresearch/simsiam/blob/main/main_simsiam.py
import random
from PIL import ImageFilter


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


# MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
train_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((exp.input_size, exp.input_size)),
    torchvision.transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
    torchvision.transforms.RandomApply([
        torchvision.transforms.ColorJitter(.4, .4, .4, .1)  # not strengthened
    ], p=0.8),
    torchvision.transforms.RandomGrayscale(p=0.2),
    torchvision.transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean['train'], std['train'])
])


# In[ ]:


# # Data augmentations for the train dataset.
# train_transform = torchvision.transforms.Compose([
#     torchvision.transforms.Resize((exp.input_size, exp.input_size)),
#     torchvision.transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
#     torchvision.transforms.RandomApply([
#             torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
#         ], p=0.8),  # not strengthened
#     torchvision.transforms.RandomGrayscale(p=0.2),
#     # torchvision.transforms.RandomApply([
#     #     simsiam.loader.GaussianBlur([.1, 2.])
#     # ], p=0.5),
#     torchvision.transforms.RandomHorizontalFlip(),
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.Normalize(mean['train'], std['train'])
# ])

# Data augmentations for the val and test datasets.
val_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((exp.input_size, exp.input_size)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean['val'], std['val'])
])

# Data augmentations for the val and test datasets.
test_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((exp.input_size, exp.input_size)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean['test'], std['test'])
])


# ## ImageFolder

# In[ ]:


# Loading the three datasets.
train_data = torchvision.datasets.ImageFolder(data_dir_target + '/train/')

val_data = torchvision.datasets.ImageFolder(data_dir_target + '/val/')

test_data = torchvision.datasets.ImageFolder(data_dir_target + '/test/')

# Building the lightly datasets from the PyTorch datasets.
train_data_lightly = lightly.data.LightlyDataset.from_torch_dataset(train_data)

val_data_lightly = lightly.data.LightlyDataset.from_torch_dataset(val_data)
# val_data_lightly = lightly.data.LightlyDataset.from_torch_dataset(
#     val_data,
#     transform=val_transform
# )

test_data_lightly = lightly.data.LightlyDataset.from_torch_dataset(
    test_data,
    transform=test_transform
)


# ## Dealing with imbalanced data

# In[ ]:


if handle_imb_classes:

    # Creating a list of labels of samples.
    train_sample_labels = train_data.targets

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

print(f'\nSampler: {sampler}')
print(f'Shuffle: {shuffle}')


# ## Collate functions

# PyTorch uses a Collate Function to combine the data in your batches together.
# 
# BaseCollateFunction (base class) takes a batch of images as input and <b>transforms each image into two different augmentations</b> with the help of random transforms. The images are then concatenated such that the output batch is exactly twice the length of the input batch.

# In[ ]:


# Base class for other collate implementations.
# This allows training.
collate_fn_train = lightly.data.collate.BaseCollateFunction(train_transform)
collate_fn_val = lightly.data.collate.BaseCollateFunction(val_transform)


# These functions could be removed if I implement a custom load dataset function with a get_item that gets and tranforms two batches of images.

# ## PyTorch dataloaders

# In[ ]:


# Dataloader for training.
dataloader_train_lightly = torch.utils.data.DataLoader(
    train_data_lightly,
    batch_size=exp.batch_size,
    shuffle=shuffle,
    collate_fn=collate_fn_train,
    drop_last=True,
    num_workers=exp.num_workers,
    worker_init_fn=exp.seed_worker,
    generator=exp.g,
    sampler=sampler
)

# Dataloader for embedding (val).
dataloader_val_lightly = torch.utils.data.DataLoader(
    val_data_lightly,
    batch_size=exp.batch_size,
    shuffle=False,
    collate_fn=collate_fn_val,
    drop_last=False,
    num_workers=exp.num_workers,
    worker_init_fn=exp.seed_worker,
    generator=exp.g
)

# # Dataloader for embedding (val).
# dataloader_val = torch.utils.data.DataLoader(
#     val_data_lightly,
#     batch_size=exp.batch_size,
#     shuffle=False,
#     drop_last=False,
#     num_workers=exp.num_workers,
#     worker_init_fn=exp.seed_worker,
#     generator=exp.g
# )

# Dataloader for embedding (test).
dataloader_test = torch.utils.data.DataLoader(
    test_data_lightly,
    batch_size=exp.batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=exp.num_workers,
    worker_init_fn=exp.seed_worker,
    generator=exp.g
)


# ## Check the balance and size of the dataset

# In[ ]:


# Check samples per class in train dataset.
samples = np.unique(train_data.targets, return_counts=True)[1]
print(f'\nSamples/class train: {samples}')
print(f'Samples train: {len(train_data.targets)}')


# In[ ]:


# Check samples per class in train dataset.
samples = np.unique(val_data.targets, return_counts=True)[1]
print(f'\nSamples/class val: {samples}')
print(f'Samples val: {len(val_data.targets)}')


# In[ ]:


# Check samples per class in test dataset.
samples = np.unique(test_data.targets, return_counts=True)[1]
print(f'\nSamples/class test: {samples}')
print(f'Samples test: {len(test_data.targets)}')


# In[ ]:


print(f'\nBatches in train dataset: {len(dataloader_train_lightly)}\n')


# ## Check the distribution of samples in the dataloader (lightly dataset)

# In[ ]:


# List to save the labels.
labels_list = []

# Accessing Data and Targets in a PyTorch DataLoader.
t0 = time.time()
for i, (images, labels, names) in enumerate(dataloader_train_lightly):
    labels_list.append(labels)

# Concatenate list of lists (batches).
labels_list = torch.cat(labels_list, dim=0).numpy()
print(f'Sample distribution computation in train dataset (s): '
      f'{(time.time()-t0):.2f}')

# Count number of unique values.
data_x, data_y = np.unique(labels_list, return_counts=True)

# Old function to plot.
# utils.simple_bar_plot(data_x, data_y,
#                       x_axis_label=r'Class',
#                       y_axis_label=r'N samples (dataloader)',
#                       plt_name=f'imbalance_classes_{handle_imb_classes}',
#                       fig_size=(15, 5), save=True)

# New function to plot (suitable for execution in shell).
fig, ax = plt.subplots(1, 1, figsize=(20, 5))
simple_bar_plot(ax,
                data_x,
                'Class',
                data_y,
                'N samples (dataloader)')

plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(left=0.15)
fig_name_save = (f'sample_distribution'
                 f'-ratio={ratio}'
                 f'-balanced={handle_imb_classes}')
fig.savefig(os.path.join(output_dir_fig, fig_name_save+fig_format),
            bbox_inches='tight')
if show:
    plt.show()
else:
    plt.close()


# ## Look at some samples (lightly dataset)

# ### Only one sample from the training batch

# In[ ]:


# Accessing Data and Targets in a PyTorch DataLoader.
for i, (images, labels, names) in enumerate(dataloader_train_lightly):
    img = images[0][0]
    label = labels[0]
    print(images[0].shape)
    print(labels.shape)
    plt.title("Label: " + str(int(label)))
    plt.imshow(torch.permute(img, (1, 2, 0)))
    if show:
        plt.show()
    else:
        plt.close()
    if i == 0:
        break  # Only a few batches.


# ### Two batches (almost)

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
        if i < exp.batch_size:
            img = batch[i]
            fig.add_subplot(rows, columns, i)
            plt.imshow(torch.permute(img, (1, 2, 0)))
    if show:
        plt.show()
    else:
        plt.close()


# Train loop.
for b, ((x0, x1), _, _) in enumerate(dataloader_train_lightly):

    # Show the images within each batch.
    show_batch(x0, 0)
    show_batch(x1, 1)
    break


# Each image is augmented differently in the two batches that are loaded at the same time during training. The dataloader from lightly is capable of providing two batches in one iteration.

# # Self-supervised models

# ## Creation

# In[ ]:


from models import SimSiam, SimCLRModel, BarlowTwins


# Reference: Lightly tutorials

# ## Backbone net (w/ ResNet18)

# This is different from the tutorial: resnet without pretrained weights (not now).

# In[ ]:


# Resnet trained from scratch.
resnet = torchvision.models.resnet18(
    weights=weights
)

# Removing head from resnet. Embedding.
backbone = torch.nn.Sequential(*list(resnet.children())[:-1])

# Model creation.
if model_name == 'simsiam':
    model = SimSiam(backbone, num_ftrs, proj_hidden_dim,
                    pred_hidden_dim, out_dim)
elif model_name == 'simclr':
    hidden_dim = resnet.fc.in_features
    model = SimCLRModel(backbone, hidden_dim)
elif model_name == 'barlowtwins':
    model = BarlowTwins(backbone)


# In[ ]:


# Model's backbone structure.
summary(
    model.backbone,
    input_size=(exp.batch_size, 3, exp.input_size, exp.input_size),
    device=exp.device
)


# ## Training setup

# SimSiam uses a symmetric negative cosine similarity loss and does therefore not require any negative samples. We build a criterion and an optimizer.
# 
# 

# In[ ]:


# Scale the learning rate.
# lr = 0.05 * exp.batch_size / 256
lr = 0.2

# Use SGD with momentum and weight decay.
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=lr,
    momentum=0.9,
    weight_decay=1e-4
)


# ## Training

# ### Loop

# In[ ]:


# Device used for training.
print(f'\nUsing {exp.device} device')
model.to(exp.device)

# Saving best model's weights.
# best_model_wts = copy.deepcopy(model.state_dict())
collapse_level = 0.
lowest_train_loss = 10000
lowest_val_loss = 10000
total_train_batches = len(dataloader_train_lightly)
total_val_batches = len(dataloader_val_lightly)
print(f'\nBatches in (train, val) datasets: ({total_train_batches}, '
      f'{total_val_batches})\n')

# ======================
# TRAINING LOOP.
# Iterating over the epochs.
for e in range(exp.epochs):

    # Timer added.
    t0 = time.time()

    # Training enabled.
    model.train()

    # ======================
    # TRAINING COMPUTATION.
    # Iterating through the dataloader (lightly dataset is different).
    running_train_loss = 0.
    for b, ((x0, x1), _, _) in enumerate(dataloader_train_lightly):

        # Move images to the GPU (same batch two transformations).
        x0 = x0.to(exp.device)
        x1 = x1.to(exp.device)

        # Run the model on both transforms of the images:
        # We get projections (z0 and z1) and
        # predictions (p0 and p1) as output.
        if model_name == 'simsiam':
            z0, p0 = model(x0)
            z1, p1 = model(x1)
            loss = 0.5 * (model.criterion(z0, p1) + model.criterion(z1, p0))
        else:
            loss = model.training_step(x0, x1)

        # Averaged loss across all training examples * batch_size.
        running_train_loss += loss.item() * exp.batch_size

        # Run backpropagation.
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if model_name == 'simsiam':
            model.check_collapse(p0, loss)

        # Show partial stats.
        if b % (total_train_batches//4) == (total_train_batches//4-1):
            print(f'T[{e}, {b + 1:5d}] | '
                  f'Running train loss: '
                  f'{running_train_loss/(b*exp.batch_size):.4f}')

    # The level of collapse is large if the standard deviation of
    # the l2 normalized output is much smaller than 1 / sqrt(dim).
    if model_name == 'simsiam':
        collapse_level = max(0., 1 - math.sqrt(out_dim) * model.avg_output_std)

    # ======================
    # TRAINING LOSS.
    # Loss averaged across all training examples for the current epoch.
    epoch_train_loss = (running_train_loss
                        / len(dataloader_train_lightly.sampler))

    # ======================
    # EVALUATION COMPUTATION.
    # The evaluation process was not okey (it's been deleted).
    model.eval()
    running_val_loss = 0.
    with torch.no_grad():
        for vb, ((x0, x1), y, _) in enumerate(dataloader_val_lightly):

            # Move images to the GPU (same batch two transformations).
            x0 = x0.to(exp.device)
            x1 = x1.to(exp.device)

            # Compute loss
            if model_name == 'simsiam':
                z0, p0 = model(x0)
                z1, p1 = model(x1)
                loss = 0.5 * (model.criterion(z0, p1) + model.criterion(z1, p0))
            else:
                loss = model.training_step(x0, x1)

            # Averaged loss across all validation examples * batch_size.
            running_val_loss += loss.item() * exp.batch_size

            # Show partial stats.
            if vb % (total_val_batches//4) == (total_val_batches//4-1):
                print(f'V[{e}, {vb + 1:5d}] | '
                      f'Running val loss: '
                      f'{running_val_loss/(vb*exp.batch_size):.4f}')

    model.train()

    # ======================
    # VALIDATION LOSS.
    # Loss averaged across all training examples for the current epoch.
    epoch_val_loss = (running_val_loss
                      / len(dataloader_val_lightly.sampler))

    # ======================
    # SAVING CHECKPOINT.
    # Save model.
    save_model = ((epoch_train_loss < lowest_train_loss)
                  or (epoch_val_loss < lowest_val_loss)
                  or (e == exp.epochs - 1))
    if save_model:

        # Update new lowest losses
        if epoch_train_loss < lowest_train_loss:
            lowest_train_loss = epoch_train_loss
        elif epoch_val_loss < lowest_val_loss:
            lowest_val_loss = epoch_val_loss

        # Move the model to CPU before saving
        # it and then back to the GPU.
        model.to('cpu')
        model.save(e,
                   epoch_train_loss,
                   epoch_val_loss,
                   handle_imb_classes,
                   ratio,
                   output_dir_model,
                   collapse_level=collapse_level)
        model.to(exp.device)

    # ======================
    # EPOCH STATISTICS.
    # Show some stats per epoch completed.
    print(f'[Epoch {e:3d}] | '
          f'Train loss: {epoch_train_loss:.4f} | '
          f'Val loss: {epoch_val_loss:.4f} | '
          f'Duration: {(time.time()-t0):.2f} s | '
          f'Saved: {save_model} | '
          f'Collapse Level (SimSiam only): {collapse_level:.4f}/1.0\n')


# In[ ]:


print(model)


# Collapse level: the closer to zero the better

# A value close to 0 indicates that the representations have collapsed. A value close to 1/sqrt(dimensions), where dimensions are the number of representation dimensions, indicates that the representations are stable. 

# ### Checking the weights of the last model

# In[ ]:


# First convolutional layer weights.
print(model.backbone[0])
print(model.backbone[0].weight[63])


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
    for i, ((x, _), y, fnames) in enumerate(dataloader_val_lightly):

        # Move the images to the GPU.
        x = x.to(exp.device)
        y = y.to(exp.device)

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
df = pca_computation(embeddings, labels, exp.seed)

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
    fig.savefig(os.path.join(output_dir_fig, fig_name_save+fig_format),
                bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

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
    fig.savefig(os.path.join(output_dir_fig, fig_name_save+fig_format),
                bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

# 3-D plot with pyplot.
if (plot == '3d-plotly' or plot == 'all') and show:
    fig = px.scatter_3d(df, x='pca_x',
                        y='pca_y', z='pca_z',
                        color='labels',
                        width=1000, height=800)  # symbol='labels'

    fig.update_traces(marker=dict(size=3))

    # Move colorbar.
    # fig.update_layout(coloraxis_colorbar=dict(yanchor="top", y=1, x=0,
    #                                           ticks="outside",
    #                                           ticksuffix=""))

    fig.show()


# # t-SNE

# https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html

# In[ ]:


# t-SNE computation for 2-D.
df = tsne_computation(embeddings, labels, exp.seed, n_components=2)

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
    fig.savefig(os.path.join(output_dir_fig, fig_name_save+fig_format),
                bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

# t-SNE computation for 3-D.
df = tsne_computation(embeddings, labels, exp.seed, n_components=3)

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
    fig.savefig(os.path.join(output_dir_fig, fig_name_save+fig_format),
                bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

# 3-D plot with pyplot.
if (plot == '3d-plotly' or plot == 'all') and show:
    fig = px.scatter_3d(df, x='tsne_x',
                        y='tsne_y', z='tsne_z',
                        color='labels',
                        width=1000, height=800)  # symbol='labels'

    fig.update_traces(marker=dict(size=3))

    # Move colorbar.
    # fig.update_layout(coloraxis_colorbar=dict(yanchor="top", y=1, x=0,
    #                                           ticks="outside",
    #                                           ticksuffix=""))

    fig.show()


# ***

# ***

# # Check each model's performance/collapse on val data

# In[ ]:


def get_scatter_plot_with_thumbnails_axes(ax, title=''):
    """
    Creates a scatter plot with image overlays
    that are plotted in a particular ax position.

    """

    # Shuffle images and find out which images to show.
    shown_images_idx = []
    shown_images = np.array([[1., 1.]])
    iterator = [i for i in range(embeddings_2d.shape[0])]
    np.random.shuffle(iterator)
    for i in iterator:

        # Only show image if it is sufficiently far away from the others.
        dist = np.sum((embeddings_2d[i] - shown_images) ** 2, 1)
        if np.min(dist) < 2e-3:
            continue
        shown_images = np.r_[shown_images, [embeddings_2d[i]]]
        shown_images_idx.append(i)

    # Plot image overlays.
    for idx in shown_images_idx:
        thumbnail_size = int(rcp['figure.figsize'][0] * 2.5)  # 2.
        path = os.path.join(data_dir_test, filenames[idx])
        img = Image.open(path)
        img = functional.resize(img, thumbnail_size)
        img = np.array(img)
        img_box = osb.AnnotationBbox(
            osb.OffsetImage(img, cmap=plt.cm.gray_r),
            embeddings_2d[idx],
            pad=0.2,
        )
        ax.add_artist(img_box)

    # Set aspect ratio.
    ratio = 1. / ax.get_data_ratio()
    ax.set_aspect(ratio, adjustable='box')
    ax.title.set_text(title)


# In[ ]:


# Validation dataset.
print(f'\nValidating...')
data_dir_test = os.path.join(data_dir_target, 'val')
print(f'Dataset directory: {data_dir_test}')

# List of trained models.
print('List of model checkpoints:')
model_list = []
for root, dirs, files in os.walk(output_dir_model):
    for i, filename in enumerate(sorted(files, reverse=False)):
        model_list.append(os.path.join(root, filename))
        print(f'{i:02}: {filename}')

# Plot setup.
ncols = 5
nrows = int(math.ceil(len(model_list) / ncols))

fig, axes = plt.subplots(nrows=nrows,
                         ncols=ncols,
                         figsize=(12*ncols, 12*nrows))

# Convert the array to 1 dimension.
axes = axes.ravel()

# Main loop over the models.
for model_id, model_name_local in enumerate(model_list):

    # Load model weights.
    model.backbone.load_state_dict(torch.load(model_name_local))

    # Empty lists.
    embeddings = []
    filenames = []

    # Disable gradients for faster calculations.
    # Put the model in evaluation mode.
    model.eval()
    with torch.no_grad():
        # for i, (x, _, fnames) in enumerate(dataloader_val):
        for i, ((x, _), _, fnames) in enumerate(dataloader_val_lightly):

            # Move the images to the GPU.
            x = x.to(exp.device)

            # Embed the images with the pre-trained backbone.
            y = model.backbone(x).flatten(start_dim=1)

            # Store the embeddings and filenames in lists.
            embeddings.append(y)
            filenames = filenames + list(fnames)

    # Concatenate the embeddings and convert to numpy.
    embeddings = torch.cat(embeddings, dim=0)
    embeddings = embeddings.cpu().numpy()

    # For the scatter plot we want to transform the images to a
    # 2-D vector space using a random Gaussian projection.
    projection = random_projection.GaussianRandomProjection(
        n_components=2,
        random_state=exp.seed
    )
    embeddings_2d = projection.fit_transform(embeddings)

    # Normalize the embeddings to fit in the [0, 1] square.
    M = np.max(embeddings_2d, axis=0)
    m = np.min(embeddings_2d, axis=0)
    embeddings_2d = (embeddings_2d - m) / (M - m)

    # Get a scatter plot with thumbnail overlays.
    start_chr_epoch = model_name_local.find('-epoch') + 1
    start_chr_time = model_name_local.find('-time')
    get_scatter_plot_with_thumbnails_axes(
        axes[model_id],
        title=model_name_local[start_chr_epoch:start_chr_time]
    )

    # Show progress.
    print(f'Subplot of model-{model_id} done!',
          end='\r',
          flush=True)

# Save figure.
fig.suptitle(f'{model_name}')
fig_name_save = (f'knn-{model}')
fig.savefig(os.path.join(output_dir_fig, fig_name_save+fig_format),
            bbox_inches='tight')
if show:
    plt.show()
else:
    plt.close()


# ***

# ***

# # Embeddings for the samples of the test dataset (WARNING: custom)

# ## Setup (NOT WORKING PROPERLY)

# In[ ]:


# Test dataset.
print('\nTesting...')
data_dir_test = os.path.join(data_dir_target, 'test')
print(f'Dataset directory: {data_dir_test}')

# Load best model weights.
idx = -1

# Print model.
print(f'Target model checkpoint: {model_list[idx]}')
model.backbone.load_state_dict(torch.load(model_list[idx]))


# ## Compute embeddings

# In[ ]:


# Empty lists.
embeddings = []
filenames = []

# Disable gradients for faster calculations.
# Put the model in evaluation mode.
model.eval()
with torch.no_grad():
    for i, (x, _, fnames) in enumerate(dataloader_test):

        # Move the images to the GPU.
        x = x.to(exp.device)

        # Embed the images with the pre-trained backbone.
        y = model.backbone(x).flatten(start_dim=1)

        # Store the embeddings and filenames in lists.
        embeddings.append(y)
        filenames = filenames + list(fnames)

# Concatenate the embeddings and convert to numpy.
embeddings = torch.cat(embeddings, dim=0)
embeddings = embeddings.cpu().numpy()


# ## Projection to 2D space

# In[ ]:


# For the scatter plot we want to transform the images to a two-dimensional
# vector space using a random Gaussian projection.
projection = random_projection.GaussianRandomProjection(
    n_components=2,
    random_state=exp.seed
)
embeddings_2d = projection.fit_transform(embeddings)

# Normalize the embeddings to fit in the [0, 1] square.
M = np.max(embeddings_2d, axis=0)
m = np.min(embeddings_2d, axis=0)
embeddings_2d = (embeddings_2d - m) / (M - m)


# ## Scatter plots

# In[ ]:


# Initialize empty figure and add subplot.
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(1, 1, 1)

# Get a scatter plot with thumbnail overlays.
get_scatter_plot_with_thumbnails_axes(
    ax,
    title='Scatter plot with samples'
)

# Save figure.
fig_name_save = (f'scatter_samples-{model}')
fig.savefig(os.path.join(output_dir_fig, fig_name_save+fig_format),
            bbox_inches='tight')
if show:
    plt.show()
else:
    plt.close()


# ## Nearest Neighbors

# ### Pick up one random sample per class

# In[ ]:


# List of subdirectories (classes).
directory_list = []
for root, dirs, files in os.walk(data_dir_test):
    for dirname in sorted(dirs):
        directory_list.append(os.path.join(root, dirname))
        # print(dirname)


# In[ ]:


# List of files (samples).
example_images = []
for classes in directory_list:

    # Random samples.
    random_file = np.random.choice(os.listdir(classes))
    path_to_random_file = classes + '/' + random_file

    # Only class and filename.
    start_chr = path_to_random_file.index('test/') + 5

    # Append filename.
    example_images.append(path_to_random_file[start_chr:])
    # print(example_images)


# ### Look for similar images

# In[ ]:


def get_image_as_np_array(filename: str):
    """
    Loads the image with filename and returns it as a numpy array.

    """
    img = Image.open(filename)
    return np.asarray(img)


def get_image_as_np_array_with_frame(filename: str, w: int = 5):
    """
    Returns an image as a numpy array with a black frame of width w.

    """
    img = get_image_as_np_array(filename)
    ny, nx, _ = img.shape

    # Create an empty image with padding for the frame.
    framed_img = np.zeros((w + ny + w, w + nx + w, 3))
    framed_img = framed_img.astype(np.uint8)

    # Put the original image in the middle of the new one.
    framed_img[w:-w, w:-w] = img
    return framed_img


def plot_nearest_neighbors_nxn(example_image: str, i: int):
    """
    Plots the example image and its eight nearest neighbors.

    """
    n_subplots = 6

    # Initialize empty figure.
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(f"Nearest Neighbor Plot Class {i}")

    # Get indexes.
    example_idx = filenames.index(example_image)

    # Get distances to the cluster center.
    distances = embeddings - embeddings[example_idx]
    distances = np.power(distances, 2).sum(-1).squeeze()

    # Sort indices by distance to the center.
    nearest_neighbors = np.argsort(distances)[:n_subplots]

    # Show images.
    for plot_offset, plot_idx in enumerate(nearest_neighbors):
        ax = fig.add_subplot(3, 3, plot_offset + 1)

        # Get the corresponding filename.
        fname = os.path.join(data_dir_test, filenames[plot_idx])
        if plot_offset == 0:
            ax.set_title(f"Example Image")
            plt.imshow(get_image_as_np_array_with_frame(fname))
        else:
            plt.imshow(get_image_as_np_array(fname))

        # Let's disable the axis.
        plt.axis("off")

    # Save figure.
    fig_name_save = (f'knn_per_class-c={i:02}-{model}')
    fig.savefig(os.path.join(output_dir_fig, fig_name_save+fig_format),
                bbox_inches='tight')
    if show:
        pass  # plt.show()
    else:
        plt.close()


# In[ ]:


# Show example images for each cluster.
for i, example_image in enumerate(example_images):
    plot_nearest_neighbors_nxn(example_image, i)


# In[ ]:




