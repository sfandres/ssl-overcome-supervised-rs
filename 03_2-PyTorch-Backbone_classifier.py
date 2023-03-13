#!/usr/bin/env python
# coding: utf-8

# **USING RESNET18 WITH AND WITHOUT PRETRAINED WEIGHTS**

# **LOADING THE SSL MODEL AND TRAINING A CLASSIFIER ON TOP OF IT**

# Reference 1: https://pytorch.org/tutorials/beginner/introyt/trainingyt.html

# Reference 2: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

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


# if is_notebook():
#     %load_ext pycodestyle_magic


# In[ ]:


# if is_notebook():
#     %pycodestyle_on


# In[ ]:


# Load the TensorBoard notebook extension
if is_notebook():
    get_ipython().run_line_magic('load_ext', 'tensorboard')


# ***

# ***

# # Imports

# ## Libraries and modules

# In[ ]:


# Custom modules.
from utils.computation import Experiment, pca_computation, tsne_computation
from utils.dataset import (
    AndaluciaDataset,
    get_mean_std_dataloader,
    show_one_batch,
    inv_norm_tensor
)

# OS module.
import os

# PyTorch.
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchinfo import summary

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import csv

# Data management.
import numpy as np

# import lightly

# Training checks.
import copy
import time
import math

# import random

# For plotting.
import matplotlib.pyplot as plt

# Showing images in the notebook.
from IPython.display import Image
from IPython.core.display import HTML

# PyTorch TensorBoard support
# from torch.utils.tensorboard import SummaryWriter


# ## Parser

# In[ ]:


# Workaround to make bool options work as expected.
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# In[ ]:


# Parser (get arguments).
if __name__ == '__main__':
    import argparse
    import sys
    parser = argparse.ArgumentParser(
        description=("Script for evaluating the self-supervised learning"
                     " models and compare them to standard approaches.")
    )
    parser.add_argument(
        'model',
        type=str,
        choices=['scratch', 'imagenet', 'ssl'],
        help=("Model for finetuning. "
              "Use 'scratch', 'imagenet' or 'ssl'.")
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='Sentinel2AndaluciaLULC',
        choices=['Sentinel2GlobalLULC', 'Sentinel2AndaluciaLULC'],
        help='Dataset name for evaluation.'
    )
    parser.add_argument(
        '--ratio',
        type=str,
        default='(0.700,0.0900,0.2100)',
        help='Dataset ratio for evaluation.'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=15,
        help='Number of epochs for training.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Number of images in a batch during training.'
    )
    parser.add_argument(
        '--show_fig',
        type=str2bool,
        default=False,
        help='Whether the images should appear.'
    )
    parser.add_argument(
        '--cluster',
        type=str2bool,
        default=False,
        help=('Whether the script runs on a cluster '
              '(large memory space available).')
    )

if is_notebook():
    args = parser.parse_args(
        args=['scratch',
              '--dataset',
              'Sentinel2AndaluciaLULC',
              '--epochs',
              '100',
              '--batch_size',
              '64',
              '--show_fig',
              'True'])
else:
    args = parser.parse_args(sys.argv[1:])


# ## Settings

# In[ ]:


# Target model.
model_name = args.model
print(f'\nTarget model for finetuning: {model_name}')

# Target dataset name.
dataset_name = args.dataset
print(f'Target dataset name: {dataset_name}')

# # Target dataset ratio.
# dataset_ratio = args.ratio
# print(f'Target dataset ratio: {dataset_ratio}')

# Setting number of epochs.
epochs = args.epochs
print(f'Number of epochs: {epochs}')

# Setting batch size.
batch_size = args.batch_size
print(f'Batch size: {batch_size}')

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
exp = Experiment(
    batch_size=batch_size,
    epochs=epochs,
    input_size=224,
)
print(f'\nDevice: {exp.device}\n')

# Get current directory.
cwd = os.getcwd()
print(f'Working directory: {cwd}')

# Input directory where the datasets are stored.
input_dir_datasets = os.path.join(cwd, 'datasets')
print(f'\nInput directory for datasets: {input_dir_datasets}')

# Output directory.
output_dir = os.path.join(cwd, 'output')

# Output directory to save the model checkpoint.
output_dir_models = os.path.join(os.path.join(output_dir, 'pytorch_models'),
                                 'finetuning')
print(f'Output directory for model checkpoints: {output_dir_models}')

# Folder to save the figures.
output_dir_figs = os.path.join(os.path.join(output_dir, 'figures'),
                               'finetuning')
fig_format = '.png'  # .pdf
print(f'Output directory for figures ({fig_format} format): {output_dir_figs}')


# ## Reproducibility

# In[ ]:


exp.reproducibility()


# ***

# ***

# # Custom dataset

# ## Compute normalization values (just once)

# In[ ]:


# splits = ['train', 'validation', 'test']

# # Load Andalucia dataset.
# andalucia_dataset = {x: AndaluciaDataset(
#     root_dir=os.path.join(input_dir_datasets, dataset_name),
#     level='Level_N2',
#     split=x,
#     transform=transforms.ToTensor(),
#     target_transform=None
# ) for x in splits}

# # Creating the dataloaders.
# dataloaders = {x: DataLoader(
#     andalucia_dataset[x],
#     batch_size=128,
#     worker_init_fn=exp.seed_worker,
#     generator=exp.g
# ) for x in splits}

# # Loop over the train, val, and test datasets.
# for x in splits:

#     # Computation.
#     print(f'- {x}:')
#     print(f'Samples to be processed: '
#           f'{len(dataloaders[x].dataset)}')
#     mean, std = get_mean_std_dataloader(dataloaders[x])

#     # Show mean and std.
#     print(f'mean: {mean}')
#     print(f'std: {std}\n')


# ## Load dataset w/ mean and std values (add transforms)

# - train:
# Samples to be processed: 15038
# tensor([0.3036, 0.3045, 0.3224])
# tensor([0.1351, 0.0921, 0.0712])
# 
# - validation:
# Samples to be processed: 2153
# tensor([0.3042, 0.3047, 0.3224])
# tensor([0.1338, 0.0910, 0.0701])
# 
# - test:
# Samples to be processed: 4298
# tensor([0.3015, 0.3031, 0.3213])
# tensor([0.1314, 0.0894, 0.0689])

# In[ ]:


splits = ['train', 'validation', 'test']

# Set mean and std values.
mean_std_dict = {
    'train': (torch.tensor([0.3036, 0.3045, 0.3224]),
              torch.tensor([0.1351, 0.0921, 0.0712])),
    'validation': (torch.tensor([0.3042, 0.3047, 0.3224]),
                   torch.tensor([0.1338, 0.0910, 0.0701])),
    'test': (torch.tensor([0.3015, 0.3031, 0.3213]),
             torch.tensor([0.1314, 0.0894, 0.0689]))
}

# Normalization transform (val and test).
transform_normal = {x: transforms.Compose([
    transforms.Resize((exp.input_size, exp.input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_std_dict[x][0],
                         std=mean_std_dict[x][1])
]) for x in splits[1:]}

# Normalization transform (train).
transform_normal['train'] = transforms.Compose([
    transforms.Resize((exp.input_size, exp.input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_std_dict['train'][0],
                         std=mean_std_dict['train'][1])
])
print(f'\n{transform_normal}')

# Load the Andalucia dataset with normalization.
andalucia_dataset_norm = {x: AndaluciaDataset(
    root_dir=os.path.join(input_dir_datasets, dataset_name),
    level='Level_N2',
    split=x,
    transform=transform_normal[x],
    target_transform=None,
    verbose=False
) for x in splits}

# Define dataloaders.
dataloader = {x: torch.utils.data.DataLoader(
    andalucia_dataset_norm[x],
    batch_size=exp.batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=exp.num_workers,
    worker_init_fn=exp.seed_worker,
    generator=exp.g
) for x in splits}

# Get classes and number.
class_names = andalucia_dataset_norm['train'].classes
print(class_names)

# Get dictionary of classes.
idx_to_class = andalucia_dataset_norm['train'].idx_to_class
# print(idx_to_class)


# ## Look at some training samples

# In[ ]:


# Create figure with subplots.
num_rows = int(dataloader['train'].batch_size ** 0.5)
num_cols = (int(dataloader['train'].batch_size / num_rows)
            + (dataloader['train'].batch_size % num_rows > 0))
fig, axes = plt.subplots(nrows=num_rows,
                         ncols=num_cols,
                         figsize=(4*num_cols, 4*num_rows))

# Take only one batch (inverse transform applied).
inv_norm=True
show_one_batch(axes, num_cols, dataloader['train'],
               andalucia_dataset_norm['train'].idx_to_class,
               batch_id=0, inv_norm=inv_norm, mean=mean_std_dict['train'][0],
               std=mean_std_dict['train'][1])

# Adjust and show image.
plt.tight_layout()
plt.show() if show else plt.close()


# In[ ]:


# Display image and label (w/ or w/o normalization).
train_features, train_labels = next(iter(dataloader['train']))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
img = inv_norm_tensor(
    img,
    mean=mean_std_dict['train'][0],
    std=mean_std_dict['train'][1]
)
label = train_labels[0]
# print(label)
class_name = idx_to_class[int(torch.argmax(label))]
plt.title(f'{class_name}')
plt.imshow(torch.permute(img, (1, 2, 0)), cmap="gray")
plt.tight_layout()
plt.show() if show else plt.close()


# ## Check balance and size

# In[ ]:


# Print some stats from the train dataset.
print(f"\n#Samples in train (from len(dataset)): {len(andalucia_dataset_norm['train'])}")
print(f"#Samples in train (from dataloader.sampler): {len(dataloader['train'].sampler)}")
print(f"#Samples in train (from dataloader.dataset): {len(dataloader['train'].dataset)}")
print(f"#Batches in train (from dataloader): {len(dataloader['train'])}")

# Print some stats from the val dataset.
print(f"\n#Samples in val (from dataset):    {len(andalucia_dataset_norm['validation'])}")
print(f"#Samples in val (from dataloader): {len(dataloader['validation'].dataset)}")
print(f"#Batches in val (from dataloader): {len(dataloader['validation'])}")

# Print some stats from the test dataset.
print(f"\n#Samples in test (from dataset):    {len(andalucia_dataset_norm['test'])}")
print(f"#Samples in test (from dataloader)  {len(dataloader['test'].dataset)}")
print(f"#Batches in test (from dataloader): {len(dataloader['test'])}")


# ***

# ***

# # Training

# https://pytorch.org/tutorials/beginner/introyt/trainingyt.html

# ## The training loop

# In[ ]:


def train_one_epoch(epoch_index, tb_writer, batch_span_train):

    running_loss = 0.
    last_loss = 0.

    # ======================
    # TRAINING COMPUTATION.
    # Iterating through the dataloader
    # We can track the batch index and do some intra-epoch reporting.
    for i, data in enumerate(dataloader['train']):

        # Every data instance is an input + label pair.
        inputs = data[0].to(exp.device)
        labels = data[1].to(exp.device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch.
        outputs = model(inputs)

        # Compute the loss and its gradients.
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights.
        optimizer.step()

        # Gather data and report.
        running_loss += loss.item()
        if i % batch_span_train == batch_span_train - 1:
            last_loss = running_loss / batch_span_train # loss per batch
            print(f'  batch {i+1:03d} loss: {last_loss:.4f}')
            tb_x = epoch_index * len(dataloader['train']) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


# ## Models and parameters

# In[ ]:


# Model: resnet with random weights.
if model_name == 'scratch':
    print('\nModel without pretrained weights')
    model = torchvision.models.resnet18(weights=None)
elif model_name == 'imagenet':
    print('\nModel with pretrained weights on imagenet-1k')
    model = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.DEFAULT
    )


# In[ ]:


# Get the number of input features to the layer.
# Adjust the final layer to the current number of classes.
print(f'Old final fully-connected layer: {model.fc}')
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(class_names))
print(f'New final fully-connected layer: {model.fc}\n')

# Parameters of newly constructed modules
# have requires_grad=True by default.
# Freezing all the network except the final layer.
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

# Show model structure.
summary(
    model,
    input_size=(exp.batch_size, 3, exp.input_size, exp.input_size),
    device=exp.device
)


# In[ ]:


# Set loss.
# loss_fn = torch.nn.CrossEntropyLoss()
loss_fn = torch.nn.BCEWithLogitsLoss()
print(f'\nLoss: {loss_fn}')

# Optimizers specified in the torch.optim package
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
print(f'Optimizer:\n{optimizer}')


# ## Per-epoch activity

# In[ ]:


# ======================
# SET UP WRITERS AND VARIABLES.
# Initializing in a separate cell so we can easily add more epochs to the same run.
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
run_path = os.path.join('runs', f'lulc_finetuning_{model_name}_{timestamp}')
print(f'\nRun path: {run_path}')
writer = SummaryWriter(run_path)

# Open the file in the write mode.
header = ['epoch', 'avg_loss', 'avg_vloss']
csv_file = os.path.join(run_path, 'training_info.csv')
with open(csv_file, 'w', newline='') as file:
    csv_writer = csv.writer(file)  # Write the header.
    csv_writer.writerow(header)

# Initial variables.
epoch_number = 0
batch_span_train = len(dataloader['train']) // 5
batch_span_val = len(dataloader['validation']) // 5
best_vloss = 1_000_000.

# Device used for training.
print(f'Using {exp.device} device')
model.to(exp.device)

# ======================
# TRAINING LOOP.
# Iterating over the epochs.
for epoch in range(epochs):
    print(f'\nEPOCH {epoch_number + 1}:')

    # ======================
    # TRAINING LOSS.
    # Make sure gradient tracking is on, and do a pass over the data.
    print('Running training...')
    model.train()
    avg_loss = train_one_epoch(epoch_number, writer, batch_span_train)
    print('Training completed!')

    # ======================
    # EVALUATION COMPUTATION.
    # We don't need gradients on to do reporting.
    print('Running evaluation...')
    model.eval()
    running_vloss = 0.0

    with torch.no_grad():
        for i, vdata in enumerate(dataloader['validation']):
            vinputs = vdata[0].to(exp.device)
            vlabels = vdata[1].to(exp.device)            
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss.item()
            if i % batch_span_val == batch_span_val - 1:
                print(f'  batch {i+1:03d} vloss: {vloss:.4f}')

    print('Evaluation completed!')

    # ======================
    # VALIDATION LOSS.
    avg_vloss = running_vloss / (i + 1)
    print(f'LOSS train {avg_loss} val {avg_vloss}')

    # ======================
    # SAVING DATA.
    # Log the running loss averaged per batch
    # for both training and validation.
    writer.add_scalars('Training vs. Validation Loss',
                       {'Training' : avg_loss, 'Validation' : avg_vloss},
                       epoch_number + 1)
    writer.flush()

    # Open the file in the append mode.
    data = [epoch, avg_loss, avg_vloss]
    with open(csv_file, 'a', newline='') as file:
        csv_writer = csv.writer(file)  # Write the data.
        csv_writer.writerow(data)

    # ======================
    # SAVING CHECKPOINT.
    # Track best performance, and save the model's state.
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = os.path.join(run_path, f'model_{model_name}_{timestamp}_{epoch_number:03d}')
        torch.save(model.state_dict(), model_path)

    epoch_number += 1


# In[ ]:


if is_notebook():
    get_ipython().run_line_magic('tensorboard', '--logdir=runs')


# ***

# ***

# UP TO HERE

# ***

# ***

# In[ ]:


# # Training function.
# def train_model(model, criterion, optimizer, device,
#                 epochs=10, save_best_model=False):
#     """
#     Main training function.

#     """

#     print(f"Using {exp.device} device")

#     # Avoiding "CUDA out of memory" in PyTorch.
#     torch.cuda.empty_cache()

#     # Loss history.
#     loss_values = {}
#     loss_values['train'] = []
#     loss_values['val'] = []
#     total_time = 0

#     # Saving best model's weights.
#     best_model_val_wts = copy.deepcopy(model.state_dict())
#     lowest_val_loss = 10000

#     # Model to GPU if available.
#     model.to(exp.device)

#     # Iterating over the epochs.
#     for epoch in range(epochs):

#         # Initialize training loss.
#         running_train_loss = 0.0

#         # Start timer.
#         since = time.time()

#         # Enable training.
#         model.train()

#         for i, data in enumerate(dataloader_train):

#             # Get the inputs; data is a list of [inputs, labels].
#             inputs, labels = data[0].to(exp.device), data[1].to(exp.device)

#             # Zero the parameter gradients.
#             optimizer.zero_grad()

#             # Forward: make predictions.
#             outputs = model(inputs)

#             # Compute the loss and its gradients.
#             loss = criterion(outputs, labels)
#             loss.backward()

#             # Averaged loss across all training examples * batch_size.
#             running_train_loss += loss.item() * inputs.size(0)

#             if i % 200 == 199:
#                 print(f'T[{epoch + 1}, {i + 1:5d}] | '
#                       f'Running loss: '
#                       f'{running_train_loss/(i*inputs.size(0)):.4f}')

#             # Adjust learning weights.
#             optimizer.step()

#         # Loss averaged across all training examples for the current epoch.
#         epoch_train_loss = running_train_loss / len(dataloader_train.sampler)

#         # Change model to evaluation mode.
#         model.eval()

#         # Initialize validating loss.
#         running_val_loss = 0.0
#         with torch.no_grad():
#             for j, vdata in enumerate(dataloader_val):

#                 # Get the inputs; data is a list of [inputs, labels].
#                 vinputs, vlabels = vdata[0].to(exp.device), vdata[1].to(exp.device)

#                 # Forward: make predictions.
#                 voutputs = model(vinputs)

#                 # Compute the loss (w/o gradients).
#                 vloss = criterion(voutputs, vlabels)

#                 # Averaged loss across all validating examples * batch_size.
#                 running_val_loss += vloss.item() * vinputs.size(0)

#                 if j % 50 == 49:
#                     print(f'V[{epoch + 1}, {j + 1:5d}] | '
#                           f'Running loss: '
#                           f'{running_val_loss/(j*inputs.size(0)):.4f}')

#         # Loss averaged across all validating examples for the current epoch.
#         epoch_val_loss = running_val_loss / len(dataloader_val.sampler)

#         # Append loss values.
#         loss_values['train'].append(epoch_train_loss)
#         loss_values['val'].append(epoch_val_loss)

#         # Deep copy the weights of the model.
#         save_weights = epoch_val_loss < lowest_val_loss
#         if save_weights:
#             lowest_val_loss = epoch_val_loss
#             best_model_train_loss = epoch_train_loss
#             best_model_val_wts = copy.deepcopy(model.state_dict())

#         # End timer.
#         time_elapsed = time.time() - since
#         total_time += time_elapsed

#         # Show stats.
#         print(f'Epoch: {epoch} | '
#               f'Train loss: {epoch_train_loss:.4f} | '
#               f'Val loss: {epoch_val_loss:.4f} | '
#               f'Elapsed: {time_elapsed // 60:.0f}m '
#               f'{time_elapsed % 60:.0f}s | '
#               f'Save weights: {save_weights}')

#     print(f'\nTraining completed in {total_time // 60:.0f}m '
#           f'{total_time % 60:.0f}s')

#     # Load best model weights.
#     model.load_state_dict(best_model_val_wts)

#     if save_best_model:

#         # Move to CPU before saving it.
#         model.to('cpu')

#         # Filename with stats.
#         save_path = f'pytorch_models/resnet18' \
#                     f'-losses={best_model_train_loss:.2f}' \
#                     f'_{lowest_val_loss:.2f}' \
#                     f'-time={datetime.now():%Y_%m_%d-%H_%M_%S}'

#         # Save this pretrained model (recommended approach).
#         torch.save(model.state_dict(), save_path)

#         print('Model successfuly saved')

#         # Move back to the GPU.
#         model.to(exp.device)

#     return model, loss_values


# In[ ]:


# def plot_losses(loss_history, title='', save_fig=False):
#     """
#     Function for plotting the training and validation losses

#     """

#     fig = plt.figure(figsize=(10, 5))
#     plt.plot(loss_history['train'], label='Train')
#     plt.plot(loss_history['val'], label='Validation')
#     plt.xlabel('Epoch', labelpad=15)
#     plt.ylabel('Loss', labelpad=15)
#     plt.title(title)
#     plt.gcf().subplots_adjust(bottom=0.15)
#     plt.gcf().subplots_adjust(left=0.15)
#     plt.legend(loc='best')
#     plt.tight_layout()
#     plt.show()

#     if save_fig:
#         fig.savefig('plt_loss_values.pdf', bbox_inches='tight')


# In[ ]:


# def evaluation_on_test(model, device):
#     """
#     Function to evaluate the performance
#     of the model on the test dataset.

#     """

#     # Avoiding "CUDA out of memory" in PyTorch.
#     torch.cuda.empty_cache()

#     correct = 0
#     total = 0

#     # Since we're not training, we don't need to calculate
#     # the gradients for our outputs with torch.no_grad():
#     model.eval()
#     with torch.no_grad():
#         for i, data in enumerate(dataloader_test):

#             # Dataset.
#             inputs, labels = data[0].to(device), data[1].to(device)

#             # Calculate outputs by running images through the network.
#             outputs = model(inputs)

#             # The class with the highest energy is what we
#             # choose as prediction.
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#             # Progress bar.
#             if i % 50 == 49:
#                 print(f'Progress: {100 * i // len(dataloader_test)}%',
#                       end='\r',
#                       flush=True)

#     print(f'Accuracy of the network on the {total} '
#           f'test images: {100 * correct // total}%')


# ***

# ***

# # ResNet18 from scrath

# ## Definition and hyperparameters

# In[ ]:


# # Model: resnet with random weights.
# model = torchvision.models.resnet18(weights=None)


# ## Adjust final layer

# Type: linear not softmax for the moment.

# In[ ]:


# # Check old final layer.
# print(model.fc)

# # Get the number of input features to the layer.
# num_ftrs = model.fc.in_features
# print(num_ftrs)

# # Adjust the final layer to the current number of classes.
# model.fc = torch.nn.Linear(num_ftrs, len(class_names))

# # Check new final layer.
# print(model.fc)

# # Freezing all the network except the final layer.
# for param in model.parameters():
#     param.requires_grad = False
# for param in model.fc.parameters():
#     param.requires_grad = True

# # Model structure.
# summary(
#     model,
#     input_size=(exp.batch_size, 3, exp.input_size, exp.input_size),
#     device=exp.device
# )


# ## Loss fcn and optimizer

# In[ ]:


# # Loss function: cross-entropy loss.
# loss_fn = torch.nn.CrossEntropyLoss()

# # Optimizers: specified in the torch.optim package
# optimizer = torch.optim.Adam(model.parameters(),
#                              lr=0.01)


# ## Training

# In[ ]:


# model, loss_history = train_model(
#     model,
#     loss_fn,
#     optimizer,
#     exp.device,
#     epochs=exp.epochs,
#     save_best_model=True
# )

# plot_losses(loss_history, 'Model w/o pretrained weights')


# ## Check performance on test dataset

# In[ ]:


# evaluation_on_test(model, exp.device)


# ## F1-score

# In[ ]:


# # Confusion matrix
# conf_mat, class_accuracy = utils.create_confusion_matrix(
#     model,
#     dataloader_test,
#     exp.device,
#     class_names
# )

# # Bar plot for accuracy values.
# utils.simple_bar_plot(range(len(class_names)),
#                       class_accuracy,
#                       'Classes',
#                       'Accuracy (%)',
#                       'class_accuracy',
#                       fig_size=(15, 5),
#                       save=False)


# ***

# ***

# # ResNet18 with pretrained weights

# ## Definition and hyperparameters

# In[ ]:


# # Model: resnet with pretrained weights.
# del model
# model = torchvision.models.resnet18(
#     weights=torchvision.models.ResNet18_Weights.DEFAULT
# )


# ## Adjust final layer

# In[ ]:


# # Check old final layer.
# print(model.fc)

# # Get the number of input features to the layer.
# num_ftrs = model.fc.in_features
# print(num_ftrs)

# # Adjust the final layer to the current number of classes.
# model.fc = torch.nn.Linear(num_ftrs, len(class_names))

# # Parameters of newly constructed modules
# # have requires_grad=True by default.
# # Check new final layer.
# print(model.fc)

# # Freezing all the network except the final layer.
# for param in model.parameters():
#     param.requires_grad = False
# for param in model.fc.parameters():
#     param.requires_grad = True

# # Model structure.
# summary(
#     model,
#     input_size=(exp.batch_size, 3, exp.input_size, exp.input_size),
#     device=exp.device
# )


# ## Loss fcn and optimizer

# In[ ]:


# # Loss function: cross-entropy loss.
# loss_fn = torch.nn.CrossEntropyLoss()

# # Optimizers: specified in the torch.optim package
# optimizer = torch.optim.Adam(model.parameters(),
#                              lr=0.01)


# ## Training

# In[ ]:


# model, loss_history = train_model(
#     model,
#     loss_fn,
#     optimizer,
#     exp.device,
#     epochs=exp.epochs,
#     save_best_model=True
# )

# plot_losses(loss_history, 'Model w/ pretrained weights')


# ## Check performance on test dataset

# In[ ]:


# evaluation_on_test(model, exp.device)


# ## F1-score

# In[ ]:


# # Confusion matrix
# conf_mat, class_accuracy = utils.create_confusion_matrix(
#     model,
#     dataloader_test,
#     exp.device,
#     class_names
# )

# # Bar plot for accuracy values.
# utils.simple_bar_plot(range(len(class_names)),
#                       class_accuracy,
#                       'Classes',
#                       'Accuracy (%)',
#                       'class_accuracy',
#                       fig_size=(15, 5),
#                       save=False)


# ***

# ***

# # SSL model

# ## Loading

# In[ ]:


# # State model class.
# resnet18 = torchvision.models.resnet18(weights=None)

# # Only backbone (w/o final fc layer).
# pt_backbone = torch.nn.Sequential(*list(resnet18.children())[:-1])


# In[ ]:


# # List of trained models.
# model_list = []
# for root, dirs, files in os.walk('pytorch_models/history_log/'):
#     for i, filename in enumerate(sorted(files, reverse=True)):
#         model_list.append(root + filename)
#         print(f'{i:02}: {filename}')


# In[ ]:


# # Loading model.
# idx = 0
# print(model_list[idx])
# pt_backbone.load_state_dict(torch.load(model_list[idx]))


# In[ ]:


# # Check if the model is loaded on GPU.
# next(pt_backbone.parameters()).is_cuda


# ## Checking the weights

# In[ ]:


# # First convolutional layer weights.
# # print(backbone)
# print(pt_backbone[0])
# print(pt_backbone[0].weight[63])


# ## Adding a final linear layer

# In[ ]:


# # Adding a linear layer on top of the model (linear classifier).
# model_ssl = torch.nn.Sequential(
#     pt_backbone,
#     torch.nn.Flatten(),
#     torch.nn.Linear(in_features=512, out_features=len(class_names), bias=True),
#     # torch.nn.Softmax(dim=1)
# )

# # Freezing all the network except the final layer.
# for param in model_ssl.parameters():
#     param.requires_grad = False
# # for param in model_ssl[0][7].parameters():
# #     param.requires_grad = True
# for param in model_ssl[2].parameters():
#     param.requires_grad = True

# # Model structure.
# summary(
#     model_ssl,
#     input_size=(exp.batch_size, 3, exp.input_size, exp.input_size),
#     device=exp.device
# )


# ## Loss fcn and optimizer

# In[ ]:


# # Loss function: cross-entropy loss.
# loss_fn = torch.nn.CrossEntropyLoss()

# # Optimizers: specified in the torch.optim package
# optimizer = torch.optim.Adam(model_ssl.parameters(),
#                              lr=0.01)


# ## Training

# In[ ]:


# model_ssl, loss_history = train_model(
#     model_ssl,
#     loss_fn,
#     optimizer,
#     exp.device,
#     epochs=exp.epochs,
#     save_best_model=True
# )

# plot_losses(loss_history, 'Model w/ ssl pretrained weights')


# ## Checking the weights after training

# They should have remained the same (frozen) except for the final layer.

# In[ ]:


# # First convolutional layer weights.
# # print(backbone)
# print(pt_backbone[0])
# print(pt_backbone[0].weight[63])


# ## Check performance on test dataset

# In[ ]:


# evaluation_on_test(model_ssl, exp.device)


# ## F1-score

# In[ ]:


# # Confusion matrix
# conf_mat, class_accuracy = utils.create_confusion_matrix(
#     model_ssl,
#     dataloader_test,
#     exp.device,
#     class_names
# )

# # Bar plot for accuracy values.
# utils.simple_bar_plot(range(len(class_names)),
#                       class_accuracy,
#                       'Classes',
#                       'Accuracy (%)',
#                       'class_accuracy',
#                       fig_size=(15, 5),
#                       save=False)


# ***

# ***
