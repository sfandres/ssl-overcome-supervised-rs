#!/usr/bin/env python
# coding: utf-8

# **FINETUNING USING THE ANDALUCIA DATASET: RANDOM, IMAGENET, AND SSL-PRETRAINED WEIGHTS**

# Reference 1: https://pytorch.org/tutorials/beginner/introyt/trainingyt.html <br>
# Reference 2: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

# ***

# ***

# # Initial configuration

# ## Libraries and modules

# In[ ]:


# Custom modules.
from utils.other import is_notebook, build_paths
from utils.reproducibility import set_seed, seed_worker
from utils.dataset import (
    AndaluciaDataset,
    get_mean_std_dataloader,
    load_mean_std_values
)
from utils.graphs import simple_bar_plot
from utils.dataset import (
    inv_norm_tensor,
    show_one_batch
)

# Arguments and paths.
import os
import sys
import argparse

# PyTorch.
import torch
import torchvision
from torchvision import transforms
from torchinfo import summary

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
import csv

# Data management.
import numpy as np

# Performance metrics.
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Training checks.
from datetime import datetime
import time

# For plotting.
import matplotlib.pyplot as plt

# Showing images in the notebook.
# from IPython.display import Image
# from IPython.core.display import HTML
# import matplotlib.font_manager


# In[ ]:


# Load notebook extensions.
if is_notebook():
    get_ipython().run_line_magic('load_ext', 'tensorboard')
    # %load_ext pycodestyle_magic
    # %pycodestyle_on


# In[ ]:


SEED = 42
AVAIL_SSL_MODELS = ['SimSiam', 'SimCLR', 'SimCLRv2', 'BarlowTwins']
MODEL_CHOICES = ['Random', 'Imagenet'] + AVAIL_SSL_MODELS


# ## Enable reproducibility

# Reference: https://pytorch.org/docs/stable/notes/randomness.html

# In[ ]:


print(f"\n{'torch initial seed:'.ljust(20)} {torch.initial_seed()}")
g = set_seed(SEED)
print(f"{'torch current seed:'.ljust(20)} {torch.initial_seed()}")


# ## Check torch CUDA

# In[ ]:


print(f"\n{'torch.cuda.is_available():'.ljust(32)} {torch.cuda.is_available()}")
print(f"{'torch.cuda.device_count():'.ljust(32)} {torch.cuda.device_count()}")
print(f"{'torch.cuda.current_device():'.ljust(32)} {torch.cuda.current_device()}")
print(f"{'torch.cuda.device(0):'.ljust(32)} {torch.cuda.device(0)}")
print(f"{'torch.cuda.get_device_name(0):'.ljust(32)} {torch.cuda.get_device_name(0)}")
print(f"{'torch.backends.cudnn.benchmark:'.ljust(32)} {torch.backends.cudnn.benchmark}")


# ## Command line arguments

# In[ ]:


# Parser (get arguments).
parser = argparse.ArgumentParser(
    description=("Script for evaluating the self-supervised learning"
                 " models and compare them to standard approaches.")
)

parser.add_argument('model_name', type=str, choices=MODEL_CHOICES,
                    help="target model.")

parser.add_argument('task', type=str, choices=['multiclass', 'multilabel'],
                    help="downstream task.")

parser.add_argument('--dataset_name', '-dn', type=str,
                    default='Sentinel2AndaluciaLULC',
                    help='dataset name for evaluation (default=Sentinel2AndaluciaLULC).')

parser.add_argument('--dataset_level', '-dl', type=str, default='Level_N2',
                    choices=['Level_N1', 'Level_N2'],
                    help="dataset level (default=Level_N2).")

parser.add_argument('--dataset_train_pc', '-dtp', type=float, default=1.,
                    help='dataset ratio for train subset (default=1.).')

parser.add_argument('--epochs', '-e', type=int, default=25,
                    help='number of epochs for training (default=25).')

parser.add_argument('--batch_size', '-bs', type=int, default=64,
                    help='number of images in a batch during training (default=64).')

parser.add_argument('--show', '-s', action='store_true',
                    help='the images should appear.');

parser.add_argument('--cluster', '-c', action='store_true',
                    help='the script runs on a cluster (large mem. space).');

parser.add_argument('--torch_compile', '-tc', action='store_true',
                    help='PyTorch 2.0 compile enabled.');


# ## Simulate and get input arguments

# In[ ]:


# Input arguments.
if is_notebook():
    args = parser.parse_args(
        args=[
            'SimSiam',
            'multiclass',
            '--show',
            '--dataset_name=Sentinel2AndaluciaLULC',
            '--dataset_train_pc=.05',
            '--batch_size=64',
            '--epochs=1'
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


# ## Build paths

# In[ ]:


# Get current directory.
cwd = os.getcwd()

# Build paths.
paths = build_paths(cwd, os.path.join('finetuning', model_name))
paths['runs'] = os.path.join(paths['runs'], f'finetuning_{dataset_train_pc:.3f}pc_train')

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

# Number of digits to the right of the decimal
decimal_places = 2


# ***

# ***

# # Dataset

# ## Compute normalization values (just once)

# This block has to be run once before the rest since the splits are not created using the split-folder package, but according to a csv file instead.

# In[ ]:


# splits = ['train', 'validation', 'test']

# # Load Andalucia dataset.
# andalucia_dataset = {x: AndaluciaDataset(
#     root_dir=os.path.join(paths['datasets'], dataset_name),
#     level=dataset_level,
#     split=x,
#     transform=transforms.ToTensor(),
#     target_transform=None
# ) for x in splits}

# # Creating the dataloaders.
# dataloaders = {x: torch.utils.data.DataLoader(
#     andalucia_dataset[x],
#     batch_size=256,
#     worker_init_fn=seed_worker,
#     generator=g
# ) for x in splits}

# # Loop over the train, val, and test datasets.
# print('dataset_mean_std.txt')
# for x in splits:

#     # Computation.
#     # print(f'Samples to be processed: '
#     #       f'{len(dataloaders[x].dataset)}')
#     print(f'{x}')
#     mean, std = get_mean_std_dataloader(dataloaders[x])

#     # Show mean and std.
#     print(f'{mean}')
#     print(f'{std}')


# ## Load normalization values

# In[ ]:


# Retrieve the path, mean and std values of each split from
# a .txt file previously generated using a custom script.
mean, std = load_mean_std_values(
    os.path.join(
        paths['datasets'],
        os.path.join(dataset_name, dataset_level)
    )
)


# In[ ]:


# Rename the key 'val' to 'validation'.
mean['validation'] = mean.pop('val')
std['validation'] = std.pop('val')


# ## Custom transforms

# In[ ]:


splits = ['train', 'validation', 'test']

# Normalization transform (val and test).
transform = {x: transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean[x],
                         std=std[x])
]) for x in splits[1:]}

# Normalization transform (train).
transform['train'] = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean['train'],
                         std['train'])
])

for t in transform:
    print(f'\n{t}: {transform[t]}')


# ## Load custom dataset

# In[ ]:


def transform_abundances(abundances: torch.Tensor) -> torch.Tensor:
    """
    Transforms the abundances from tensor to max value.

    Args:
        abundances (torch.Tensor): list of abundances per batch.
    """

    max_val, max_idx = torch.max(abundances, dim=0)

    return max_idx.item()


# In[ ]:


# Load the Andalucia dataset with normalization.
andalucia_dataset = {x: AndaluciaDataset(
    root_dir=os.path.join(paths['datasets'], dataset_name),
    level='Level_N2',
    split=x,
    train_ratio=dataset_train_pc,
    transform=transform[x],
    target_transform=transform_abundances if task == 'multiclass' else None,
    verbose=True
) for x in splits}


# ## PyTorch dataloaders

# In[ ]:


# Define dataloaders.
dataloader = {x: torch.utils.data.DataLoader(
    andalucia_dataset[x],
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    drop_last=False,
    worker_init_fn=seed_worker,
    generator=g
) for x in splits}

# Check if shuffle is enabled.
if isinstance(dataloader['train'].sampler, torch.utils.data.RandomSampler):
    print('\nShuffle enabled in training!')
else:
    print('\nShuffle disabled in training!')


# In[ ]:


# Get classes and number.
class_names = andalucia_dataset['train'].classes
# print(class_names)

# Get dictionary of classes.
idx_to_class = andalucia_dataset['train'].idx_to_class
# print(idx_to_class)


# ## Check the balance and size of the dataset

# In[ ]:


# Print some stats from the train dataset.
for split in splits:
    print(f"\n#Samples in {split} (from len(dataset)):"
          f"       {len(andalucia_dataset[split])}")
    print(f"#Samples in {split} (from dataloader.sampler):"
          f" {len(dataloader[split].sampler)}")
    print(f"#Samples in {split} (from dataloader.dataset):"
          f" {len(dataloader[split].dataset)}")
    print(f"#Batches in {split} (from dataloader):"
          f"         {len(dataloader[split])}")
    if split == 'train':
        print(f'Train subset ratio (%): {dataset_train_pc}')


# ## Check the distribution of samples in the train dataloader

# In[ ]:


# List to save the labels.
labels_list = []

# Accessing Data and Targets in a PyTorch DataLoader.
t0 = time.time()
for i, (images, labels) in enumerate(dataloader['train']):
    labels_list.append(labels)

# Concatenate list of lists (batches).
labels_list = torch.cat(labels_list, dim=0).numpy()
print(f'\nSample distribution computation in train dataset (s): '
      f'{(time.time()-t0):.2f}')

# Count number of unique values.
data_x, data_y = np.unique(labels_list, return_counts=True)

# New function to plot (suitable for execution in shell).
fig, ax = plt.subplots(1, 1, figsize=(20, 5))
simple_bar_plot(ax,
                data_x,
                'Class',
                data_y,
                'N samples (dataloader)')

plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(left=0.15)
fig_name_save = (f'sample_distribution')
fig.savefig(os.path.join(paths['images'], fig_name_save+fig_format),
            bbox_inches='tight')

plt.show() if show else plt.close()


# ## Look at some training samples

# ### Only one sample from the first batch

# In[ ]:


# Display image and label (w/ or w/o normalization).
train_features, train_labels = next(iter(dataloader['train']))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
img = inv_norm_tensor(
    img,
    mean=mean['train'],
    std=std['train']
)
label = train_labels[0]
print(label)
print(idx_to_class)
if task == 'multiclass':
    class_name = idx_to_class[int(label)]
elif task == 'multilabel':
    class_name = idx_to_class[int(torch.argmax(label))]
plt.title(f'{class_name}')
plt.imshow(torch.permute(img, (1, 2, 0)), cmap="gray")
plt.tight_layout()
plt.show() if show else plt.close()


# ### One batch

# In[ ]:


# Create figure with subplots.
num_rows = int(dataloader['train'].batch_size ** 0.5)
num_cols = (int(dataloader['train'].batch_size / num_rows)
            + (dataloader['train'].batch_size % num_rows > 0))
fig, axes = plt.subplots(nrows=num_rows,
                         ncols=num_cols,
                         figsize=(4*num_cols, 4*num_rows))

# Take only one batch (inverse transform applied).
inv_norm = True
show_one_batch(axes, num_cols, dataloader['train'], task,
               andalucia_dataset['train'].idx_to_class,
               batch_id=0, inv_norm=inv_norm, mean=mean['train'],
               std=std['train'])

# Adjust and show image.
plt.tight_layout()
plt.show() if show else plt.close()


# ***

# ***

# # Finetuning

# Reference 1

# <p style="color:red"><b>-----------------------------------------------------------------</b></p>
# <p style="color:red"><b>----------> REVISED UP TO THIS POINT -----------</b></p>
# <p style="color:red"><b>-----------------------------------------------------------------</b></p>

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
        inputs = data[0].to(device)
        labels = data[1].to(device)

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
            last_loss = running_loss / batch_span_train  # loss per batch
            print(f'  batch {i+1:03d} loss: {last_loss:.4f}')
            tb_x = epoch_index * len(dataloader['train']) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


# ## Models and parameters

# In[ ]:


# Model: resnet with random weights.
if model_name == 'Random':
    print('\nModel without pretrained weights')
    model = torchvision.models.resnet18(weights=None)

    # Get the number of input features to the layer.
    # Adjust the final layer to the current number of classes.
    print(f'\nOld final fully-connected layer: {model.fc}')
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(class_names))
    print(f'New final fully-connected layer: {model.fc}')

    # Parameters of newly constructed modules
    # have requires_grad=True by default.
    # Freezing all the network except the final layer.
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

# Model: resnet with pretrained weights (Imagenet-1k).
elif model_name == 'Imagenet':
    print('\nModel with pretrained weights on imagenet-1k')
    model = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.DEFAULT
    )

    # Get the number of input features to the layer.
    # Adjust the final layer to the current number of classes.
    print(f'\nOld final fully-connected layer: {model.fc}')
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(class_names))
    print(f'New final fully-connected layer: {model.fc}')

    # Parameters of newly constructed modules
    # have requires_grad=True by default.
    # Freezing all the network except the final layer.
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

# Model: resnet with pretrained weights (SSL).
elif model_name in AVAIL_SSL_MODELS:
    print('\nModel with pretrained weights using SSL')
    resnet18 = torchvision.models.resnet18(weights=None)

    # Only backbone.
    pt_backbone = torch.nn.Sequential(*list(resnet18.children())[:-1])

    # List of trained models.
    model_list = []
    print()
    for root, dirs, files in os.walk(paths['log_checkpoints']):
        for filename in files:
            if model_name in filename and 'ckpt' not in filename:
                model_list.append(os.path.join(root, filename))

    # Sort model list and show the items.
    model_list = sorted(model_list, reverse=True)
    for i, model in enumerate(model_list):
        print(f'{i:02} --> {model}')

    # Loading model.
    idx = 0
    pt_backbone.load_state_dict(torch.load(model_list[idx]))
    print(f'\nLoaded: {model_list[idx]}')

    # # Get SSL model name and overwrite it.
    # model_name = model_list[idx].split('/')[10].split('-')[0]
    # print(f'Model name: {model_name}')

    # Adding a linear layer on top of the model (linear classifier).
    # Here, the output of the model is directly passed to the CrossEntropyLoss function
    # without any activation function. The CrossEntropyLoss function applies the softmax
    # activation internally, so you don't need to include a softmax layer in your model.
    model = torch.nn.Sequential(
        pt_backbone,
        torch.nn.Flatten(),
        torch.nn.Linear(in_features=512,
                        out_features=len(class_names),
                        bias=True),
    )

    # Check if the model is loaded on GPU.
    print(f'Saved in GPU: {next(pt_backbone.parameters()).is_cuda}')

    # Get the number of input features to the layer.
    # Adjust the final layer to the current number of classes.
    # print(f'\nOld final fully-connected layer: {model[-1]}')
    # num_ftrs = model[-1].in_features
    # model[-1] = torch.nn.Linear(num_ftrs, len(class_names))
    print(f'New final fully-connected layer: {model[-1]}\n')

    # Parameters of newly constructed modules
    # have requires_grad=True by default.
    # Freezing all the network except the final layer.
    for param in model.parameters():
        param.requires_grad = False
    for param in model[-1].parameters():
        param.requires_grad = True


# In[ ]:


# Show model structure.
if show:
    print(summary(
        model,
        input_size=(batch_size, 3, input_size, input_size),
        device=device)
    )


# In[ ]:


# Set loss.
if task == 'multiclass':
    loss_fn = torch.nn.CrossEntropyLoss()
elif task == 'multilabel':
    loss_fn = torch.nn.BCEWithLogitsLoss()
print(f'\nLoss: {loss_fn}')

# Optimizers specified in the torch.optim package
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
print(f'Optimizer:\n{optimizer}')


# ## Per-epoch activity

# In[ ]:


# ======================
# SET UP WRITERS AND VARIABLES.
# Initializing in a separate cell so we can easily add more epochs to the same run.
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
run_path = os.path.join(paths['runs'],
                        f'{task}_{model_name}_{timestamp}')
print(f'\nRun path: {run_path}')
writer = SummaryWriter(run_path)

# Open the file in the write mode.
header = ['epoch', 'avg_loss', 'avg_vloss']
csv_file = os.path.join(run_path,
                        f'training_info_{task}_{model_name}_{dataset_train_pc:.3f}pc_train.csv')
with open(csv_file, 'w', newline='') as file:
    csv_writer = csv.writer(file)  # Write the header.
    csv_writer.writerow(header)

# Initial variables.
epoch_number = 0
batch_span_train = max(len(dataloader['train']) // 5,
                       len(dataloader['train']) // 2)
batch_span_val = max(len(dataloader['validation']) // 5,
                     len(dataloader['validation']) // 2)
best_vloss = 1_000_000.

# Compile model (only for PT2.0).
if torch_compile:
    model = torch.compile(model)
    torch.set_float32_matmul_precision('high')

# Device used for training.
print(f'Using {device} device')
model.to(device)

# ======================
# TRAINING LOOP.
# Iterating over the epochs.
t0 = time.time()
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
            vinputs = vdata[0].to(device)
            vlabels = vdata[1].to(device)            
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
    # writer.add_scalars('Training vs. Validation Loss',
    #                    {'Training': avg_loss, 'Validation': avg_vloss},
    #                    epoch_number + 1)
    writer.add_scalars('Validation Loss',
                       {'Validation': avg_vloss},
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
        model_path = os.path.join(run_path,
                                  f'{model_name}_vloss={avg_vloss:.4f}_e={epoch_number:03d}_t={timestamp}')
        torch.save(model.state_dict(), model_path)

    epoch_number += 1

print(f'\nTotal duration: {(time.time()-t0):.2f} s')


# In[ ]:


if is_notebook():
    get_ipython().run_line_magic('tensorboard', '--logdir=output/runs/')


# ***

# ***

# # Evaluation on test set

# In[ ]:


# Avoiding "CUDA out of memory" in PyTorch.
torch.cuda.empty_cache()

# Initialize the probabilities, predictions and labels lists.
y_prob = []
y_pred = []
y_true = []

# Initialize the counters for top1 and top5 accuracy.
top1_correct = 0
top5_correct = 0

# Since we're not training, we don't need to calculate
# the gradients for our outputs with torch.no_grad():
batches_test = len(dataloader['test'])
model.eval()
with torch.no_grad():
    for i, data in enumerate(dataloader['test']):

        # Dataset.
        inputs, labels = data[0].to(device), data[1].to(device)

        # Forward pass.
        outputs = model(inputs)

        # Get model predictions.
        if task == 'multiclass':
            # Convert logits to probabilities using a softmax function.
            probs = torch.softmax(outputs, dim=1)
            # Take the argmax of the probabilities to obtain the predicted class labels.
            preds = torch.argmax(probs, dim=1)
        elif task == 'multilabel':
            # Convert logits to probabilities using a sigmoid function.
            probs = torch.sigmoid(outputs)
            # Scale predicted abundances to sum to 1 across all the classes for each sample.
            preds_sum = probs.sum(dim=1, keepdim=True)
            preds = probs / preds_sum

        # Append true and predicted labels to the lists (option 2).
        y_prob.append(probs)
        y_pred.append(preds)
        y_true.append(labels)

        # Compute top1 and top5 accuracy (option 1).
        # if task == 'multiclass':
        #     top1_correct += torch.sum(preds == labels).item()
        #     top5_correct += torch.sum(torch.topk(probs, k=5, dim=1)[1] == labels.view(-1, 1)).item()

        # Progress bar.
        if i % (batches_test//8) == (batches_test//8 - 1) or i == batches_test - 1:
            progress = 100 * (i + 1) // batches_test
            if progress == 100:
                print("Progress: 100%", end='\n', flush=True)
            else:
                print(f"Progress: {progress}%", end='\r', flush=True)


# In[ ]:


if task == 'multiclass':
    # Compute top1 and top5 accuracy (option 1).
    # top1_accuracy = top1_correct / len(dataloader['test'].dataset)
    # top5_accuracy = top5_correct / len(dataloader['test'].dataset)
    # print(f"\nTop-1 Accuracy: {top1_accuracy:.4f}")
    # print(f"Top-5 Accuracy: {top5_accuracy:.4f}")

    # Concatenate the lists into tensors (option 2).
    y_prob_cpu = torch.cat(y_prob).to('cpu')
    y_pred_cpu = torch.cat(y_pred).to('cpu')
    y_true_cpu = torch.cat(y_true).to('cpu')

    # Compute top1 and top5 accuracy (option 2).
    top1_accuracy = torch.sum(torch.eq(y_pred_cpu, y_true_cpu)).item() / len(y_true_cpu)
    top5_accuracy = torch.sum(torch.topk(y_prob_cpu, k=5, dim=1)[1] == y_true_cpu.view(-1, 1)).item() / len(y_true_cpu)
    print(f"\nTop-1 accuracy: {top1_accuracy:.4f}")
    print(f"Top-5 accuracy: {top5_accuracy:.4f}")

    # F1 metrics.
    f1_per_class = f1_score(y_true_cpu, y_pred_cpu, average=None)
    f1_micro = f1_score(y_true_cpu, y_pred_cpu, average='micro')
    f1_macro = f1_score(y_true_cpu, y_pred_cpu, average='macro')
    f1_weighted = f1_score(y_true_cpu, y_pred_cpu, average='weighted')
    print(f"\nPer class F-1 score:\n{f1_per_class}")
    print(f"\nF1-Score micro: {f1_micro:.4f}")
    print(f"F1-Score macro: {f1_macro:.4f}")
    print(f"F1-Score weighted: {f1_weighted:.4f}")

    # Data to be written.
    data2 = []
    header = ['model', 'vloss', 'f1_micro', 'f1_macro', 'f1_weighted', 'f1_per_class', 'top1_accuracy', 'top5_accuracy']
    data = [model_name, avg_vloss, f1_micro, f1_macro, f1_weighted, np.round(f1_per_class, decimal_places), top1_accuracy, top5_accuracy]
    data_rounded = [round(elem, decimal_places) if isinstance(elem, float) else elem for elem in data]
    print(f'\nData to csv: {data_rounded}')

elif task == 'multilabel':

    # Concatenate the lists into tensors.
    y_prob_cpu = torch.cat(y_prob).to('cpu')
    y_pred_cpu = torch.cat(y_pred).to('cpu')
    y_true_cpu = torch.cat(y_true).to('cpu')

    # Compute global RMSE and MAE.
    rmse = mean_squared_error(y_true_cpu, y_pred_cpu, squared=False)
    mae = mean_absolute_error(y_true_cpu, y_pred_cpu)
    print(f"\nRMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    # Compute RMSE and MAE per class.
    rmse_per_class = mean_squared_error(y_true_cpu, y_pred_cpu, multioutput='raw_values', squared=False)
    mae_per_class = mean_absolute_error(y_true_cpu, y_pred_cpu, multioutput='raw_values')
    print(f"\nRMSE per class:\n{rmse_per_class}")
    print(f"MAE per class:\n{mae_per_class}")

    # Data to be written.
    header = ['model', 'vloss', 'rmse', 'mae', 'rmse_per_class', 'mae_per_class']
    data = [model_name, avg_vloss, rmse, mae, np.round(rmse_per_class, decimal_places), np.round(mae_per_class, decimal_places)]
    data_rounded = [round(elem, decimal_places) if isinstance(elem, float) else elem for elem in data]
    print(f'\nData to csv: {data_rounded}')


# In[ ]:


# Open the file in the write mode.
csv_file = os.path.join(paths['runs'], f'results_on_test_set_{task}_{dataset_train_pc:.3f}pc_train.csv')

# Check if file exists (header).
if not os.path.isfile(csv_file):
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

# Write data.
with open(csv_file, 'a', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow(data_rounded)


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
