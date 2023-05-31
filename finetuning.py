# Custom modules.
from utils.other import build_paths
from utils.reproducibility import set_seed, seed_worker
from utils.dataset import (
    AndaluciaDataset,
    load_mean_std_values
)
from utils.simsiam import SimSiam
from utils.simclr import SimCLR
from utils.mocov2 import MoCov2
from utils.barlowtwins import BarlowTwins
import pandas as pd

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

# PyTorch DDP.
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from finetuning_trainer import Trainer

AVAIL_SSL_MODELS = ['BarlowTwins', 'MoCov2', 'SimCLR', 'SimCLRv2', 'SimSiam']
MODEL_CHOICES = ['Random', 'Imagenet'] + AVAIL_SSL_MODELS
SEED = 42


def get_args() -> argparse.Namespace:
    """
    Parse and retrieve command-line arguments.

    Returns:
        An 'argparse.Namespace' object containing the parsed arguments.
    """

    # Get arguments.
    parser = argparse.ArgumentParser(
        description='Script for training the self-supervised learning models.'
    )

    # General arguments.
    parser.add_argument('model_name', type=str,
                        choices=MODEL_CHOICES,
                        help='target SSL model.')

    parser.add_argument('task_name', type=str, choices=['multiclass', 'multilabel'],
                        help="downstream task.")

    parser.add_argument('--backbone_name', '-bn', type=str, default='resnet18',
                        choices=['resnet18', 'resnet50'],
                        help='backbone model name (default: resnet18).')

    parser.add_argument('--input_data', '-id', type=str,
                        help='path to the input directory (if necessary).')

    parser.add_argument('--dataset_name', '-dn', type=str,
                        default='Sentinel2AndaluciaLULC',
                        help='dataset name for training '
                             '(default: Sentinel2AndaluciaLULC).')
    
    parser.add_argument('--dataset_level', '-dl', type=str,default='Level_N2',
                        choices=['Level_N1', 'Level_N2'],
                        help="dataset level (default=Level_N2).")

    parser.add_argument('--dataset_train_pc', '-dtp', type=float, default=1.,
                        help='dataset ratio for train subset (default=1.).')

    parser.add_argument('--epochs', '-e', type=int, default=25,
                        help='number of epochs for training (default: 25).')

    parser.add_argument('--save_every', '-se', type=int, default=5,
                        help='save model checkpoint every n epochs (default: 5).')

    parser.add_argument('--batch_size', '-bs', type=int, default=64,
                        help='number of images in a batch during training '
                             '(default: 64).')

    parser.add_argument('--num_workers', '-nw', type=int, default=1,
                        help='number of subprocesses to use for data loading. '
                             '0 means that the data will be loaded in the '
                             'main process (default: 1).')

    parser.add_argument('--show', '-s', action='store_true',
                        help='the images pops up.')

    parser.add_argument('--verbose', '-v', action='store_true',
                        help='provides additional details for debugging purposes.')

    parser.add_argument('--torch_compile', '-tc', action='store_true',
                        help='PyTorch 2.0 compile enabled.')

    parser.add_argument('--distributed', '-d', action='store_true',
                        help='enables distributed training.')

    return parser.parse_args(sys.argv[1:])


def ddp_setup():
    """
    Initializes the default distributed process group,
    and this will also initialize the distributed package.
    """
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def transform_abundances(
        abundances: torch.Tensor
) -> torch.Tensor:
    """
    Transforms the abundances from tensor to max value.

    Args:
        abundances (torch.Tensor): list of abundances per batch.
    """

    max_val, max_idx = torch.max(abundances, dim=0)

    return max_idx.item()



def main(args):

    ddp_setup()

    # Enable reproducibility.
    print(f"\n{'torch initial seed:'.ljust(33)}{torch.initial_seed()}")
    g = set_seed(SEED)
    print(f"{'torch current seed:'.ljust(33)}{torch.initial_seed()}")

    # Check torch CUDA and CPUs available (for num_workers).
    print(f"{'torch.cuda.is_available():'.ljust(33)}"
          f"{torch.cuda.is_available()}")
    print(f"{'torch.cuda.device_count():'.ljust(33)}"
          f"{torch.cuda.device_count()}")
    print(f"{'torch.cuda.current_device():'.ljust(33)}"
          f"{torch.cuda.current_device()}")
    print(f"{'torch.cuda.device(0):'.ljust(33)}"
          f"{torch.cuda.device(0)}")
    print(f"{'torch.cuda.get_device_name(0):'.ljust(33)}"
          f"{torch.cuda.get_device_name(0)}")
    print(f"{'torch.backends.cudnn.benchmark:'.ljust(33)}"
          f"{torch.backends.cudnn.benchmark}")
    print(f"{'os.sched_getaffinity:'.ljust(33)}"
          f"{len(os.sched_getaffinity(0))}")
    print(f"{'os.cpu_count():'.ljust(33)}"
          f"{os.cpu_count()}")

    # Convert the parsed arguments into a dictionary and declare
    # variables with the same name as the arguments.
    print()
    args_dict = vars(args)
    for arg_name in args_dict:
        arg_name_col = f'{arg_name}:'
        print(f'{arg_name_col.ljust(20)} {args_dict[arg_name]}')

    # Build paths.
    print()
    cwd = os.getcwd()
    paths = build_paths(cwd, os.path.join('finetuning', args.model_name))
    if args.input_data:
        paths['datasets'] = args.input_data

    # Show built paths.
    if args.verbose:
        for path in paths:
            path_name_col = f'{path}:'
            print(f'{path_name_col.ljust(20)} {paths[path]}')

    # Size of the images.
    input_size = 224

    # Format of the saved images.
    fig_format = '.png'

    # Number of digits to the right of the decimal
    decimal_places = 2

    # ======================
    # DATASET.
    # ======================
    # Default values.
    sampler = None
    shuffle = True

    #--------------------------
    # Load normalization values.
    #--------------------------

    # Retrieve the path, mean and std values of each split from
    # a .txt file previously generated using a custom script.
    mean, std = load_mean_std_values(
        os.path.join(
            paths['datasets'],
            os.path.join(args.dataset_name, args.dataset_level)
        )
    )

    # Rename the key 'val' to 'validation'.
    mean['validation'] = mean.pop('val')
    std['validation'] = std.pop('val')

    if args.verbose:
        print(f'\nMean: {mean}')
        print(f'Std:  {std}')

    #--------------------------
    # Custom transforms.
    #--------------------------

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

    if args.verbose:
        for t in transform:
            print(f'\n{t}: {transform[t]}')

    #--------------------------
    # Load custom dataset.
    #--------------------------

    # Load the Andalucia dataset with normalization.
    andalucia_dataset = {x: AndaluciaDataset(
        root_dir=os.path.join(paths['datasets'], args.dataset_name),
        level='Level_N2',
        split=x,
        train_ratio=args.dataset_train_pc,
        transform=transform[x],
        target_transform=transform_abundances if args.task_name == 'multiclass' else None,
        verbose=args.verbose
    ) for x in splits}


    #--------------------------
    # If distributed (option).
    #--------------------------
    if args.distributed:
        sampler=DistributedSampler(andalucia_dataset['train'])
        shuffle=False

    #--------------------------
    # PyTorch dataloaders.
    #--------------------------
    print(f'Sampler: {sampler}')
    print(f'Shuffle: {shuffle}')

    # Define dataloaders.
    dataloader = {x: torch.utils.data.DataLoader(
        andalucia_dataset[x],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g
    ) for x in splits}

    if args.verbose:
        for d in dataloader:
            print(f'\n{d}: {vars(dataloader[d])}')

    # Get classes and number.
    # Get dictionary of classes.
    class_names = andalucia_dataset['train'].classes
    idx_to_class = andalucia_dataset['train'].idx_to_class
    if args.verbose:
        print(f'\n{class_names}')
        print(f'{idx_to_class}')

    #--------------------------
    # Check the balance and size of the dataset.
    #--------------------------
    # Check samples per class, total samples and batches of each dataset.
    for d in andalucia_dataset:
        # samples = np.unique(andalucia_dataset[d].targets, return_counts=True)[1]
        print(f'\n{d}:')
        # print(f'  - #Samples (from dataset):  {len(dataset[d].targets)}')
        # print(f'  - #Samples/class (from dataset):\n{samples}')
        print(f'  - #Batches (from dataloader): {len(dataloader[d])}')
        print(f'  - #Samples (from dataloader): {len(dataloader[d])*args.batch_size}')

    #--------------------------
    # Check the distribution of samples in the dataloader.
    #--------------------------
    # Not copied yet.

    #--------------------------
    # Look at some training samples.
    #--------------------------
    # Not copied yet.

    # ======================
    # FINE-TUNING.
    # ======================

    #--------------------------
    # Models and parameters.
    #--------------------------

    # Model: resnet with random weights.
    if args.model_name == 'Random':
        print('\nModel without pretrained weights')
        model = torchvision.models.resnet18(weights=None)

        # Get the number of input features to the layer.
        # Adjust the final layer to the current number of classes.
        print(f'Old final fully-connected layer: {model.fc}')
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
    elif args.model_name == 'Imagenet':
        print('\nModel with pretrained weights on imagenet-1k')
        model = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT
        )

        # Get the number of input features to the layer.
        # Adjust the final layer to the current number of classes.
        print(f'Old final fully-connected layer: {model.fc}')
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
    elif args.model_name in AVAIL_SSL_MODELS:
        print('\nModel with pretrained weights using SSL')
        resnet = torchvision.models.resnet18(weights=None)

        snapshot_name = f'snapshot_{args.model_name}_{args.backbone_name}.pt'
        snapshot = torch.load(os.path.join(paths['input'], snapshot_name))
        print(f'Model loaded from {snapshot_name}')

        # Removing head from resnet: Encoder.
        backbone = torch.nn.Sequential(*list(resnet.children())[:-1])
        input_dim = resnet.fc.in_features

        paths['ray_tune'] = os.path.join(paths['input'], 'best_configs')

        # Build the filename.
        filename_lr = f'ray_tune_results_lr_{args.backbone_name}_{args.model_name}.csv'

        # Load the CSV file into a pandas dataframe.
        df_lr = pd.read_csv(os.path.join(paths['ray_tune'], filename_lr),
                            usecols=lambda col: col.startswith('loss')
                            or col.startswith('config/'))
    
        hidden_dim = df_lr.loc[0, 'config/hidden_dim']
        out_dim = df_lr.loc[0, 'config/out_dim']

        print(f"{'Model name:'.ljust(18)} {args.model_name}")
        print(f"{'Backbone name:'.ljust(18)} {args.backbone_name}")
        print(f"{'Hidden layer dim.:'.ljust(18)} {hidden_dim}")
        print(f"{'Output layer dim.:'.ljust(18)} {out_dim}")

        if args.model_name == 'SimSiam':
            model = SimSiam(backbone=backbone, input_dim=input_dim, proj_hidden_dim=out_dim,
                            pred_hidden_dim=hidden_dim, output_dim=out_dim)
        elif args.model_name == 'SimCLR':
            model = SimCLR(backbone=backbone, input_dim=input_dim,
                        hidden_dim=hidden_dim, output_dim=out_dim,
                        num_layers=2, memory_bank_size=0)
        elif args.model_name == 'SimCLRv2':
            model = SimCLR(backbone=backbone, input_dim=input_dim,
                        hidden_dim=hidden_dim, output_dim=out_dim,
                        num_layers=3, memory_bank_size=65536)
        elif args.model_name == 'BarlowTwins':
            model = BarlowTwins(backbone=backbone, input_dim=input_dim,
                                hidden_dim=hidden_dim, output_dim=out_dim)
        elif args.model_name == 'MoCov2':
            model = MoCov2(backbone=backbone, input_dim=input_dim,
                        hidden_dim=hidden_dim, output_dim=out_dim)

        model.load_state_dict(snapshot["MODEL"])

        # Removing head from resnet: Encoder.
        model = torch.nn.Sequential(
            model.backbone,
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=input_dim,
                            out_features=len(class_names),
                            bias=True),
        )

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

    # Setting the device.
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = 0
    print(f"Device: {device}")

    # Show model structure.
    if args.verbose:
        print(summary(
            model,
            input_size=(args.batch_size, 3, input_size, input_size),
            device=device)
        )

    # Configure the loss.
    if args.task_name == 'multiclass':
        loss_fn = torch.nn.CrossEntropyLoss()
    elif args.task_name == 'multilabel':
        loss_fn = torch.nn.BCEWithLogitsLoss()
    if args.verbose:
        print(f'\nLoss: {loss_fn}')

    # Configure the optimizer.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    if args.verbose:
        print(f'Optimizer:\n{optimizer}')

    # Training.
    trainer = Trainer(
        model, dataloader, loss_fn,
        optimizer,
        save_every=args.save_every,
        snapshot_path=f'snapshot_{args.task_name}_pctrain_{args.dataset_train_pc:.3f}_{args.model_name}.pt'
    )
    trainer.train(args.epochs, args, test=True, save_csv=True)


    return 0


if __name__ == "__main__":

    # Get arguments.
    args = get_args()

    # Main function.
    sys.exit(main(args))
