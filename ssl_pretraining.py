# Custom modules.
from utils.other import build_paths
from utils.reproducibility import set_seed, seed_worker
from utils.dataset import load_dataset_based_on_ratio, GaussianBlur
from utils.simsiam import SimSiam
from utils.simclr import SimCLR
from utils.mocov2 import MoCov2
from utils.barlowtwins import BarlowTwins
from utils.graphs import simple_bar_plot
from utils.check_embeddings import (
    create_list_embeddings,
    pca_computation,
    tsne_computation
)
from ssl_eval import linear_eval_backbone
import seaborn as sns
from utils.dataset import inv_norm_tensor

# Arguments and paths.
import os
import sys
import argparse
import csv
import copy

# PyTorch.
import torch
import torch.nn as nn
import torchvision
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
    resnet50,
    ResNet50_Weights
)
from torchvision import transforms
from torchinfo import summary

# PyTorch DDP.
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# Data management.
import numpy as np
import pandas as pd

# SSL library.
import lightly

# Training checks.
import time
import math

# Hyperparameter tunning.
from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

# For plotting.
import matplotlib.pyplot as plt

AVAIL_SSL_MODELS = ['SimSiam', 'SimCLR', 'SimCLRv2', 'BarlowTwins', 'MoCov2']
NUM_DECIMALS = 3
FIG_FORMAT='.png'


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
                        choices=AVAIL_SSL_MODELS,
                        help='target SSL model.')

    parser.add_argument('--backbone_name', '-bn', type=str, default='resnet18',
                        choices=['resnet18', 'resnet50'],
                        help='backbone model name (default: resnet18).')

    parser.add_argument('--input_data', '-id', type=str,
                        help='path to the input directory (if necessary).')

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

    parser.add_argument('--save_every', '-se', type=int, default=0,
                        help='save model checkpoint every n epochs (default: 0, no checkpoints).')

    parser.add_argument('--eval_every', '-ee', type=int, default=0,
                        help='online linear evaluation every n epochs (default: 0, no evaluation).')

    parser.add_argument('--batch_size', '-bs', type=int, default=64,
                        help='number of images in a batch during training '
                             '(default: 64).')

    parser.add_argument('--num_workers', '-nw', type=int, default=1,
                        help='number of subprocesses to use for data loading. '
                             '0 means that the data will be loaded in the '
                             'main process (default: 1).')

    parser.add_argument('--ini_weights', '-iw', type=str, default='random',
                        choices=['random', 'imagenet'],
                        help='initial weights (default: random).')

    parser.add_argument('--seed', '-s', type=int, default=42,
                        help='seed for the experiments (default: 42).')

    parser.add_argument('--show', '-sw', action='store_true',
                        help='the images pops up.')

    parser.add_argument('--verbose', '-v', action='store_true',
                        help='provides additional details for debugging purposes.')

    parser.add_argument('--balanced_dataset', '-bd', action='store_true',
                        help='whether the dataset should be balanced.')

    parser.add_argument('--reduced_dataset', '-rd', action='store_true',
                        help='whether the dataset should be reduced.')

    parser.add_argument('--torch_compile', '-tc', action='store_true',
                        help='PyTorch 2.0 compile enabled.')

    parser.add_argument('--partially_frozen', '-pf', action='store_true',
                        help='freezes the weights of the first layers.')

    parser.add_argument('--distributed', '-d', action='store_true',
                        help='enables distributed training.')

    # Specific for Ray Tune.
    parser.add_argument('--ray_tune', '-rt', type=str,
                        choices=['gridsearch', 'loguniform'],
                        help='enables Ray Tune (tunes everything or only lr).')

    parser.add_argument('--grace_period', '-rtgp', type=int,
                        help='only stop trials at least this old in time.')

    parser.add_argument('--num_samples_trials', '-rtnst', type=int,
                        help='number of samples to tune the hyperparameters.')

    parser.add_argument('--gpus_per_trial', '-rtgpt', type=int,
                        help='number of gpus to be used per trial.')

    return parser.parse_args(sys.argv[1:])


def ddp_setup():
    """
    Initializes the default distributed process group,
    and this will also initialize the distributed package.
    """
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def show_two_batches(
        batch1: torch.Tensor = None,
        batch2: torch.Tensor = None,
        batch_id: int = None,
        **kwargs
) -> None:
    """
    Shows the images in two batches, one above the other, in the same figure.

    Args:
        batch1 (torch.Tensor): first batch of images.
        batch2 (torch.Tensor): second batch of images.
        batch_id (int): batch identification number.
    """

    # Figure settings
    batch_size = batch1.shape[0]
    total_images = batch_size * 2
    columns = 8
    rows = math.ceil(total_images / columns)

    # Calculate width and height based on the number of columns and rows
    width = columns * 5
    height = rows * 5

    # Figure creation and show
    fig, axes = plt.subplots(nrows=rows, ncols=columns, figsize=(width, height))
    fig.suptitle(f'Batch {batch_id}')

    # Iterate over two times the batch size.
    for i in range(total_images):
        row = i // columns
        col = i % columns
        if i < batch_size:
            img = batch1[i]
            pbatch = 1
            pimage = i
        else:
            img = batch2[i % batch_size]
            pbatch = 2
            pimage = i % batch_size
        if 'mean' in kwargs and 'std' in kwargs:  # Revert normalization
            img = inv_norm_tensor(
                img,
                mean=kwargs['mean']['train'],
                std=kwargs['std']['train']
            )
        axes[row, col].imshow(torch.permute(img, (1, 2, 0)))
        axes[row, col].set_title(f'Batch {pbatch} - Image {pimage}')
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()


def save_snapshot(snapshot_path, epoch, model_state_dict, optimizer, warmup_scheduler, cosine_scheduler):
    snapshot = {
        "EPOCH": epoch + 1,  # +1 so that the training resumes at the same point.
        "MODEL": model_state_dict,
        "OPTIMIZER": optimizer.state_dict(),
        "WARMUP_SCHEDULER": warmup_scheduler.state_dict(),
        "COSINE_SCHEDULER": cosine_scheduler.state_dict()
    }
    torch.save(snapshot, snapshot_path)
    print(f"Training snapshot saved --> {snapshot_path.rsplit('/', 1)[-1]}")


def load_snapshot(snapshot_path, local_rank, model, optimizer, warmup_scheduler, cosine_scheduler):
    loc = f"cuda:{local_rank}"
    snapshot = torch.load(snapshot_path, map_location=loc)
    epoch = snapshot["EPOCH"]
    model.load_state_dict(snapshot["MODEL"])
    optimizer.load_state_dict(snapshot['OPTIMIZER'])
    warmup_scheduler.load_state_dict(snapshot['WARMUP_SCHEDULER'])
    cosine_scheduler.load_state_dict(snapshot['COSINE_SCHEDULER'])
    print(f"Resuming training from snapshot at Epoch {epoch} <-- {snapshot_path.rsplit('/', 1)[-1]}")

    return epoch, model, optimizer, warmup_scheduler, cosine_scheduler


def save_to_csv(csv_file, data):
    # Open the file in the append mode.
    with open(csv_file, 'a', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(data)


def train(
    config: dict = None
) -> None:
    """
    Trains the SSL models.

    Args:
        config (dict): training configuration hyperparameters.
    """

    # Retrieve arguments from the dictionary (clean code).
    args = config['args']

    # ======================
    # DEVICES.
    # Setting the device.
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = int(os.environ["LOCAL_RANK"])
    # gpu_id = int(os.environ["LOCAL_RANK"])  # WORKS.
    # print(f"\n{'Device:'.ljust(18)} {gpu_id}")

    # Unique identifier across all the nodes.
    global_rank = int(os.environ["RANK"])
    print(f"\n{'Global_rank:'.ljust(18)} {global_rank}")

    # Uniquely identifies each GPU-process on a node
    local_rank = int(os.environ["LOCAL_RANK"])
    print(f"{'Local_rank:'.ljust(18)} {local_rank}")

    # ======================
    # DEFINE MODELS.
    # Setting the model and initial weights.
    if args.backbone_name == 'resnet18':
        if args.ini_weights == 'imagenet':
            resnet = resnet18(
                weights=ResNet18_Weights.DEFAULT,
                # zero_init_residual=True
            )
        elif args.ini_weights == 'random':
            resnet = resnet18(
                weights=None,
                # zero_init_residual=True
            )
    elif args.backbone_name == 'resnet50':
        if args.ini_weights == 'imagenet':
            resnet = resnet50(
                weights=ResNet50_Weights.DEFAULT,
                # zero_init_residual=True
            )
        elif args.ini_weights == 'random':
            resnet = resnet50(
                weights=None,
                # zero_init_residual=True
            )

    # Removing head from resnet: Encoder.
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    input_dim = hidden_dim = resnet.fc.in_features
    print(f"{'Model name:'.ljust(18)} {args.model_name}")
    print(f"{'Backbone name:'.ljust(18)} {args.backbone_name}")
    print(f"{'Hidden layer dim.:'.ljust(18)} {config['hidden_dim']}")
    print(f"{'Output layer dim.:'.ljust(18)} {config['out_dim']}")
    print(f"{'Momentum:'.ljust(18)} {config['momentum']}")
    print(f"{'Weight decay:'.ljust(18)} {config['weight_decay']}")

    if args.model_name == 'SimSiam':
        model = SimSiam(backbone=backbone, input_dim=input_dim, proj_hidden_dim=config['out_dim'],
                        pred_hidden_dim=config['hidden_dim'], output_dim=config['out_dim'])
    elif args.model_name == 'SimCLR':
        model = SimCLR(backbone=backbone, input_dim=input_dim,
                       hidden_dim=config['hidden_dim'], output_dim=config['out_dim'],
                       num_layers=2, memory_bank_size=0)
    elif args.model_name == 'SimCLRv2':
        model = SimCLR(backbone=backbone, input_dim=input_dim,
                       hidden_dim=config['hidden_dim'], output_dim=config['out_dim'],
                       num_layers=3, memory_bank_size=65536)        # 4096
    elif args.model_name == 'BarlowTwins':
        model = BarlowTwins(backbone=backbone, input_dim=input_dim,
                            hidden_dim=config['hidden_dim'], output_dim=config['out_dim'])
    elif args.model_name == 'MoCov2':
        model = MoCov2(backbone=backbone, input_dim=input_dim,
                       hidden_dim=config['hidden_dim'], output_dim=config['out_dim'])
    # Send to GPU.
    model = model.to(local_rank)

    # ======================
    # CONFIGURE OPTIMIZER AND SCHEDULERS.
    # Set the initial learning rate.
    lr_init = config["lr"]
    print(f"{'Initial lr:'.ljust(18)} {lr_init}")

    # Use SGD with momentum and weight decay.
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr_init,
        momentum=config["momentum"],            # 0.9
        weight_decay=config["weight_decay"]     # 5e-4
    )

    # Define the warmup duration.
    warmup_epochs = config['warmup_epochs']
    print(f"{'Warmup epochs:'.ljust(18)} {warmup_epochs}")
    print(f"{'Total epochs:'.ljust(18)} {args.epochs}")

    # Linear warmup for the first defined epochs.
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: min(1, epoch / warmup_epochs),
        verbose=args.verbose
    )

    # Cosine decay afterwards.
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs-warmup_epochs,
        verbose=args.verbose
    )

    # # ======================
    # # LOAD CHECKPOINTS (IF ENABLED).
    # if args.resume_training:

    #     # List of checkpoints.
    #     ckpt_list = []
    #     print()
    #     for root, dirs, files in os.walk(config['paths']['checkpoints']):
    #         for i, filename in enumerate(sorted(files, reverse=True)):
    #             if filename[:4] == 'ckpt' and args.backbone_name in filename:
    #                 ckpt_list.append(os.path.join(root, filename))
    #                 print(f'{i:02} --> {filename}')

    #     # Load the best checkpoint.
    #     print(f'\nLoaded: {ckpt_list[0]}')
    #     ckpt = torch.load(ckpt_list[0])

    #     # Load from dict.
    #     epoch = ckpt['epoch'] + 1
    #     model.backbone.load_state_dict(ckpt['model_state_dict'])
    #     optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    #     warmup_scheduler.load_state_dict(ckpt['warmup_scheduler_state_dict'])
    #     cosine_scheduler.load_state_dict(ckpt['cosine_scheduler_state_dict'])

    # else:
    #     epoch = 0  # start training from scratch.

    epoch = 0

    # General name for csv files and snapshots.
    general_name = f'{args.model_name}_{args.backbone_name}_bd={args.balanced_dataset}_iw={args.ini_weights}'

    # ======================
    # LOADING SNAPSHOT (IF EXISTS).
    # Path to folder and file.
    snapshot_path = os.path.join(
        config['paths']['snapshots'],
        f'snapshot_{general_name}.pt'
    )

    # Load if exists a previous snapshot.
    if not args.ray_tune and os.path.isfile(snapshot_path):
        print("\nLoading snapshot...")
        epoch, model, optimizer, warmup_scheduler, cosine_scheduler = load_snapshot(snapshot_path, local_rank, model, optimizer, warmup_scheduler, cosine_scheduler)

    # Parameters of newly constructed modules
    # have requires_grad=True by default.
    # Freezing all the network except the first
    # part of the backbone to preserve imagenet weights.
    for param in model.backbone.parameters():
        param.requires_grad = True
    if args.partially_frozen and args.ini_weights == 'imagenet':
        print('\nThe first layers of the backbone are frozen!')
        for param in model.backbone[:5].parameters():
            param.requires_grad = False

    # ======================
    # COMPILING (IF ENABLED) AND GPU SUPPORT.
    # Compile model (only for PT2.0).
    if args.torch_compile:
        model = torch.compile(model)
        torch.set_float32_matmul_precision('high')

    # Device used for training.
    if args.distributed:
        model = DDP(model, device_ids=[local_rank])  ## model.to(device)

    # ======================
    # INITIAL PARAMETERS AND INFO.
    total_train_batches = len(config['dataloader']['train'])
    total_val_batches = len(config['dataloader']['val'])
    no_collapse = 1./math.sqrt(config['out_dim'])
    if args.verbose:
        print(f'\nOptimizer:\n{optimizer}')
        print(f'Warmup scheduler: {warmup_scheduler}')
        print(f'Cosine scheduler: {cosine_scheduler}')
        print(f'Batches in (train, val) datasets: '
              f'({total_train_batches}, {total_val_batches})')
    csv_path=os.path.join(config['paths']['csv_results'], f'csv_{general_name}.csv')

    # ======================
    # TRAINING LOOP.
    # Iterating over the epochs.
    for epoch in range(epoch, args.epochs):

        # Timer added.
        t0 = time.time()

        # ======================
        # TRAINING COMPUTATION.
        # Initialization.
        print(f"\nLearning rate: {optimizer.param_groups[0]['lr']}")
        collapse_level = 0.
        running_train_loss = 0.
        model.train()

        # Show model's backbone structure.
        if args.verbose:
            summary(
                resnet,
                input_size=(config['bsz'], 3,
                            config['input_size'],
                            config['input_size']),
                device=local_rank
            )

        # Iterating through the dataloader (lightly dataset is different).
        if args.distributed:
            config['dataloader']['train'].sampler.set_epoch(epoch)
        for b, ((x0, x1), _, _) in enumerate(config['dataloader']['train']):

            # Move images to the GPU (same batch with two transformations).
            x0 = x0.to(local_rank)
            x1 = x1.to(local_rank)

            # Zero the parameter gradients; otherwise,
            # they will be accumulated per batch.
            optimizer.zero_grad()

            # Forward + backward + optimize: Compute the loss, run
            # backpropagation, and update the parameters of the model.
            loss = model(x0, x1)
            loss.backward()
            optimizer.step()

            # The level of collapse is large if the standard deviation of
            # the l2 normalized output is much smaller than 1 / sqrt(out_dim).
            # 0 means collapse; the closer to the upper value, the better.
            if args.model_name == 'SimSiam':
                if args.distributed:
                    collapse_level = model.module.check_collapse()
                else:
                    collapse_level = model.check_collapse()

            # Print statistics.
            # Averaged loss across all training examples * batch_size.
            running_train_loss += loss.detach() * config['bsz']

            # Show partial stats (only four times per epoch).
            if args.verbose:
                if b % (total_train_batches//4) == (total_train_batches//4-1):
                    print(
                        f'[GPU:{global_rank}] | '
                        f'T[{epoch},{b+1:5d}] | '
                        f'Averaged loss: '
                        f"{running_train_loss/(b*config['bsz']):.4f}"
                    )

        # ======================
        # TRAINING LOSS.
        # Loss averaged across all training examples for the current epoch.
        epoch_train_loss = (running_train_loss
                            / len(config['dataloader']['train'].sampler))

        # ======================
        # UPDATE LEARNING RATE SCHEDULER.
        if (epoch < warmup_epochs):
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        # ======================
        # EPOCH STATISTICS.
        # Show some stats per epoch completed.
        print(
            f"[GPU:{global_rank}] | "
            f"[Epoch: {epoch}] | "
            f"Train loss: {epoch_train_loss:.4f} | "
            f"Steps (nb): {len(config['dataloader']['train'])} | "
            # f"Val loss: {epoch_val_loss:.4f} | "
            f"Duration: {(time.time()-t0):.2f} s | "
            f"Collapse (SimSiam): {collapse_level:.4f}/{no_collapse:.4f}\n"
        )

        # ======================
        # SAVING CHECKPOINT.
        # Custom functions for saving the checkpoints.
        if (global_rank == 0 and not args.ray_tune and not args.save_every == 0) and (epoch % args.save_every == 0 or epoch == args.epochs - 1):

            if args.distributed:
                model_state_dict = model.module.state_dict()
            else:
                model_state_dict = model.state_dict()

            # Single snapshot (overwritten) in case of failure.
            save_snapshot(snapshot_path, epoch, model_state_dict, optimizer, warmup_scheduler, cosine_scheduler)

            # Checkpoint every few epochs.
            save_snapshot(
                os.path.join(
                    config['paths']['checkpoints'],
                    f'ckpt_{general_name}_epoch={epoch:03d}.zip'
                ),
                epoch, model_state_dict, optimizer, warmup_scheduler, cosine_scheduler
            )

        # ======================
        # SAVING CSV FILE.
        if global_rank == 0 and not args.ray_tune:
            if epoch == 0:
                header = ['epoch', 'train_loss']
                with open(csv_path, 'w', newline='') as file:
                    csv_writer = csv.writer(file)
                    csv_writer.writerow(header)

            data = [epoch_train_loss]

            data_rounded = [format(elem, f'.{NUM_DECIMALS}f') if not isinstance(elem, list) else elem for elem in data]
            save_to_csv(csv_path, [f"{epoch:02d}"]+data_rounded)

        # ======================
        # LINEAR EVALUATION.
        if (global_rank == 0 and not args.ray_tune and not args.eval_every == 0) and (epoch % args.eval_every == 0 or epoch == args.epochs - 1):

            if epoch == args.epochs - 1:
                eval_epochs = 50
            else:
                eval_epochs = 10

            if args.distributed:
                fronzen_backbone = copy.deepcopy(model.module.backbone)
            else:
                fronzen_backbone = copy.deepcopy(model.backbone)

            model.eval()
            print(f'\nEvaluating ({eval_epochs} epochs)...')
            linear_eval_backbone(
                epoch,
                eval_epochs,
                fronzen_backbone,
                input_dim,
                29,
                config['dataloader'],
                config['bsz'],
                local_rank,
                config['paths'],
                args,
                general_name,
                input_size=224,
                verbose=False,
                dropout=False
            )
            print('Evaluation done!')

        # ======================
        # RAY TUNE REPORTING STAGE.
        if args.ray_tune:
            tune.report(loss=epoch_train_loss.cpu())

    print('\nGenerating the embeddings...')
    embeddings, labels = create_list_embeddings(model, config['dataloader'], local_rank, distributed=args.distributed)

    # t-SNE computation for 2-D.
    df = tsne_computation(embeddings, labels, args.seed, n_components=2)

    # 2-D plot.
    fig_name_save = (f'tsne_2d-{general_name}')
    fig = plt.figure(figsize=(10, 10))
    sns.set(font_scale=1.6)
    sns.scatterplot(
        x='tsne_x',
        y='tsne_y',
        hue='labels',
        palette=sns.color_palette('hls', 29),
        data=df,
        legend='full',
        alpha=0.9
    ).set_title(fig_name_save)
    plt.legend(loc='right', fontsize='12', title_fontsize='12')
    fig.savefig(os.path.join(config['paths']['images'], fig_name_save+FIG_FORMAT),
                bbox_inches='tight')
    plt.show() if args.show else plt.close()

    # PCA computation for 2-D.
    df = pca_computation(embeddings, labels, args.seed)

    # 2-D plot.
    fig_name_save = (f'pca_2d-{general_name}')
    fig = plt.figure(figsize=(10, 10))
    sns.set(font_scale=1.6)
    sns.scatterplot(
        x='pca_x',
        y='pca_y',
        hue='labels',
        palette=sns.color_palette('hls', 29),
        data=df,
        legend='full',
        alpha=0.9
    ).set_title(fig_name_save)
    plt.legend(loc='right', fontsize='12', title_fontsize='12')
    fig.savefig(os.path.join(config['paths']['images'], fig_name_save+FIG_FORMAT),
                bbox_inches='tight')
    plt.show() if args.show else plt.close()


def main(args):

    ddp_setup()

    # Enable reproducibility.
    print(f"\n{'torch initial seed:'.ljust(33)}{torch.initial_seed()}")
    g = set_seed(args.seed)
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
    paths = build_paths(cwd, 'pretraining')
    if args.input_data:
        paths['datasets'] = args.input_data

    # Show built paths.
    if args.verbose:
        for path in paths:
            path_name_col = f'{path}:'
            print(f'{path_name_col.ljust(20)} {paths[path]}')

    # Size of the images.
    input_size = 224

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
    paths[args.dataset_name], mean, std = load_dataset_based_on_ratio(
        paths['datasets'],
        args.dataset_name,
        args.dataset_ratio,
        args.verbose
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

    if args.verbose:
        for t in transform:
            print(f'\n{t}: {transform[t]}')

    #--------------------------
    # ImageFolder.
    #--------------------------
    # Loading the three datasets with ImageFolder.

    dataset = {x: torchvision.datasets.ImageFolder(
        os.path.join(paths[args.dataset_name], x),
        transform[x]
    ) for x in splits[1:]}

    dataset['train'] = torchvision.datasets.ImageFolder(
        os.path.join(paths[args.dataset_name], 'train')
    )

    if args.verbose:
        for d in dataset:
            print(f'\n{d}: {dataset[d]}')
            print(f'\n{d}: {dataset[d].class_to_idx}')

    #--------------------------
    # Dealing with imbalanced data (option).
    #--------------------------
    if args.balanced_dataset:

        # Creating a list of labels of samples.
        train_sample_labels = dataset['train'].targets

        # Calculating the number of samples per label/class.
        class_sample_count = np.unique(train_sample_labels,
                                    return_counts=True)[1]
        print(f'Initial imbalanced dataset (samples/class):\n{class_sample_count}')

        # Weight per sample not per class.
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in train_sample_labels])

        # Casting.
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()

        # Sampler, imbalanced data.
        sampler = torch.utils.data.WeightedRandomSampler(
            samples_weight,
            len(samples_weight)
        )
        shuffle = False

    #--------------------------
    # Creating a reduced subset (option).
    #--------------------------
    if args.reduced_dataset:

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
        sampler = torch.utils.data.SubsetRandomSampler(keep_indices)
        shuffle = False

    #--------------------------
    # If distributed (option).
    #--------------------------
    if args.distributed:
        sampler = DistributedSampler(
            dataset['train'],
            shuffle=True,
            seed=args.seed,
            drop_last=True
        )
        shuffle=False

    # Configure correct batch size.
    world_size = int(os.environ['WORLD_SIZE'])
    bsz = int(args.batch_size / world_size)
    print(f'\nNew batch size considering a word size of {world_size} GPUs: {bsz}')

    #--------------------------
    # Cast to Lightly dataset.
    #--------------------------
    # Builds a LightlyDataset from a PyTorch (or torchvision) dataset.
    # Returns a tuple (sample, target, fname) when accessed using __getitem__.
    lightly_dataset = {x: lightly.data.LightlyDataset.from_torch_dataset(
        dataset[x]) for x in splits}

    if args.verbose:
        for d in lightly_dataset:
            print(f'\n{d}: {lightly_dataset[d]}')

    #--------------------------
    # Collate functions.
    #--------------------------
    # Base class for other collate implementations.
    collate_fn = {x: lightly.data.collate.BaseCollateFunction(
        transform[x]) for x in splits}

    if args.verbose:
        for c in collate_fn:
            print(f'\n{c}: {collate_fn[c]}')

    #--------------------------
    # PyTorch dataloaders.
    #--------------------------
    # Dataloader for validating and testing.
    dataloader = {x: torch.utils.data.DataLoader(
        dataset[x],                                     # lightly_dataset[x]
        batch_size=bsz,
        shuffle=True,                                   # False
        num_workers=args.num_workers,
        # collate_fn=collate_fn[x],
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker if not args.ray_tune else None,
        generator=g if not args.ray_tune else None
    ) for x in splits[1:]}

    # Dataloader for training.
    dataloader['train'] = torch.utils.data.DataLoader(
        lightly_dataset['train'],
        batch_size=bsz,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn['train'],
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker if not args.ray_tune else None,
        generator=g if not args.ray_tune else None
    )

    print(f'Sampler: {sampler}')
    if args.verbose:
        for d in dataloader:
            print(f'\n{d}: {vars(dataloader[d])}')

    #--------------------------
    # Check the balance and size of the dataset.
    #--------------------------
    # Check samples per class, total samples and batches of each dataset.
    if args.verbose:
        for d in dataset:
            samples = np.unique(dataset[d].targets, return_counts=True)[1]
            print(f'\n{d}:')
            # print(f'  - #Samples (from dataset):  {len(dataset[d].targets)}')
            print(f'  - #Samples/class (from dataset):\n{samples}')
            print(f'  - #Batches (from dataloader): {len(dataloader[d])}')
            print(f'  - #Samples (from dataloader): {len(dataloader[d])*bsz}')

    #--------------------------
    # Check the distribution of samples in the dataloader (lightly dataset).
    #--------------------------

    # if not args.distributed:

    #     # List to save the labels.
    #     print('\nCreating the sample distribution plot...')
    #     labels_list = []

    #     # Accessing Data and Targets in a PyTorch DataLoader.
    #     t0 = time.time()
    #     for i, (images, labels, names) in enumerate(dataloader['train']):
    #         labels_list.append(labels)

    #     # Concatenate list of lists (batches).
    #     labels_list = torch.cat(labels_list, dim=0).numpy()
    #     print(f'\nSample distribution computation in train dataset (s): '
    #         f'{(time.time()-t0):.2f}')

    #     # Count number of unique values.
    #     data_x, data_y = np.unique(labels_list, return_counts=True)

    #     # New function to plot (suitable for execution in shell).
    #     fig, ax = plt.subplots(1, 1, figsize=(20, 5))
    #     simple_bar_plot(ax,
    #                     data_x,
    #                     'Class',
    #                     data_y,
    #                     'N samples (dataloader)')

    #     plt.gcf().subplots_adjust(bottom=0.15)
    #     plt.gcf().subplots_adjust(left=0.15)
    #     fig_name_save = (f'sample_distribution'
    #                     f'-ratio={args.dataset_ratio}'
    #                     f'-balanced_dataset={args.balanced_dataset}'
    #                     f'-reduced_dataset={args.reduced_dataset}')
    #     fig.savefig(os.path.join(paths['images'], fig_name_save+FIG_FORMAT),
    #                 bbox_inches='tight')

    #     plt.show() if args.show else plt.close()
    #     print('Done!')

    #--------------------------
    # Look at some training samples (lightly dataset).
    #--------------------------
    # Accessing Data and Targets in a PyTorch DataLoader.
    if args.show:
        for i, (images, labels, names) in enumerate(dataloader['train']):
            img = images[0][0]
            label = labels[0]
            name = names[0]
            img = inv_norm_tensor(  # Revert normalization.
                img,
                mean=mean['train'],
                std=std['train']
            )
            if args.verbose:
                print(images[0].shape)
                print(labels.shape)
                print(name)
            plt.title("Label: " + str(int(label)))
            plt.imshow(torch.permute(img, (1, 2, 0)))
            plt.show()
            if i == 0:  # Only a few batches.
                break

    # Show the images within the first batch.
    if args.show:
        for b, ((x0, x1), _, _) in enumerate(dataloader['train']):
            show_two_batches(x0, x1, 1, mean=mean, std=std)
            if b == 1:  # Only a few batches.
                break

    # ======================
    # SELF-SUPERVISED MODELS.
    # ======================

    #--------------------------
    # Training (also with hyperparameter tuning using Ray Tune)
    #--------------------------
    if args.ray_tune:

        cpus_per_trial = args.num_workers
        print(f'\nMax. number of epochs:    {args.epochs}')
        print(f'Number of samples:        {args.num_samples_trials}')
        print(f'Number of CPUs per trial: {cpus_per_trial}')
        print(f'Number of GPUS per trial: {args.gpus_per_trial}')

        # Configuration.
        if args.ray_tune == 'loguniform':

            # Build the filename.
            filename = f'ray_tune_{args.backbone_name}_{args.model_name}.csv'

            # Load the CSV file into a pandas dataframe.
            df = pd.read_csv(os.path.join(paths['ray_tune'], filename),
                             usecols=lambda col: col.startswith('loss')
                             or col.startswith('config/'))

            # Configuration.
            print(f'Setting the configuration from {filename} and tuning the lr')
            config = {
                'args': args,
                'input_size': input_size,
                'dataloader': dataloader,
                'bsz': bsz,
                'paths': paths,
                'hidden_dim': df.loc[0, 'config/hidden_dim'],
                'out_dim': df.loc[0, 'config/out_dim'],
                'lr': tune.loguniform(1e-4, 1e-1),
                'momentum': df.loc[0, 'config/momentum'],
                'weight_decay': df.loc[0, 'config/weight_decay'],
                'warmup_epochs': 1,
            }

        elif args.ray_tune == 'gridsearch':

            print(f'Setting a new configuration using tune.grid_search')
            config = {
                'args': args,
                'input_size': input_size,
                'dataloader': dataloader,
                'bsz': bsz,
                'paths': paths,
                'hidden_dim': tune.grid_search([128, 256, 512]),
                'out_dim': tune.grid_search([128, 256, 512]),
                'lr': tune.grid_search([1e-4, 1e-3, 1e-2, 1e-1]),
                # 'momentum': 0.9,
                'momentum': tune.grid_search([0.99, 0.9]),          # 0.97, 0.95
                # 'weight_decay': 0,
                'weight_decay': tune.grid_search([0, 1e-4, 1e-5]),  # 1e-3
                'warmup_epochs': 1,
            }

        # Ray tune configuration.
        scheduler = ASHAScheduler(
            metric='loss',
            mode='min',
            max_t=args.epochs,  # max_num_epochs
            grace_period=args.grace_period
        )

        reporter = CLIReporter(
            # ``parameter_columns=["l1", "l2", "lr", "batch_size"]``,
            # metric_columns=["loss", "accuracy", "training_iteration"])
            metric_columns=['loss', 'training_iteration']
        )

        result = tune.run(
            partial(train),
            resources_per_trial={'cpu': cpus_per_trial, 'gpu': args.gpus_per_trial},
            name=args.model_name,
            config=config,
            num_samples=args.num_samples_trials,
            local_dir=paths['ray_tune'],
            scheduler=scheduler,
            verbose=1,
            progress_reporter=reporter
        )

        # Sorted dataframe for the last reported results of all of the trials.
        df = result.results_df
        df = df.sort_values(by=['loss'], ascending=True)

        # Create the name of the file.
        if args.ray_tune == 'loguniform':
            filename = f'ray_tune_lr_{args.backbone_name}_{args.model_name}.csv'
        elif args.ray_tune == 'gridsearch':
            filename = f'ray_tune_{args.backbone_name}_{args.model_name}.csv'

        # Write the results to a CSV file.
        df.to_csv(os.path.join(paths['ray_tune'], filename))

        # Get best results.
        best_trial = result.get_best_trial("loss", "min", "last")
        print(f"Best trial config: {best_trial.config}")
        print(f"Best trial final val loss: {best_trial.last_result['loss']}")

    else:

        # Set warm-up epochs.
        if args.epochs < 100:
            warmup_epochs = max(1, int(0.1 * args.epochs))
        else:
            warmup_epochs = 10

        # Build the filename.
        # filename_lr = f'ray_tune_lr_{args.backbone_name}_{args.model_name}.csv'
        filename_lr = f'ray_tune_{args.backbone_name}_{args.model_name}.csv'

        # Load the CSV file into a pandas dataframe.
        # df_lr = pd.read_csv(os.path.join(paths['best_configs'], filename_lr),
        #                     usecols=lambda col: col.startswith('loss')
        #                     or col.startswith('config/'))
        df_lr = pd.read_csv(os.path.join(paths['best_configs'], filename_lr))

        # Configuration.
        print(f'\nSetting the best configuration for the model from file: {filename_lr}')
        config = {
            'args': args,
            'input_size': input_size,
            'dataloader': dataloader,
            'bsz': bsz,
            'paths': paths,
            'hidden_dim': df_lr.loc[0, 'hidden_dim'],    # [0, 'config/hidden_dim'],
            'out_dim': df_lr.loc[0, 'out_dim'],
            'lr': df_lr.loc[0, 'lr'],
            'momentum': df_lr.loc[0, 'momentum'],
            'weight_decay': df_lr.loc[0, 'weight_decay'],
            'warmup_epochs': warmup_epochs,
        }

        # Launch training.
        train(config)

    destroy_process_group()

    return 0


if __name__ == "__main__":

    # Get arguments.
    args = get_args()

    # Main function.
    sys.exit(main(args))
