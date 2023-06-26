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
from utils.graphs import simple_bar_plot
from trainer import Trainer

# Arguments and paths.
import os
import sys
import argparse

# PyTorch.
import torch
import torchvision
from torchvision import transforms
from torchinfo import summary
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
    resnet50,
    ResNet50_Weights
)

# PyTorch TensorBoard support.
from torch.utils.tensorboard import SummaryWriter
import csv

# Data management.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Performance metrics.
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time

# PyTorch DDP.
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# Hyperparameter tunning.
from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

AVAIL_SSL_MODELS = ['BarlowTwins', 'MoCov2', 'SimCLR', 'SimCLRv2', 'SimSiam']
MODEL_CHOICES = ['Random', 'Supervised'] + AVAIL_SSL_MODELS
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

    parser.add_argument('--train_rate', '-tr', type=float, default=1.,
                        help='dataset ratio for train subset (default=1.).')

    parser.add_argument('--epochs', '-e', type=int, default=25,
                        help='number of epochs for training (default: 25).')

    parser.add_argument('--learning_rate', '-lr', type=float, default=0.01,
                        help='learning rate for fine-tuning (default: 0.01).')

    parser.add_argument('--save_every', '-se', type=int, default=5,
                        help='save model checkpoint every n epochs (default: 5).')

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

    parser.add_argument('--dropout', '-do', type=float,
                        help='adds a dropout layer before the linear classifier '
                             'with the given probability.')

    parser.add_argument('--transfer_learning', '-tl', type=str, required=True,
                        choices=['LP', 'FT', 'LP+FT'],
                        help='sets the main transfer learning algorithm to use.')

    parser.add_argument('--show', '-sw', action='store_true',
                        help='the images pops up.')

    parser.add_argument('--verbose', '-v', action='store_true',
                        help='provides additional details for debugging purposes.')

    parser.add_argument('--balanced_dataset', '-bd', action='store_true',
                        help='whether the dataset should be balanced.')

    parser.add_argument('--torch_compile', '-tc', action='store_true',
                        help='PyTorch 2.0 compile enabled.')

    # Create a mutually exclusive group.
    group = parser.add_mutually_exclusive_group()

    group.add_argument('--distributed', '-d', action='store_true',
                       help='enables distributed training.')

    group.add_argument('--ray_tune', '-rt', type=str,
                        choices=['gridsearch', 'loguniform'],
                        help='enables Ray Tune (tunes everything or only lr).')

    # Specific for Ray Tune.
    parser.add_argument('--grace_period', '-rtgp', type=int,
                        help='only stop trials at least this old in time.')

    parser.add_argument('--num_samples_trials', '-rtnst', type=int,
                        help='number of samples to tune the hyperparameters.')

    parser.add_argument('--gpus_per_trial', '-rtgpt', type=int,
                        help='number of gpus to be used per trial.')

    return parser.parse_args(sys.argv[1:])


def ddp_setup() -> None:
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
    paths = build_paths(cwd, 'finetuning')
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
    mean, std = load_mean_std_values(
        os.path.join(
            paths['datasets'],
            os.path.join(args.dataset_name, args.dataset_level)
        )
    )

    # # Rename the key 'val' to 'validation'.
    # mean['validation'] = mean.pop('val')
    # std['validation'] = std.pop('val')

    if args.verbose:
        print(f'\nMean: {mean}')
        print(f'Std:  {std}')

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
        train_ratio=args.train_rate,
        transform=transform[x],
        target_transform=transform_abundances if args.task_name == 'multiclass' else None,
        seed=args.seed,
        verbose=args.verbose
    ) for x in splits}


    #--------------------------
    # Dealing with imbalanced data (option).
    #--------------------------

    if args.task_name == 'multiclass':

        # Creating a list of labels of samples.
        train_sample_labels = andalucia_dataset['train'].targets

        # Calculating the number of samples per label/class.
        class_and_sample_counts = np.unique(train_sample_labels,
                                            return_counts=True)
        class_count = class_and_sample_counts[0]
        sample_count_per_class = class_and_sample_counts[1]
        print('Initial imbalanced dataset:')
        print(f'Diff. classes --> {class_count}')
        print(f'Samples/class --> {sample_count_per_class}')

        # Weight per sample not per class.
        weight = 1. / sample_count_per_class
        index_map = {value: index for index, value in enumerate(class_count)}  # Map, e.g., 0--> 0, 21 --> 1, etc.
        samples_weight = np.array([weight[index_map[t]] for t in train_sample_labels])

        # Casting.
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()

        # Sampler, imbalanced data.
        sampler = torch.utils.data.WeightedRandomSampler(
            samples_weight,
            len(samples_weight)
        )
        shuffle = False
        print('Using balanced dataloader as default option!')

    #--------------------------
    # If distributed (option).
    #--------------------------
    if args.distributed:
        sampler=DistributedSampler(andalucia_dataset['train'])
        shuffle=False

    #--------------------------
    # PyTorch dataloaders.
    #--------------------------
    # Dataloader for validating and testing.
    dataloader = {x: torch.utils.data.DataLoader(
        andalucia_dataset[x],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker if not args.ray_tune else None,
        generator=g if not args.ray_tune else None
    ) for x in splits[1:]}

    # Dataloader for training.
    dataloader['train'] = torch.utils.data.DataLoader(
        andalucia_dataset['train'],
        batch_size=args.batch_size,
        shuffle=shuffle,                      # Careful.
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker if not args.ray_tune else None,
        generator=g if not args.ray_tune else None
    )

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
        samples = np.unique(andalucia_dataset[d].targets, return_counts=True)[1]
        print(f'\n{d}:')
        print(f'  - #Samples (from dataset):  {len(andalucia_dataset[d].targets)}')
        print(f'  - #Samples/class (from dataset):\n{samples}')
        print(f'  - #Batches (from dataloader): {len(dataloader[d])}')
        print(f'  - #Samples (from dataloader): {len(dataloader[d])*args.batch_size}')

    #--------------------------
    # Check the distribution of samples in the dataloader (lightly dataset).
    #--------------------------

    if args.task_name == 'multiclass':

        # List to save the labels.
        print('\nCreating the sample distribution plot...')
        labels_list = []

        # Accessing Data and Targets in a PyTorch DataLoader.
        t0 = time.time()
        for _, labels in dataloader['train']:
            labels_list.append(labels)

        # Concatenate list of lists (batches).
        labels_list = torch.cat(labels_list, dim=0).numpy()
        print(f'Sample distribution computation in train dataset (s): '
              f'{(time.time()-t0):.2f}')

        # Calculating the number of samples per label/class.
        class_and_sample_counts = np.unique(labels_list,
                                            return_counts=True)
        class_count = class_and_sample_counts[0]
        sample_count_per_class = class_and_sample_counts[1]
        print('Resulting balanced dataloader:')
        print(f'Diff. classes     --> {class_count}')
        print(f'New samples/class --> {sample_count_per_class}')

        # New function to plot (suitable for execution in shell).
        fig, ax = plt.subplots(1, 1, figsize=(20, 5))
        simple_bar_plot(ax,
                        class_count,
                        'Class',
                        sample_count_per_class,
                        'N samples (dataloader)')

        plt.gcf().subplots_adjust(bottom=0.15)
        plt.gcf().subplots_adjust(left=0.15)
        fig_name_save = (f'sample_distribution'
                        f'-train_ratio={args.train_rate}')
        fig.savefig(os.path.join(paths['images'], fig_name_save+FIG_FORMAT),
                    bbox_inches='tight')

        plt.show() if args.show else plt.close()
        print('Done!')

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
    # Setting the model and initial weights.
    if args.backbone_name == 'resnet18':
        if args.ini_weights == 'imagenet':
            resnet = resnet18(
                weights=ResNet18_Weights.DEFAULT,
                # zero_init_residual=True
            )
            print('Using ImageNet weights')
        else:
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
            print('Using ImageNet weights')
        else:
            resnet = resnet50(
                weights=None,
                # zero_init_residual=True
            )

    # Model: random and supervised resnet.
    # if args.model_name == 'Random' or 
    if args.model_name == 'Supervised':
        print(f'\n{args.model_name} model {args.backbone_name} with {args.ini_weights} weights')
        model = resnet

        # Get the number of input features to the layer.
        print(f'Old final fully-connected layer: {model.fc}')
        num_ftrs = model.fc.in_features

        # Create the new final fully-connected layer.
        final_fc = torch.nn.Linear(num_ftrs, len(class_names))

        # Check if the dropout argument is passed and create the modified model accordingly
        if args.dropout:
            model.fc = torch.nn.Sequential(
                torch.nn.Dropout(p=args.dropout, inplace=True),
                final_fc
            )
            print(f'Dropout layer added: {model.fc}')
        else:
            model.fc = final_fc
            print('No dropout layer')

        print(f'New final fully-connected layer: {model.fc}')

        # Parameters of newly constructed modules
        # have requires_grad=True by default.
        # Freezing all the network if chosen.
        if args.transfer_learning == 'FT':          # Fine-tuning (FT).
            for param in model.parameters():
                param.requires_grad = True
            for param in model.fc.parameters():
                param.requires_grad = True
            print('Fine-tuning adjusted')
        else:                                       # Linear probing (LP) / LP+FT.
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
            print('Linear probing adjusted')

    # Model: resnet with pretrained weights (SSL).
    elif args.model_name in AVAIL_SSL_MODELS:
        print(f'\nModel {args.backbone_name} with pretrained weights using {args.model_name} SSL')

        # Load snapshot from pretraining.
        snapshot_name = f'snapshot_pt_{args.model_name}_{args.backbone_name}_balanced={args.balanced_dataset}_weights={args.ini_weights}.pt'
        snapshot = torch.load(os.path.join(paths['input'], snapshot_name))
        print(f'Model loaded from {snapshot_name}')

        # Removing head from resnet: Encoder.
        backbone = torch.nn.Sequential(*list(resnet.children())[:-1])
        input_dim = resnet.fc.in_features

        # Build the filename.
        # filename_lr = f'ray_tune_results_lr_{args.backbone_name}_{args.model_name}.csv'
        filename_lr = f'ray_tune_{args.backbone_name}_{args.model_name}.csv'

        # Load the CSV file into a pandas dataframe.
        # df_lr = pd.read_csv(os.path.join(paths['best_configs'], filename_lr),
        #                     usecols=lambda col: col.startswith('loss')
        #                     or col.startswith('config/'))
        df_lr = pd.read_csv(os.path.join(paths['best_configs'], filename_lr))
    
        hidden_dim = df_lr.loc[0, 'hidden_dim']          # [0, 'config/hidden_dim']
        out_dim = df_lr.loc[0, 'out_dim']

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

        # Define your model.
        model = torch.nn.Sequential(
            model.backbone,
            torch.nn.Flatten(),
        )

        # Add dropout layer if the dropout argument is passed.
        if args.dropout:
            model.add_module('dropout', torch.nn.Dropout(p=args.dropout, inplace=True))
            print(f'Dropout layer added: {model.dropout}')
        else:
            print('No dropout layer')

        # Add the final linear layer
        model.add_module('linear',
                         torch.nn.Linear(in_features=input_dim,
                                         out_features=len(class_names),
                                         bias=True))

        # Get the number of input features to the layer.
        # Adjust the final layer to the current number of classes.
        # print(f'\nOld final fully-connected layer: {model[-1]}')
        # num_ftrs = model[-1].in_features
        # model[-1] = torch.nn.Linear(num_ftrs, len(class_names))
        print(f'New final fully-connected layer: {model[-1]}')

        # Parameters of newly constructed modules
        # have requires_grad=True by default.
        # Freezing all the network if chosen.
        if args.transfer_learning == 'FT':          # Fine-tuning (FT).
            for param in model.parameters():
                param.requires_grad = True
            for param in model[-1].parameters():
                param.requires_grad = True
            print('Fine-tuning adjusted')
        else:                                       # Linear probing (LP) / LP+FT.
            for param in model.parameters():
                param.requires_grad = False
            for param in model[-1].parameters():
                param.requires_grad = True
            print('Linear probing adjusted')

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
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=0)
    if args.verbose:
        print(f'Optimizer:\n{optimizer}')

    # Training.
    general_name = f'{args.task_name}_tr={args.train_rate:.3f}_{args.backbone_name}_{args.model_name}_tl={args.transfer_learning}_lr={args.learning_rate}_bd={args.balanced_dataset}_iw={args.ini_weights}_do={args.dropout}'
    trainer = Trainer(
        model,
        dataloader,
        loss_fn,
        optimizer,
        save_every=args.save_every,
        snapshot_path=os.path.join(paths['snapshots'], f'snapshot_{general_name}.pt'),
        csv_path=os.path.join(paths['csv_results'], f'{general_name}.csv'),
        distributed=args.distributed,
        lightly_train=False,
        ray_tune = args.ray_tune,
        ignore_ckpts=False
    )

    if not args.ray_tune:

        print(f'\nNormal training (DDP set to {args.distributed})')

        config = {
            'args': args,
            'test': True,
            'save_csv': True
        }

        trainer.train(config) 
    
    else:

        print(f'\nSetting a new configuration using tune.grid_search\n')

        config = {
            'args': args,
            'test': True,
            'save_csv': True,
            'input_size': input_size,
            'dataloader': dataloader,
            'lr': tune.grid_search([1e-4, 1e-3, 1e-2, 1e-1]),
            'momentum': 0.9,
            'weight_decay': 0
        }

        # Ray tune configuration.
        scheduler = ASHAScheduler(
            metric='loss',
            mode='min',
            max_t=args.epochs,                  # max_num_epochs
            grace_period=args.grace_period
        )

        reporter = CLIReporter(
            # ``parameter_columns=["l1", "l2", "lr", "batch_size"]``,
            # metric_columns=["loss", "accuracy", "training_iteration"])
            metric_columns=['loss', 'training_iteration']
        )

        result = tune.run(
            partial(trainer.train),
            resources_per_trial={'cpu': args.num_workers, 'gpu': args.gpus_per_trial},
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

        # Create the name of the file and write the results to a CSV file.
        filename = f'ray_tune_{general_name}.csv'
        df.to_csv(os.path.join(paths['ray_tune'], filename))

        # Print.
        best_trial = result.get_best_trial("loss", "min", "last")
        print(f"Best trial config: {best_trial.config}")
        print(f"Best trial final val loss: {best_trial.last_result['loss']}")

    return 0


if __name__ == "__main__":

    # Get arguments.
    args = get_args()

    # Main function.
    sys.exit(main(args))
