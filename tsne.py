import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import sys
import torch
import torchvision
from utils import (barlowtwins, mocov2, simclr, simsiam)
import pandas as pd
from datetime import datetime
from sklearn.manifold import TSNE
from torchvision import transforms
from utils.other import build_paths
from utils.dataset import (
    AndaluciaDataset,
    load_mean_std_values
)
from utils.check_embeddings import create_list_embeddings
from utils.dataset import load_dataset_based_on_ratio
from utils.reproducibility import set_seed, seed_worker


def get_args() -> argparse.Namespace:
    """
    Parse and retrieve command-line arguments.

    Returns:
        An 'argparse.Namespace' object containing the parsed arguments.
    """

    # Get arguments.
    parser = argparse.ArgumentParser(
        description='Script that merges the .csv files.'
    )

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

    parser.add_argument('--batch_size', '-bs', type=int, default=64,
                        help='number of images in a batch during training '
                             '(default: 64).')

    parser.add_argument('--num_workers', '-nw', type=int, default=2,
                        help='number of subprocesses to use for data loading. '
                             '0 means that the data will be loaded in the '
                             'main process (default: 2).')

    parser.add_argument('--seed', '-s', type=int, default=42,
                        help='seed for the experiments (default: 42).')

    parser.add_argument('--verbose', '-v', action='store_true',
                        help='provides additional details for debugging purposes.')

    parser.add_argument('--save_fig', '-sf', type=str, choices=['png', 'pdf'],
                        help='format of the output image (default: png).')

    return parser.parse_args(sys.argv[1:])


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

    # ======================
    # INITIAL CONFIGURATION.
    # ======================
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
    paths = build_paths(cwd, 'embeddings')
    if args.input_data:
        paths['datasets'] = args.input_data

    # Show built paths.
    if args.verbose:
        for path in paths:
            path_name_col = f'{path}:'
            print(f'{path_name_col.ljust(20)} {paths[path]}')

    # Size of the images.
    input_size = 224

    # args.dataset_level = 'Level_N2'
    # args.train_rate = 1.0
    # args.task_name = 'multiclass'

    # ======================
    # DATASET.
    # ======================
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

    # mean, std = load_mean_std_values(
    #     os.path.join(
    #         paths['datasets'],
    #         os.path.join(args.dataset_name, args.dataset_level)
    #     )
    # )

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
    ]) for x in splits}

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
    ) for x in splits}

    # Load the Andalucia dataset with normalization.
    # dataset = {x: AndaluciaDataset(
    #     root_dir=os.path.join(paths['datasets'], args.dataset_name),
    #     level='Level_N2',
    #     split=x,
    #     train_ratio=args.train_rate,
    #     transform=transform[x],
    #     target_transform=transform_abundances if args.task_name == 'multiclass' else None,
    #     seed=args.seed,
    #     verbose=args.verbose
    # ) for x in splits}

    if args.verbose:
        for d in dataset:
            print(f'\n{d}: {dataset[d]}')
            print(f'\n{d}: {dataset[d].class_to_idx}')

    #--------------------------
    # PyTorch dataloaders.
    #--------------------------
    # Dataloader for validating and testing.
    dataloader = {x: torch.utils.data.DataLoader(
        dataset[x],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g
    ) for x in splits}

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
            print(f'  - #Samples (from dataloader): {len(dataloader[d])*args.batch_size}')

    # ======================
    # MODELS.
    # ======================

    args.model_name = 'BarlowTwins'
    args.backbone_name = 'resnet18'
    weights = args.model_name

    # Load snapshot from pretraining.
    snapshot_name = f'snapshot_{args.model_name}_{args.backbone_name}_bd=False_iw=random.pt'
    snapshot = torch.load(os.path.join(paths['input'], snapshot_name))
    print(f'Model loaded from {snapshot_name}')

    # Removing head from resnet: Encoder.
    resnet = torchvision.models.resnet18(weights=None)              
    backbone = torch.nn.Sequential(*list(resnet.children())[:-1])
    input_dim = resnet.fc.in_features

    # Build the filename.
    filename_lr = f'ray_tune_{args.backbone_name}_{args.model_name}.csv'
    df_lr = pd.read_csv(os.path.join(paths['best_configs'], filename_lr))
    hidden_dim = df_lr.loc[0, 'hidden_dim']
    out_dim = df_lr.loc[0, 'out_dim']

    print(f"{'Model name:'.ljust(18)} {args.model_name}")
    print(f"{'Backbone name:'.ljust(18)} {args.backbone_name}")
    print(f"{'Hidden layer dim.:'.ljust(18)} {hidden_dim}")
    print(f"{'Output layer dim.:'.ljust(18)} {out_dim}")

    if args.model_name == 'SimSiam':
        model = simsiam.SimSiam(backbone=backbone, input_dim=input_dim, proj_hidden_dim=out_dim,
                                pred_hidden_dim=hidden_dim, output_dim=out_dim)
    elif args.model_name == 'SimCLR':
        model = simclr.SimCLR(backbone=backbone, input_dim=input_dim,
                              hidden_dim=hidden_dim, output_dim=out_dim,
                              num_layers=2, memory_bank_size=0)
    elif args.model_name == 'SimCLRv2':
        model = simclr.SimCLR(backbone=backbone, input_dim=input_dim,
                              hidden_dim=hidden_dim, output_dim=out_dim,
                              num_layers=3, memory_bank_size=65536)
    elif args.model_name == 'BarlowTwins':
        model = barlowtwins.BarlowTwins(backbone=backbone, input_dim=input_dim,
                                        hidden_dim=hidden_dim, output_dim=out_dim)
    elif args.model_name == 'MoCov2':
        model = mocov2.MoCov2(backbone=backbone, input_dim=input_dim,
                              hidden_dim=hidden_dim, output_dim=out_dim)

    model.load_state_dict(snapshot["MODEL"])

    model = torch.nn.Sequential(
        model.backbone,
        torch.nn.Flatten(),
        )

    # Create ResNet18 backbone with random weights
    # weights = torchvision.models.ResNet18_Weights.DEFAULT           # weights=torchvision.models.ResNet18_Weights.DEFAULT
    # model = torchvision.models.resnet18(weights=weights)              
    # model.fc = torch.nn.Identity()                                  # Remove the fully connected layer

    # Send the model to GPU.
    model = model.to('cuda')

    # ======================
    # EMBEDDINGS.
    # ======================
    # Iterate over the samples and save the embeddings.
    print('\nGenerating and saving the embeddings...')
    # embeddings, labels = create_list_embeddings(model, dataloader, 'cuda', distributed=False)
    embeddings = []
    labels = []
    model.eval()
    with torch.no_grad():
        for images, targets in dataloader['test']:
            images = images.to('cuda')
            outputs = model(images)
            embeddings.append(outputs.to('cpu').numpy())
            labels.extend(targets.numpy())

    embeddings = np.concatenate(embeddings)
    labels = np.array(labels)
    # np.savetxt(os.path.join(paths['images'], "embeddings.csv"), embeddings, delimiter=",")
    # np.savetxt(os.path.join(paths['images'], "labels.csv"), labels, delimiter=",")
    print('Completed!')

    # Perform t-SNE on the embeddings.
    print('\nComputing t-SNE...')
    tsne = TSNE(n_components=2, verbose=1, random_state=args.seed)
    embeddings_tsne = tsne.fit_transform(embeddings)
    print('Completed!')

    # Plot the embeddings in 2D
    num_classes = len(np.unique(labels))
    fig = plt.figure(figsize=(10, 10))
    sns.set(font_scale=1.6)
    gfg = sns.scatterplot(
        x=embeddings_tsne[:, 0],
        y=embeddings_tsne[:, 1],
        hue=labels,
        palette=sns.color_palette('hls', num_classes),
        legend='full',
        alpha=0.7
    )                                                                       # .set_title(f't-SNE_{weights}')
    gfg.set_ylim(-110, 110)
    gfg.set_xlim(-110, 124)
    plt.legend(loc='right', fontsize='12', title_fontsize='12')

    # Save figure or show.
    if args.save_fig:
        save_path = os.path.join(
            paths['images'],
            f't-SNE_iw={weights}_s={args.seed}.{args.save_fig}'             # -{datetime.now():%Y_%m_%d-%H_%M_%S}
        )
        fig.savefig(save_path, bbox_inches='tight')
        print(f'\nFigure saved at {save_path}')
    else:
        plt.title(f't-SNE_{weights}')
        plt.show()

    return 0


if __name__ == "__main__":

    # Get arguments.
    args = get_args()

    # Main function.
    sys.exit(main(args))
