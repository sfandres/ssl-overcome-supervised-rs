"""Creates the final figures from the csv files that contain
   the mean and std values of the merged dataframes.

Usage: plot_final_graphs.py [-h] --input INPUT [--output OUTPUT] [--save_fig {png,pdf}] [--verbose]

Script that creates the figures for the paper.

options:
  -h, --help            show this help message and exit
  --input INPUT, -i INPUT
                        path to the parent folder where the different train ratio folders are stored.
  --output OUTPUT, -o OUTPUT
                        path to the folder where the figure will be saved.
  --save_fig {png,pdf}, -sf {png,pdf}
                        format of the output image (default: png).
  --verbose, -v         provides additional details for debugging purposes.

Author:
    A.J. Sanchez-Fernandez - 10/07/2023
"""


import pandas as pd
import numpy as np
import argparse
import sys
import os
from datetime import datetime
from matplotlib import pyplot as plt

def set_plt() -> None:
    """
    Configure matplotlib figures.
    """

    SMALL_SIZE = 15
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 20

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def get_args() -> argparse.Namespace:
    """
    Parse and retrieve command-line arguments.

    Returns:
        An 'argparse.Namespace' object containing the parsed arguments.
    """

    # Get arguments.
    parser = argparse.ArgumentParser(
        description='Script that creates the figures for the paper.'
    )

    parser.add_argument('--input', '-i', required=True,    # nargs='+',
                        help='path to the parent folder where the different train ratio folders are stored.')

    parser.add_argument('--output', '-o', default='./',
                        help='path to the folder where the figure will be saved.')

    parser.add_argument('--metric', '-m', required=True,
                        choices=['top1', 'f1_micro', 'top5', 'f1_macro', 'f1_weighted', 'rmse', 'mae'],
                        help='parameter to be displayed in the y-axis.')

    parser.add_argument('--save_fig', '-sf', type=str, choices=['png', 'pdf'],
                        help='format of the output image (default: png).')

    parser.add_argument('--verbose', '-v', action='store_true',
                        help='provides additional details for debugging purposes.')

    return parser.parse_args(sys.argv[1:])


def main(args):

    # Configure matplotlib.
    set_plt()

    # Print target folders.
    if args.verbose:
        print(f"\n---------------------------------------------------") 
        print(f"{'Input folder:'.ljust(16)}{args.input}")
        print(f"{'Output folder:'.ljust(16)}{args.output}")

    # Get task from first item and set target metric.
    task = args.input.split('/')[-2]
    metric = args.metric
    if task == 'multiclass':
        loc = 'lower right'
        y_lim = 0.8
        text_space = 0.05
    elif task == 'multilabel':
        loc = 'upper right'
        y_lim = 0.2
        text_space = 0.01
    else:
        loc = None
    print(f"{'Task:'.ljust(16)}{task}") if args.verbose else None

    # Get a list of all directories.
    x = [1, 5, 10, 25, 50, 100]
    dirs = os.listdir(args.input)
    filtered_dirs = sorted([d for d in dirs if 'p' in d])
    if args.verbose:
        print(f"{'Target ratios:'.ljust(16)}{x}")
        print(f"{'Target dirs:'.ljust(16)}{filtered_dirs}")
        print(f"{'Target metric:'.ljust(16)}{metric}")

    # Set the transfer learning algorithms.
    transfer_learning_algs = ['_tl=LP_', '_tl=FT_', '_tl=LP+FT_']
    print(f"{'TL algorithms:'.ljust(16)}{transfer_learning_algs}") if args.verbose else None

    # Set the models.
    models = ['Barlow Twins', 'ImageNet', 'Random']
    print(f"{'Models:'.ljust(16)}{models}") if args.verbose else None

    # Iterate over the algorithms.
    for transfer in transfer_learning_algs:

        # Show information.
        if args.verbose:
            print(f"\n---------------------------------------------------") 
            print(f"{'Curr TL:'.ljust(13)}{transfer}")

        # Create fig.
        fig = plt.figure(figsize=(10, 6))

        # Iterate over the models.
        for model in models:

            # Create the last filters according to the target model.
            if model == 'Barlow Twins':
                filter1 = 'BarlowTwins'
                filter2 = '_iw=random'
            elif model == 'ImageNet':
                filter1 = 'Supervised'
                filter2 = '_iw=imagenet'
            elif model == 'Random':
                filter1 = 'Supervised'
                filter2 = '_iw=random'
            mean_files, std_files = [], []
            mean_values, std_values = [], []
            print(f"\n{'Curr model:'.ljust(13)}{model}") if args.verbose else None

            # Iterate over the dirs.
            for ratio in filtered_dirs:

                # Build current path.
                curr_path = os.path.join(args.input, ratio)
                # print(f"\n{'* Path:'.ljust(9)}{curr_path}") if args.verbose else None

                # Get a list of all directories.
                files = os.listdir(curr_path)
                filtered_files = sorted([f for f in files if transfer in f])

                # Filter current files.
                curr_mean_file = [f for f in filtered_files
                             if ('pp_mean_' in f and filter1 in f and filter2 in f)][0]
                curr_std_file = [f for f in filtered_files
                            if ('pp_std_' in f and filter1 in f and filter2 in f)][0]

                # Append filenames.
                mean_files.append(os.path.join(curr_path, curr_mean_file))
                std_files.append(os.path.join(curr_path, curr_std_file))

                # Append mean and std values.
                res_mean_last_epoch = pd.read_csv(os.path.join(curr_path, curr_mean_file)).iloc[-1, :][metric]
                mean_values.append(res_mean_last_epoch)
                res_std_last_epoch = pd.read_csv(os.path.join(curr_path, curr_std_file)).iloc[-1, :][metric]
                std_values.append(res_std_last_epoch)

            # Show information.
            if args.verbose:
                print('Target files:')
                for mfile, sfile, mvalue, svalue in zip(mean_files, std_files, mean_values, std_values):
                    print(mfile, '-->', mvalue)
                    print(sfile, '-->', svalue)

            # Plot the current model's values.
            y = np.array(mean_values)
            lower_y = y - np.array(std_values)
            upper_y = y + np.array(std_values)
            plt.plot(x, y, 'x-', label=model)
            plt.fill_between(x, lower_y, upper_y, alpha=0.1)
            # for j, k in zip(x, y):
            #     plt.text(j-1, k+text_space, f'{round(k, 2):.2f}', ha='center', va='top')     # str(round(k, 2)).lstrip('0')

        # Adjust labels for the plot.
        if metric == 'f1_macro':
            metric_label = 'Macro F1 score'
        elif metric == 'rmse':
            metric_label = 'RMSE'

        # Configure current plot.
        plt.xlabel('Train ratio (%)', labelpad=15)
        plt.xticks(x)
        plt.ylabel(metric_label, labelpad=15)
        plt.ylim(0, y_lim)
        plt.legend(loc=loc)
        plt.grid(axis='y', color='gainsboro', linestyle='-', linewidth=0.25)
        plt.subplots_adjust(bottom=0.15)
        plt.tight_layout()

        # Save figure or show.
        if args.save_fig:
            save_path = os.path.join(
                args.output,
                f'{task}_{metric}{transfer[:-1]}.{args.save_fig}'      # -{datetime.now():%Y_%m_%d-%H_%M_%S}
            )
            fig.savefig(save_path, bbox_inches='tight')
            print(f'\nFigure saved at {save_path}')
        else:
            plt.title(transfer)
            plt.show()

    return 0


if __name__ == "__main__":

    # Get arguments.
    args = get_args()

    # Main function.
    sys.exit(main(args))
