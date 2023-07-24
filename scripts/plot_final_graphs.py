"""Creates the final figures from the csv files that contain
   the mean and std values of the merged dataframes.

Usage: plot_final_graphs.py [-h] --input INPUT [--output OUTPUT] --metric {top1,f1_micro,top5,f1_macro,f1_weighted,rmse,mae,f1_per_class,rmse_per_class}
                            [--save_fig {png,pdf}] [--verbose]

Script that creates the figures for the paper.

options:
  -h, --help            show this help message and exit
  --input INPUT, -i INPUT
                        path to the parent folder where the different train ratio folders are stored.
  --output OUTPUT, -o OUTPUT
                        path to the folder where the figure will be saved.
  --metric {top1,f1_micro,top5,f1_macro,f1_weighted,rmse,mae,f1_per_class,rmse_per_class}, -m {top1,f1_micro,top5,f1_macro,f1_weighted,rmse,mae,f1_per_class,rmse_per_class}
                        parameter to be displayed in the y-axis.
  --save_fig {png,pdf}, -sf {png,pdf}
                        format of the output image (default: png).
  --verbose, -v         provides additional details for debugging purposes.

Author:
    A.J. Sanchez-Fernandez - 19/07/2023
"""


import pandas as pd
import numpy as np
import argparse
import sys
import os
from datetime import datetime
from matplotlib import pyplot as plt


MARKER_SIZE = 9
SMALL_SIZE = 18
MEDIUM_SIZE = 22
BIGGER_SIZE = 24


# =========================================
def set_plt() -> None:
    """
    Configure matplotlib figures.
    """

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


# =========================================
def compute_max_bar(data: dict, metric: str, verbose: bool) -> dict:
    """
    Compute the max value per bar for the plot.

    Args:
        data (dict): dictionary with the array of arrays per model.
        metric (str): target metric.
        verbose (bool): show more information.

    Returns:
        The dictionary with max/min values and 0s.
    """

    # Get the number of columns (assuming all models have the same number of columns).
    num_arrays = len(data[list(data.keys())[0]])
    num_columns_per_array = len(data[list(data.keys())[0]][0])

    # Initialize a list to store the maximum values per column.
    ref_values_array = []

    # Iterate through the arrays, positions and then models.
    for i in range(num_arrays):

        # Configure the metric.
        if metric == 'f1_per_class':
            ref_values = [0] * num_columns_per_array
            fn = max
        elif metric == 'rmse_per_class':
            ref_values = [1] * num_columns_per_array
            fn = min

        for j in range(num_columns_per_array):

            # Build the max values array.
            for model in data.values():
                ref_values[j] = fn(ref_values[j], model[i][j])

            # Write 0s if not max.
            for model in data.values():
                if model[i][j] != ref_values[j]:
                    model[i][j] = 0

        ref_values_array.append(ref_values)

    if verbose:
        print('\nReference values (max values) from compute_max_bar():')
        print(ref_values_array)
        print('\nFinal data to plot:')
        print(data)

    return data


# =========================================
def compute_diff_bar(data: dict, verbose: bool) -> dict:
    """
    Compute the difference per bar for the plot.

    Args:
        data (dict): dictionary with the array of arrays per model.
        metric (str): target metric.
        verbose (bool): show more information.

    Returns:
        The dictionary with difference values.
    """

    # Get the number of columns (assuming all models have the same number of columns).
    num_arrays = len(data[list(data.keys())[0]])
    num_columns_per_array = len(data[list(data.keys())[0]][0])

    # Initialize a list to store the maximum values per column.
    array_of_values = []

    # Iterate through the arrays, positions and then models.
    for i in range(num_arrays):

        ref_values_array = []

        for j in range(num_columns_per_array):

            res = data['Barlow Twins'][i][j] - data['ImageNet'][i][j]
            ref_values_array.append(round(res, 3))

        array_of_values.append(ref_values_array)

    if verbose:
        print('\nFinal data to plot:')
        print(array_of_values)

    return array_of_values


# =========================================
def get_args() -> argparse.Namespace:
    """
    Parse and retrieve command-line arguments.

    Returns:
        An 'argparse.Namespace' object containing the parsed arguments.
    """

    parser = argparse.ArgumentParser(description='Script that creates the figures for the paper.')

    parser.add_argument('--input', '-i', required=True,    # nargs='+',
                        help='path to the parent folder where the different train ratio folders are stored.')

    parser.add_argument('--output', '-o', default='./',
                        help='path to the folder where the figure will be saved.')

    parser.add_argument('--metric', '-m', required=True,
                        choices=['top1', 'f1_micro', 'top5', 'f1_macro', 'f1_weighted',
                                 'rmse', 'mae', 'f1_per_class', 'rmse_per_class'],
                        help='parameter to be displayed in the y-axis.')

    parser.add_argument('--bar', '-b', default='both', choices=['both', 'best', 'diff'],
                        help='type of bar plot.')

    parser.add_argument('--save_fig', '-sf', type=str, choices=['png', 'pdf'],
                        help='format of the output image (default: png).')

    parser.add_argument('--verbose', '-v', action='store_true',
                        help='provides additional details for debugging purposes.')

    return parser.parse_args(sys.argv[1:])


# =========================================
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
    if task == 'multiclass':
        loc = 'lower right'
        y_lim = 1.0
        text_space = 0.05
    elif task == 'multilabel':
        loc = 'upper right'
        y_lim = 0.3
        text_space = 0.01
    else:
        loc = None
    print(f"{'Task:'.ljust(16)}{task}") if args.verbose else None

    # Adjust labels for the plot and other parameters.
    bar_width = 0.05
    bar_space = bar_width + bar_width / 2
    dict_colors = {
        'Barlow Twins': 'blue',
        'ImageNet': 'orange',
        # 'Random': 'green'
    }
    if args.metric == 'f1_macro':
        metric_label = 'Macro F1 score'
    elif args.metric == 'f1_per_class':
        metric_label = 'F1 score per class'
    elif args.metric == 'rmse':
        metric_label = 'RMSE'
    elif args.metric == 'rmse_per_class':
        metric_label = 'RMSE per class'
    else:
        metric_label = args.metric

    # Horizontal axis.
    x = [1, 5, 10, 25, 50, 100]
    x_axis = np.arange(len(x))

    # Get a list of all directories.
    bar_dict = {}
    dirs = os.listdir(args.input)
    filtered_dirs = sorted([d for d in dirs if 'p' in d])
    if args.verbose:
        print(f"{'Target ratios:'.ljust(16)}{x}")
        print(f"{'Target dirs:'.ljust(16)}{filtered_dirs}")
        print(f"{'Target metric:'.ljust(16)}{args.metric}")

    # Set the transfer learning algorithms.
    transfer_learning_algs = ['_tl=FT_']     # '_tl=LP_', , '_tl=LP+FT_'
    print(f"{'TL algorithms:'.ljust(16)}{transfer_learning_algs}") if args.verbose else None

    # Set the models.
    models = ['Barlow Twins', 'ImageNet']   #'Random', 
    print(f"{'Models:'.ljust(16)}{models}") if args.verbose else None

    # Iterate over the algorithms.
    for transfer in transfer_learning_algs:

        # Show information.
        if args.verbose:
            print(f"\n---------------------------------------------------") 
            print(f"{'Curr TL:'.ljust(13)}{transfer}")

        # Create fig.
        if 'per_class' in args.metric:
            fig = plt.figure(figsize=(18, 6))
        else:
            fig = plt.figure(figsize=(8, 6))

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
            for nf, ratio in enumerate(filtered_dirs):

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

                # Get last values.
                res_mean_last_epoch = pd.read_csv(os.path.join(curr_path, curr_mean_file)).iloc[-1, :][args.metric]
                res_std_last_epoch = pd.read_csv(os.path.join(curr_path, curr_std_file)).iloc[-1, :][args.metric]

                # Convert the list of strings to a list of integers.
                if 'per_class' in args.metric:
                    res_mean_last_epoch = [float(x) for x in res_mean_last_epoch.strip('[]').split(',')]
                    res_std_last_epoch = [float(x) for x in res_std_last_epoch.strip('[]').split(',')]

                # Append values.
                mean_values.append(res_mean_last_epoch)
                std_values.append(res_std_last_epoch)

            # Save the bar values.
            bar_dict[model] = mean_values

            # Show information.
            if args.verbose:
                print('Target files:')
                for mfile, sfile, mvalue, svalue in zip(mean_files, std_files, mean_values, std_values):
                    print(mfile, '-->', mvalue)
                    print(sfile, '-->', svalue)
                print('\nMean values:', mean_values)
                print('Std values:', std_values)

            # Bar plots with all values.
            if 'per_class' in args.metric:
                if args.bar == 'both':
                    y_trans = np.transpose(bar_dict[model])
                    for nf, values in enumerate(y_trans):
                        plt.bar(x_axis + nf*bar_space - bar_space*len(y_trans)/2, values, width=bar_width, color=dict_colors[model])
                    labels = list(dict_colors.keys())
                    handles = [plt.Rectangle((0,0),1,1, color=dict_colors[label]) for label in labels]
                    plt.legend(handles, labels)

            # Line and marker plot. 
            else:
                y = np.array(mean_values)
                lower_y = y - np.array(std_values)
                upper_y = y + np.array(std_values)
                plt.plot(x_axis, y, 'x-', label=model, markersize=MARKER_SIZE, color=dict_colors[model])
                plt.fill_between(x_axis, lower_y, upper_y, alpha=0.1, color=dict_colors[model])
                plt.legend(loc=loc)
                # for j, k in zip(x_axis, y):
                #     plt.text(j-0.25, k+text_space, f'{round(k, 2):.2f}', ha='center', va='top')     # str(round(k, 2)).lstrip('0')

        if args.verbose:
            print('\nDictionary including means:')
            print(bar_dict)

        if 'per_class' in args.metric:
            if args.bar == 'best':
                data = compute_max_bar(bar_dict, args.metric, args.verbose)
                for model in models:
                    y_trans = np.transpose(data[model])
                    for nf, values in enumerate(y_trans):
                        plt.bar(x_axis + nf*bar_space - bar_space*len(y_trans)/2, values, width=bar_width, color=dict_colors[model])
                labels = list(dict_colors.keys())
                handles = [plt.Rectangle((0,0),1,1, color=dict_colors[label]) for label in labels]
                plt.legend(handles, labels)
            elif args.bar == 'diff':
                data = compute_diff_bar(bar_dict, args.verbose)
                y_trans = np.transpose(data)
                print(y_trans)
                print()
                for nf, values in enumerate(y_trans):
                    print(values)
                    plt.bar(x_axis + nf*bar_space - bar_space*len(y_trans)/2, values, width=bar_width)

        # Configure current plot.
        plt.ylim(0, y_lim)
        if 'f1_per_class' in args.metric and args.bar == 'diff':
            plt.ylim(-0.3, 0.3)
        elif 'rmse_per_class' in args.metric and args.bar == 'diff':
            plt.ylim(-0.05, 0.05)
        plt.xticks(x_axis, x)
        plt.xlabel('Train ratio (%)', labelpad=15)
        plt.ylabel(metric_label, labelpad=15)
        plt.grid(axis='y', color='gainsboro', linestyle='-', linewidth=0.25)
        plt.subplots_adjust(bottom=0.15)
        plt.tight_layout()

        # Save figure or show.
        if args.save_fig:
            save_path = os.path.join(
                args.output,
                f'exp_{task}_m={args.metric}{transfer[:-1]}_{args.bar}.{args.save_fig}'      # -{datetime.now():%Y_%m_%d-%H_%M_%S}
            )
            fig.savefig(save_path, bbox_inches='tight')
            print(f'Figure saved at {save_path}')
        else:
            plt.title(transfer)
            plt.show()

    return 0


if __name__ == "__main__":

    # Get arguments.
    args = get_args()

    # Main function.
    sys.exit(main(args))
