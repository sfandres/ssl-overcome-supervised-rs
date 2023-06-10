"""Prints the plot of the accuracy values
   obtained during training from an csv file.

Author:
    A.J. Sanchez-Fernandez - 31/05/2023
"""


import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import math
from datetime import datetime
import numpy as np


def get_args() -> argparse.Namespace:
    """
    Parse and retrieve command-line arguments.

    Returns:
        An 'argparse.Namespace' object containing the parsed arguments.
    """

    # Get arguments.
    parser = argparse.ArgumentParser(
        description='Script that plots the accuracy values from .csv file.'
    )

    parser.add_argument('--input', '-i', nargs='+', required=True,
                        help='csv file(s) to plot.')

    parser.add_argument('--downstream_task', '-dt', type=str, required=True,
                        choices=['multiclass', 'multilabel'],
                        help='type of downstream task (metrics in the y-axis).')

    parser.add_argument('--save_fig', '-sf', type=str, choices=['png', 'pdf'],
                        help='format of the output image (default: png).')

    return parser.parse_args(sys.argv[1:])


def main(args):

    # Show filenames.
    print('Files used for the plot:')
    for i, filename in enumerate(args.input):
        print(f'  {i:02d} --> {filename}')

    # Target metrics.
    if args.downstream_task == "multiclass":
        metrics = ['train_loss', 'val_loss', 'top1', 'top5', 'f1_macro', 'f1_weighted', 'f1_per_class']  # 'f1_micro'
        bbox_to_anchor = (0.45, -0.7)
    else:   
        metrics = ['train_loss', 'val_loss', 'rmse', 'mae', 'rmse_per_class']
        bbox_to_anchor = (0.45, -0.45)

    # Calculate the number of rows and columns for subplots.
    num_metrics = len(metrics)
    num_columns = 2
    num_rows = math.ceil(num_metrics / num_columns)
    bar_width = 0.2
    bar_space = 0.3

    # Create a subplot for each metric.
    fig = plt.figure(figsize=(40, 8*num_rows))
    grid = plt.GridSpec(num_rows, num_columns, hspace=0.6, wspace=0.3)

    # Create axes.
    axes = []
    for i in range(len(metrics)):

        # Calculate the subplot indices.
        row_index = i // num_columns
        col_index = i % num_columns

        # If not the latest row.
        if i < len(metrics) - 1:
            axes.append(fig.add_subplot(grid[row_index, col_index]))
        else:
            axes.append(fig.add_subplot(grid[-1, :]))

    # Iterate over the metrics.
    for i, metric in enumerate(metrics):

        # Iterate over each CSV file.
        for nf, filename in enumerate(args.input):

            # Read the CSV file into a pandas DataFrame.
            df = pd.read_csv(filename)

            # Extract the 'epoch' name.
            x_label = list(df.columns)[0]
            x = df[x_label]

            # Extract the metrics and plot them.
            try:

                # Special case with metrics per class.
                if metric == 'f1_per_class' or metric == 'rmse_per_class':
                    column_values_str = df[metric].iloc[-1]
                    y = [float(x) for x in column_values_str.strip('[]').split(',')]
                    x = np.array(range(len(y)))
                    axes[i].bar(x + nf * bar_space, y, width=bar_width, label=filename.rsplit('/', 1)[-1][:-4])
                    for j, k in zip(x, y):
                        axes[i].text(j + nf * bar_space, k, str(round(k, 2)), ha='center', va='bottom')
                        axes[i].set_xticks([j + (nf * bar_space)/2 for j in x], x)
                    if args.downstream_task == 'multiclass':
                        axes[i].set_ylim(0, 1)
                    else:
                        axes[i].set_ylim(0, max(y)+0.1)
                    axes[i].set_xlabel('Classes')
                else:
                    y = df[metric]
                    axes[i].plot(x, y, label=filename.rsplit('/', 1)[-1][:-4])
            except KeyError:
                print(f"KeyError: Column '{metric}' not found in dataframe. "
                      f"This may not be the right metric for the task at hand.")
                return 1
            axes[i].set_ylabel(metric.capitalize())

    # Set x-label for the bottom-most subplot.
    axes[-2].set_xlabel(x_label)
    axes[-3].set_xlabel(x_label)

    # Create a legend below the last row of subplots.
    handles, labels = axes[0].get_legend_handles_labels()
    plt.legend(handles=handles,
               labels=labels,
               loc='lower center',
               ncol=2,
               labelspacing=0.,
               bbox_to_anchor=bbox_to_anchor,
               fancybox=True,
               shadow=True)

    # Adjust spacing between subplots.
    plt.subplots_adjust(wspace=0.2, hspace=0.1)

    # # Adjust subplot spacing.
    # plt.tight_layout(rect=[0, 0, 0.5, 1.0])

    # Save figure or show.
    if args.save_fig:
        fig.savefig(f'fig_{args.downstream_task}-{datetime.now():%Y_%m_%d-%H_%M_%S}.{args.save_fig}',
                    bbox_inches='tight')
    else:
        plt.show()

    return 0


if __name__ == "__main__":

    # Get arguments.
    args = get_args()

    # Main function.
    sys.exit(main(args))
