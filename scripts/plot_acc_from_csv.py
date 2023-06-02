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

    # Target metrics.
    if args.downstream_task == "multiclass":
        metrics = ['loss', 'top1', 'top5', 'f1_micro', 'f1_macro', 'f1_weighted']
        bbox_to_anchor = (-0.08, -0.35)
    else:
        metrics = ['loss', 'rmse', 'mae']
        bbox_to_anchor = (-0.08, -0.25)

    # Calculate the number of rows and columns for subplots.
    num_metrics = len(metrics)
    num_columns = 2
    num_rows = math.ceil(num_metrics / num_columns)

    # Create a subplot for each metric.
    fig, axs = plt.subplots(num_rows, num_columns, sharex=True, figsize=(40, 8*num_rows))

    # Iterate over the metrics.
    for i, metric in enumerate(metrics):

        # Calculate the subplot indices.
        row_index = i // num_columns
        col_index = i % num_columns

        # Iterate over each CSV file.
        for filename in args.input:

            # Read the CSV file into a pandas DataFrame.
            df = pd.read_csv(filename)

            # Extract the header values.
            headers = list(df.columns)
            x_label = headers[0]

            # Extract the first and second columns.
            x = df[x_label]
            try:
                y = df[metric]
            except KeyError:
                print(f"KeyError: Column '{metric}' not found in dataframe. "
                    f"This may not be the right metric for the task at hand.")
                return 1

            # Plot the metric data.
            axs[row_index, col_index].plot(x, y, label=filename)
            axs[row_index, col_index].set_ylabel(metric.capitalize())

    # Set x-label for the bottom-most subplot
    axs[-1, 0].set_xlabel(x_label)
    axs[-1, 1].set_xlabel(x_label)

    # Create a legend below the last row of subplots
    handles, labels = axs[-1, 0].get_legend_handles_labels()
    plt.legend(handles=handles,
               labels=labels,
               loc='lower center',
               ncol=3,
               labelspacing=0.,
               bbox_to_anchor=bbox_to_anchor,
               fancybox=True,
               shadow=True)

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.2, hspace=0.1)

    # Adjust subplot spacing.
    # plt.tight_layout(rect=[0, 0, 0.5, 1.0])

    # Save figure.
    if args.save_fig:
        fig.savefig(f'fig_{args.downstream_task}-{datetime.now():%Y_%m_%d-%H_%M_%S}.{args.save_fig}',
                    bbox_inches='tight')

    # Show the plot
    plt.show()



    return 0


if __name__ == "__main__":

    # Get arguments.
    args = get_args()

    # Main function.
    sys.exit(main(args))
