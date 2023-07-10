"""Creates the final figures from the csv files that contain
   the mean and std values of the merged dataframes.

Usage: plot_final_graphs.py [-h] --input INPUT --output OUTPUT

Script that creates the figures for the paper.

options:
  -h, --help            show this help message and exit
  --input INPUT, -i INPUT
                        path to the folder where the csv file(s) are stored.
  --output OUTPUT, -o OUTPUT
                        path to the folder where the figure will be saved.

Author:
    A.J. Sanchez-Fernandez - 10/07/2023
"""


import pandas as pd
import numpy as np
import argparse
import sys
import os


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
                        help='path to the folder where the csv file(s) are stored.')

    parser.add_argument('--output', '-o', required=True,
                        help='path to the folder where the figure will be saved.')

    return parser.parse_args(sys.argv[1:])


def main(args):

    # Print target folders.
    print(f'Input folder: {args.input}')
    print(f'Output folder: {args.output}')

    return 0


if __name__ == "__main__":

    # Get arguments.
    args = get_args()

    # Main function.
    sys.exit(main(args))
