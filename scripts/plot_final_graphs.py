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
from matplotlib import pyplot as plt


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

    parser.add_argument('--verbose', '-v', action='store_true',
                        help='provides additional details for debugging purposes.')

    return parser.parse_args(sys.argv[1:])


def main(args):

    # Print target folders.
    if args.verbose:
        print(f"\n---------------------------------------------------") 
        print(f"{'Input folder:'.ljust(16)}{args.input}")
        print(f"{'Output folder:'.ljust(16)}{args.output}")

    # Get a list of all directories.
    dirs = os.listdir(args.input)
    filtered_dirs = sorted([d for d in dirs if 'p' in d])
    print(f"{'Target dirs:'.ljust(16)}{filtered_dirs}") if args.verbose else None

    # Set the transfer learning algorithms.
    transfer_learning_algs = ['_tl=LP_', '_tl=FT_', '_tl=LP+FT_']
    print(f"{'TL algorithms:'.ljust(16)}{transfer_learning_algs}") if args.verbose else None

    # Iterate over the algorithms.
    for tla in transfer_learning_algs:

        # Show information.
        if args.verbose:
            print(f"\n---------------------------------------------------") 
            print(f"{'Current TL:'.ljust(13)}{tla}")

        # Iterate over the dirs.
        for dir in filtered_dirs:

            # Build current path.
            curr_path = os.path.join(args.input, dir)
            print(f"\n{'* Path:'.ljust(9)}{curr_path}") if args.verbose else None

            # Get a list of all directories.
            files = os.listdir(curr_path)
            filtered_files = sorted([f for f in files if tla in f])
            print(f"{'* Target files:'}\n{filtered_files}") if args.verbose else None







    # Filter the files to include only the ones with the desired pattern.
    results = {}
    lp_mean = sorted([f for f in files
                      if ('pp_mean_' in f and '_tl=LP_' in f)])
    lp_std = sorted([f for f in files
                     if ('pp_std_' in f and '_tl=LP_' in f)])
    ft_mean = sorted([f for f in files
                      if ('pp_mean_' in f and '_tl=FT_' in f)])
    ft_std = sorted([f for f in files
                     if ('pp_std_' in f and '_tl=FT_' in f)])
    lpft_mean = sorted([f for f in files
                        if ('pp_mean_' in f and '_tl=LP+FT_' in f)])
    lpft_std = sorted([f for f in files
                       if ('pp_std_' in f and '_tl=LP+FT_' in f)])
    results['lp'] = {'mean': lp_mean, 'std': lp_std}
    # results['ft'] = {'mean': ft_mean, 'std': ft_std}
    # results['lpft'] = {'mean': lpft_mean, 'std': lpft_std}

    # Verbose.
    if args.verbose:
        print('\nFiles being loaded:')
        for tl in results:
            print(f'{tl}:')
            for metric in results[tl]:
                print(f"{f'* {metric}:'.ljust(8)} {results[tl][metric]}")

    # Iterate over the different transfer learning algorithms.
    for transfer_learning in results:

        # Read the CSV file into a pandas DataFrame.
        filename = os.path.join(args.input, results[transfer_learning]['mean'][0])
        print(f'\nfilename: {filename}')
        res_last_epoch = pd.read_csv(filename).iloc[-1, :]
        print(res_last_epoch)

    return 0


if __name__ == "__main__":

    # Get arguments.
    args = get_args()

    # Main function.
    sys.exit(main(args))
