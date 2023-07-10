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

    parser.add_argument('--verbose', '-v', action='store_true',
                        help='provides additional details for debugging purposes.')

    return parser.parse_args(sys.argv[1:])


def main(args):

    # Print target folders.
    if args.verbose:
        print(f"\n{'Input folder:'.ljust(16)}{args.input}")
        print(f"{'Output folder:'.ljust(16)}{args.output}")

    # Get a list of all files in the directory.
    files = os.listdir(args.input)

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
    results['ft'] = {'mean': ft_mean, 'std': ft_std}
    results['lpft'] = {'mean': lpft_mean, 'std': lpft_std}

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
