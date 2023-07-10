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

    # Set the models.
    # models = ['SSL', 'Supervised-ImageNet', 'Supervised-random']
    models = ['Supervised-ImageNet', 'Supervised-random']
    print(f"{'Models:'.ljust(16)}{models}") if args.verbose else None

    # Iterate over the algorithms.
    for tla in transfer_learning_algs:

        # Show information.
        if args.verbose:
            print(f"\n---------------------------------------------------") 
            print(f"{'Curr TL:'.ljust(13)}{tla}")

        # Iterate over the models.
        for model in models:

            # Create the last filters according to the target model.
            if model == 'SSL':
                filter1 = 'BarlowTwins'
                filter2 = '_iw=random'
            elif model == 'Supervised-ImageNet':
                filter1 = 'Supervised'
                filter2 = '_iw=imagenet'
            elif model == 'Supervised-random':
                filter1 = 'Supervised'
                filter2 = '_iw=random'
            mean_files, std_files = [], []
            mean_values, std_values = [], []
            print(f"\n{'Curr model:'.ljust(13)}{model}")

            # Iterate over the dirs.
            for ratio in filtered_dirs:

                # Build current path.
                curr_path = os.path.join(args.input, ratio)
                # print(f"\n{'* Path:'.ljust(9)}{curr_path}") if args.verbose else None

                # Get a list of all directories.
                files = os.listdir(curr_path)
                filtered_files = sorted([f for f in files if tla in f])

                # Filter current files.
                curr_mean_file = [f for f in filtered_files
                             if ('pp_mean_' in f and filter1 in f and filter2 in f)][0]
                curr_std_file = [f for f in filtered_files
                            if ('pp_std_' in f and filter1 in f and filter2 in f)][0]

                # Append filenames.
                mean_files.append(os.path.join(curr_path, curr_mean_file))
                std_files.append(os.path.join(curr_path, curr_std_file))

                # Append mean and std values.
                res_mean_last_epoch = pd.read_csv(os.path.join(curr_path, curr_mean_file)).iloc[-1, :]['f1_macro']
                mean_values.append(res_mean_last_epoch)
                res_std_last_epoch = pd.read_csv(os.path.join(curr_path, curr_std_file)).iloc[-1, :]['f1_macro']
                std_values.append(res_std_last_epoch)

            # Show information.
            print('Target files:')
            for mfile, sfile, mvalue, svalue in zip(mean_files, std_files, mean_values, std_values):
                print(mfile, '-->', mvalue)
                print(sfile, '-->', svalue)






                # # Filter current files.
                # curr_mean = [f for f in mean_files
                #              if ('pp_mean_' in f and filter1 in f and filter2 in f)][0]
                # curr_std = [f for f in std_files
                #             if ('pp_std_' in f and filter1 in f and filter2 in f)][0]

                # # Create final filenames.
                # curr_mean_filename = os.path.join(curr_path, curr_mean)
                # curr_std_filename = os.path.join(curr_path, curr_std)
                # if args.verbose:
                #     print(f"\n{'- File:'.ljust(9)}{curr_mean_filename}")
                #     print(f"{'- File:'.ljust(9)}{curr_std_filename}")

                # # Read the CSV file into a pandas DataFrame.
                # res_mean_last_epoch = pd.read_csv(curr_mean_filename).iloc[-1, :]['f1_macro']
                # res_std_last_epoch = pd.read_csv(curr_std_filename).iloc[-1, :]['f1_macro']

                # # Append results.
                # # results[model].append((res_mean_last_epoch, res_std_last_epoch))
                # print(res_mean_last_epoch)
                # print(res_std_last_epoch)


    return 0


if __name__ == "__main__":

    # Get arguments.
    args = get_args()

    # Main function.
    sys.exit(main(args))
