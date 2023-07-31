import pandas as pd
import numpy as np
import argparse
import sys
import os
from datetime import datetime
from matplotlib import pyplot as plt
import csv


# =========================================
def get_args() -> argparse.Namespace:
    """
    Parse and retrieve command-line arguments.

    Returns:
        An 'argparse.Namespace' object containing the parsed arguments.
    """

    parser = argparse.ArgumentParser(description='Script that takes the values from the csv files for the paper.')

    parser.add_argument('--input', '-i', required=True,    # nargs='+',
                        help='path to the parent folder where the different train ratio folders are stored.')

    parser.add_argument('--output', '-o', default='./',
                        help='path to the folder where the figure will be saved.')

    parser.add_argument('--verbose', '-v', action='store_true',
                        help='provides additional details for debugging purposes.')

    return parser.parse_args(sys.argv[1:])


# =========================================
def main(args):

    # Configure the target models.
    models = ['Random', 'ImageNet', 'Barlow Twins']
    print(f"{'Models:'.ljust(16)}{models}") if args.verbose else None

    # Get task from first item and set target metric and reference.
    task = args.input.split('/')[-2]
    print(f"{'Task:'.ljust(16)}{task}") if args.verbose else None

    # Horizontal axis.
    x = [1, 5, 10, 25, 50, 100]

    # Get a list of all directories.
    dirs = os.listdir(args.input)
    filtered_dirs = sorted([d for d in dirs if 'p' in d])
    if args.verbose:
        print(f"{'Target ratios:'.ljust(16)}{x}")
        print(f"{'Target dirs:'.ljust(16)}{filtered_dirs}")

    # Specify the csv file name.
    filename = os.path.join(args.output, f'result_table_{task}.csv')
    print(f"{'Output file:'.ljust(16)}{filename}") if args.verbose else None

    # Set the transfer learning algorithms.
    transfer_learning_algs = ['_tl=FT_']     # '_tl=LP_' '_tl=LP+FT_'
    print(f"{'TL algorithms:'.ljust(16)}{transfer_learning_algs}") if args.verbose else None

    # Writing the data to CSV
    with open(filename, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        if task == 'multiclass':
            csvwriter.writerow(['Model', 'TR (\%)'] + ['F1_per_class' for x in range(10)] + ['Micro F1', 'Macro F1', 'Weighted F1'])
            csvwriter.writerow(['Model', 'TR (\%)'] + [x for x in range(10)] + ['Micro F1', 'Macro F1', 'Weighted F1'])
        elif task == 'multilabel':
            csvwriter.writerow(['Model', 'TR (\%)'] + ['RMSE_per_class' for x in range(10)] + ['RMSE'])
            csvwriter.writerow(['Model', 'TR (\%)'] + [x for x in range(10)] + ['RMSE'])

    # Iterate over the algorithms.
    for transfer in transfer_learning_algs:

        # Show information.
        if args.verbose:
            print(f"\n---------------------------------------------------") 
            print(f"{'Curr TL:'.ljust(13)}{transfer}")
        
        # Iterate over the models.
        for model in models:

            # Create the last filters according to the target model.
            if model == 'Barlow Twins':
                filter1 = 'BarlowTwins'
                filter2 = '_iw=random'
                model_name = 'FS-BarlowTwins'
            elif model == 'ImageNet':
                filter1 = 'Supervised'
                filter2 = '_iw=imagenet'
                model_name = 'FS-ImageNet'
            elif model == 'Random':
                filter1 = 'Supervised'
                filter2 = '_iw=random'
                model_name = 'FS-Random'

            # Show information.
            if args.verbose:
                print(f"\n---------------------------------------------------") 
                print(f"{'Curr model:'.ljust(13)}{model}")

            # Iterate over the dirs.
            for nf, ratio in enumerate(filtered_dirs):

                # Build current path.
                curr_path = os.path.join(args.input, ratio)
                print(f"\n{'* Path:'.ljust(10)}{curr_path}") if args.verbose else None

                # Get a list of all directories.
                files = os.listdir(curr_path)
                filtered_files = sorted([f for f in files if transfer in f])

                # Filter current files.
                curr_mean_file = [f for f in filtered_files
                                  if ('pp_mean_' in f and filter1 in f and filter2 in f)][0]
                curr_std_file = [f for f in filtered_files
                                 if ('pp_std_' in f and filter1 in f and filter2 in f)][0]
                if args.verbose:
                    print(f"\n{'** Files:'.ljust(10)}{curr_mean_file}")
                    print(f"{'** Files:'.ljust(10)}{curr_std_file}") 

                # Get last row.
                res_mean_last_epoch = pd.read_csv(os.path.join(curr_path, curr_mean_file)).iloc[-1, :]
                res_std_last_epoch = pd.read_csv(os.path.join(curr_path, curr_std_file)).iloc[-1, :]

                # Get per-class accuracy value.
                if task == 'multiclass':
                    metric_per_class = 'f1_per_class'
                elif task == 'multilabel':
                    metric_per_class = 'rmse_per_class'
                per_class_mean = res_mean_last_epoch[metric_per_class]
                per_class_std = res_std_last_epoch[metric_per_class]
                per_class_mean = [float(x) for x in per_class_mean.strip('[]').split(',')]
                per_class_std = [float(x) for x in per_class_std.strip('[]').split(',')]
                per_class_res = [f'{x:.3f}$\pm${y:.3f}' for x, y in zip(per_class_mean, per_class_std)]
                if args.verbose:
                    print(f"{'Mean:'.ljust(13)}{per_class_mean}")
                    print(f"{'Std:'.ljust(13)}{per_class_std}")
                    print(f"{'Result:'.ljust(13)}{per_class_res}")

                # Get overall accuracy values.
                if task == 'multiclass':
                    # Micro-F1.
                    f1_micro_mean = res_mean_last_epoch['f1_micro']
                    f1_micro_std = res_std_last_epoch['f1_micro']
                    f1_micro_res = f'{f1_micro_mean:.3f}+-{f1_micro_std:.3f}'
                    # Macro-F1.
                    f1_macro_mean = res_mean_last_epoch['f1_macro']
                    f1_macro_std = res_std_last_epoch['f1_macro']
                    f1_macro_res = f'{f1_macro_mean:.3f}+-{f1_macro_std:.3f}'
                    # Weighted-F1.
                    f1_weighted_mean = res_mean_last_epoch['f1_weighted']
                    f1_weighted_std = res_std_last_epoch['f1_weighted']
                    f1_weighted_res = f'{f1_weighted_mean:.3f}+-{f1_weighted_std:.3f}'
                    if args.verbose:
                        print(f"{'Micro F1:'.ljust(13)}{f1_micro_res}")
                        print(f"{'Macro F1:'.ljust(13)}{f1_macro_res}")
                        print(f"{'Weighted F1:'.ljust(13)}{f1_weighted_res}")

                elif task == 'multilabel':
                    # RMSE.
                    rmse_mean = res_mean_last_epoch['rmse']
                    rmse_std = res_std_last_epoch['rmse']
                    rmse_res = f'{rmse_mean:.3f}+-{rmse_std:.3f}'
                    print(f"{'RMSE:'.ljust(13)}{rmse_res}") if args.verbose else None

                # Final row.
                if task == 'multiclass':
                    row = [model_name, ratio[:-1]] + per_class_res + [f1_micro_res] + [f1_macro_res] + [f1_weighted_res]
                elif task == 'multilabel':
                    row = [model_name, ratio[:-1]] + per_class_res + [rmse_res]
                print(f"{'Final row:'.ljust(13)}\n{row}") if args.verbose else None

                # Writing the data to CSV
                with open(filename, 'a', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow(row)

    return 0


if __name__ == "__main__":

    # Get arguments.
    args = get_args()

    # Main function.
    sys.exit(main(args))