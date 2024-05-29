import pandas as pd
import numpy as np
import argparse
import sys
import os
from datetime import datetime
from matplotlib import pyplot as plt
import csv


MARKER_SIZE = 12
SMALL_SIZE = 18
MEDIUM_SIZE = 22
BIGGER_SIZE = 24


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
def get_args() -> argparse.Namespace:
    """
    Parse and retrieve command-line arguments.

    Returns:
        An 'argparse.Namespace' object containing the parsed arguments.
    """

    parser = argparse.ArgumentParser(description='Script that takes the values from the csv files for the paper.')

    parser.add_argument('--input_df_means_path', '-i', required=True,    # nargs='+',
                        help='path to the input dataframe with means.')

    parser.add_argument('--output', '-o', default='./',
                        help='path to the folder where the csv will be saved.')

    parser.add_argument('--verbose', '-v', action='store_true',
                        help='provides additional details for debugging purposes.')

    return parser.parse_args(sys.argv[1:])


def main(args: argparse.Namespace) -> bool:
    """
    Main function.

    Args:
        args (argparse.Namespace): the parsed command-line arguments.

    Returns:
        bool: true if the script is executed successfully.
    """
    args.input_df_means_path = os.path.expanduser(args.input_df_means_path)

    if args.verbose:
        print()
        args_dict = vars(args)
        for arg_name in args_dict:
            arg_name_col = f'{arg_name}:'
            print(f'{arg_name_col.ljust(20)} {args_dict[arg_name]}')

    # Configure matplotlib.
    set_plt()

    # Extract the task from the input DataFrame path.
    task = args.input_df_means_path.split('/')[-1].split('_')[1]
    print(f'Task: {task}') if args.verbose else None

    # Read the input DataFrames.
    df_means = pd.read_csv(args.input_df_means_path)
    df_stds = pd.read_csv(args.input_df_means_path.replace('means', 'stds'))

    # Filter the dataframes to keep only FT data.
    df_means = df_means[df_means['label'].str.contains('FT')]
    df_stds = df_stds[df_stds['label'].str.contains('FT')]

    # List of keywords to search for in column names.
    if task == 'multiclass':
        keywords = ['train_ratio', 'label', 'test_f1_micro', 'test_f1_macro', 'test_f1_weighted', 'test_f1_per_class_']
    elif task == 'multilabel':
        keywords = ['train_ratio', 'label', 'test_rmse', 'test_rmse_per_class_']

    # Extract columns containing any of the keywords.
    filtered_columns_both = [col for col in df_means.columns if any(keyword in col for keyword in keywords)]
    print(filtered_columns_both)

    # Grouping by 'train_ratio' and calculating the mean for each group.
    grouped_df_means = df_means[filtered_columns_both].groupby(['train_ratio', 'label']).mean(numeric_only=True)
    grouped_df_stds = df_stds[filtered_columns_both].groupby(['train_ratio', 'label']).mean(numeric_only=True)
    if args.verbose:
        print(f'\nGROUPED MEANS DATA:\n{grouped_df_means}')
        print(f'\nGROUPED STDS DATA:\n{grouped_df_stds}')

    # Reset index to merge dataframes on 'train_ratio' and 'label'.
    grouped_df_means.reset_index(inplace=True)
    grouped_df_stds.reset_index(inplace=True)

    # Create the final DataFrame with combined mean ± std values.
    cols_to_not_merge = ['train_ratio', 'label', 'epoch', 'file_name', 'model', 'transfer', 'weights', 'label']
    dict_acronyms = {'FS-ImageNet-FT': 'FS-IN-FT', 'FS-Random-FT': 'FS-R-FT', 'SSL-BarlowTwins-FT': 'SSL-BT-FT'}
    final_df = grouped_df_means.copy()
    for col in grouped_df_means.columns:
        if col not in cols_to_not_merge:
            final_df[col] = grouped_df_means[col].map('{:.3f}'.format).astype(str) + '$\pm$' + grouped_df_stds[col].map('{:.3f}'.format).astype(str)
    final_df['label'] = final_df['label'].apply(lambda x: f'\\textit{{{dict_acronyms[x]}}}')
    if args.verbose:
        print(f'\nFINAL DATA:\n{final_df}')

    # Save the final DataFrame to a csv file.
    output_file = os.path.join(args.output, f'exp_{task}_table.csv')
    final_df.to_csv(output_file, index=False)
    print(f'Output: {output_file}') if args.verbose else None

    return 0


if __name__ == '__main__':
    args = get_args()                                                                                       # Parse and retrieve command-line arguments.
    sys.exit(main(args))                                                                                    # Execute the main function.
