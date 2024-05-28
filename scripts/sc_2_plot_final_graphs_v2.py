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
def get_args() -> argparse.Namespace:
    """
    Parse and retrieve command-line arguments.

    Returns:
        An 'argparse.Namespace' object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Script that creates the figures for the paper.')

    parser.add_argument('--input', '-i', required=True,
                        help='path to the parent folder where the different train ratio folders are stored.')

    parser.add_argument('--output', '-o', default='./',
                        help='path to the folder where the figure will be saved.')

    parser.add_argument('--metric', '-m', required=True,
                        choices=['top1', 'f1_micro', 'top5', 'f1_macro', 'f1_weighted',
                                 'rmse', 'mae'],
                        help='parameter to be displayed in the y-axis.')

    parser.add_argument('--save_fig', '-sf', type=str, choices=['png', 'pdf'],
                        help='format of the output image (default: png).')

    parser.add_argument('--verbose', '-v', action='store_true',
                        help='provides additional details for debugging purposes.')

    return parser.parse_args(sys.argv[1:])


# =========================================
def create_new_column_pandas(row: pd.core.series.Series) -> str:
    """
    Function to apply conditions and create a new column.

    Returns:
        A string with the new column name.
    """
    if row['weights'] == 'imagenet' and row['model'] == 'Supervised':
        return f"FS-ImageNet-{row['transfer']}"
    if row['weights'] == 'random' and row['model'] == 'Supervised':
        return f"FS-Random-{row['transfer']}"
    elif row['model'] == 'BarlowTwins':
        return f"SSL-BarlowTwins-{row['transfer']}"
    else:
        return None


# =========================================
def main(args):

    # Print target folders.
    if args.verbose:
        print(f"\n---------------------------------------------------") 
        print(f"{'Input folder:'.ljust(16)}{args.input}")
        print(f"{'Output folder:'.ljust(16)}{args.output}")

    # Configure matplotlib.
    set_plt()

    # Set up dictionaries.
    dict_color_models = {'BarlowTwins': 'blue', 'ImageNet': 'orange', 'Random': 'green'}
    dict_marker_models = {'FT': 'o', 'LP': 'o'}
    dict_lines_models = {'FT': '-', 'LP': '--'}
    dict_metrics = {'top1': 'Top-1 Accuracy', 'f1_micro': 'Micro F1', 'top5': 'Top-5 Accuracy',
                    'f1_macro': 'Macro F1', 'f1_weighted': 'Weighted F1', 'rmse': 'RMSE', 'mae': 'MAE'}
    list_models = dict_color_models.keys()

    # Get task from first item and set target metric and reference.
    task = args.input.split('/')[-2]
    print(f"{'Task:'.ljust(16)}{task}") if args.verbose else None

    # Horizontal axis.
    x = [1, 5, 10, 25, 50, 100]

    # Get a list of all directories in root directory.
    dirs = os.listdir(args.input)
    filtered_dirs = sorted([d for d in dirs if 'p' in d])
    if args.verbose:
        print(f"{'Target ratios:'.ljust(16)}{x}")
        print(f"{'Target dirs:'.ljust(16)}{filtered_dirs}")
        print(f"{'Target metric:'.ljust(16)}{args.metric}")

    # Initialize an empty list to store DataFrames.
    dfs_mean = []
    dfs_std = []

    # Iterate through the folders in the root directory.
    for folder in filtered_dirs:

        folder_path = os.path.join(args.input, folder)
        mean_csv_files = [file for file in sorted(os.listdir(folder_path))
                          if 'pp_mean_' in file]

        # Iterate through the CSV files in the folder.
        for file_name in mean_csv_files:

            file_path = os.path.join(folder_path, file_name)

            # Get the features from the file name.
            features = file_name.split('.csv')[0].replace('=', '_').split('_')

            # Load the CSV into a DataFrame.
            df = pd.read_csv(file_path)

            # Add columns for train ratio and file name.
            df['file_name'] = file_name
            df['train_ratio'] = str(int(float(features[4]) * 100))
            df['model'] = features[6]
            df['transfer'] = features[10]
            df['weights'] = features[12]

            # Find the row with the maximum value in the metric column.
            if args.metric == 'rmse' or args.metric == 'mae':
                found_df = df[df['val_' + args.metric] == df['val_' + args.metric].min()]
                found_row = found_df.iloc[0]
                found_row_epoch = found_df.iloc[0]['epoch']
            else:
                found_df = df[df['val_' + args.metric] == df['val_' + args.metric].max()]
                found_row = found_df.iloc[0]
                found_row_epoch = found_df.iloc[0]['epoch']

            # Find the corresponding std row.
            std_file_name = file_name.replace('pp_mean_', 'pp_std_')
            file_path = os.path.join(folder_path, std_file_name)
            std_df = pd.read_csv(file_path)
            std_found_row = std_df[std_df['epoch'] == found_row_epoch].iloc[0]

            # Append both rows to the lists.
            dfs_mean.append(found_row)
            dfs_std.append(std_found_row)

    # Create a DataFrame of means and stds.
    df_means = pd.DataFrame(dfs_mean)
    df_means = df_means.reset_index(drop=True)
    df_means['label'] = df_means.apply(create_new_column_pandas, axis=1)
    # df_means = df_means.round(3)
    df_means.to_csv(os.path.join(args.output, f'exp_{task}_m={args.metric}_means.csv'), index=False)
    print(df_means)

    df_stds = pd.DataFrame(dfs_std)
    df_stds['epoch'] = df_stds['epoch'].astype(int)
    df_stds = df_stds.reset_index(drop=True)
    # df_stds = df_stds.round(3)
    df_stds.to_csv(os.path.join(args.output, f'exp_{task}_m={args.metric}_stds.csv'), index=False)
    print(df_stds)

    # Plotting.
    fig = plt.figure(figsize=(18, 6))

    # Iterate over unique models and transfer methods.
    for model in list_models:

        # Filter the DataFrame by model.
        filtered_df = df_means[df_means['label'].str.contains(model)]
        if args.verbose:
            print(filtered_df)

        # Iterate over unique transfer methods.
        for transfer in filtered_df['transfer'].unique():

            # Filter the DataFrame by transfer method.
            subset = filtered_df[filtered_df['transfer'] == transfer]
            if args.verbose:
                print(subset)

            # Extract indices from df1
            indices_to_filter = subset.index

            # Filter the stds by indexing with the indices from df1.
            subset_std = df_stds.loc[indices_to_filter]
            print(subset_std)

            # Create the labels.
            if model == 'BarlowTwins':
                label = f'SSL-BarlowTwins-{transfer}'
            elif model == 'ImageNet':
                label = f'FS-ImageNet-{transfer}'
            elif model == 'Random':
                label = f'FS-Random-{transfer}'

            # Plot the data.
            plt.plot(subset['train_ratio'], subset['test_' + args.metric], label=label, color=dict_color_models[model], marker=dict_marker_models[transfer], linestyle=dict_lines_models[transfer])
            plt.fill_between(subset['train_ratio'], subset['test_' + args.metric]-subset_std['test_' + args.metric], subset['test_' + args.metric]+subset_std['test_' + args.metric], alpha=0.125, color=dict_color_models[model])

    # Customize the plot
    plt.xlabel('Train ratio (%)', labelpad=15)
    plt.ylabel(dict_metrics[args.metric], labelpad=15)
    plt.legend(title='Model', loc='center', bbox_to_anchor=(1.3, 0.5), ncol=1)
    plt.grid(axis='both', color='gainsboro', linestyle='-', linewidth=0.25, zorder=0)
    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()

    # Save figure or show.
    if args.save_fig:
        save_path = os.path.join(
            args.output,
            f'exp_{task}_m={args.metric}.{args.save_fig}'
        )
        fig.savefig(save_path, bbox_inches='tight')
        print(f'Figure saved at {save_path}')
    else:
        plt.show()


if __name__ == "__main__":

    # Get arguments.
    args = get_args()

    # Main function.
    sys.exit(main(args))
