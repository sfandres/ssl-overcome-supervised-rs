"""
Writes to a csv the mean and std values of the merged
dataframes generated from the different experimental trials.

Usage: compute_mean_std_from_csv.py [-h] [--input INPUT] [--output OUTPUT]

Script that merges the input .csv files.

options:
  -h, --help       show this help message and exit
  --input INPUT    path to the input directory where the csv files are stored.
  --output OUTPUT  path to the output directory where the generated csv files will be stored.

Author:
    A.J. Sanchez-Fernandez - 07/07/2023
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
    parser = argparse.ArgumentParser(
        description='Script that merges the .csv files.'
    )

    parser.add_argument('--input', '-i', required=True,
                        help='csv file(s) to be merged.')

    parser.add_argument('--output', '-o', required=True,
                        help='folder to output files.')

    return parser.parse_args(sys.argv[1:])


def main(args):

    # Print target folder.
    print(f'Target folder: {args.input}')

    # Get a list of all files in the directory.
    files = os.listdir(args.input)

    # Filter the files to include only the ones with the desired pattern.
    filtered_files = [f for f in files
                      if '_s=' in f]

    # Sort the files based on the first part of the filename.
    sorted_files = sorted(
        filtered_files,
        key=lambda f: f.split('_lr=')[0],
        reverse=False
    )

    # Create a list of lists to store the arranged files.
    arranged_files = []
    current_group = []
    current_prefix = None

    # Iterate through the sorted files.
    for file in sorted_files:

        prefix = file.split('_s=')[0]                               # Get the prefix.
        
        if prefix != current_prefix:                                # If the prefix is different from the previous file, start a new group.
            if current_group:
                arranged_files.append(current_group)                # Start a new group with the current file.
            current_group = [file]
            current_prefix = prefix
        else:
            current_group.append(file)                              # Add the file to the current group.

    # Add the last group to the arranged_files list.
    if current_group:
        arranged_files.append(current_group)                         

    # Print the arranged files.
    for i, group in enumerate(arranged_files):
        print()
        for file in group:
            print(f'{i} --> {file}')

    # Iterate over the groups of files.
    for group in arranged_files:

        print()
        dataframes = []

        # Get task from first item.
        task = group[0].split('_tr=')[0]
        print(f"{'Task:'.ljust(8)}{task}")

        # Target metric.
        if task == 'multiclass':
            per_class_metric = 'f1_per_class'
        elif task == 'multilabel':
            per_class_metric = 'rmse_per_class'
        else:
            per_class_metric = None

        # Get prefix from first item.
        prefix = group[0].split('_s=')[0]
        print(f"{'Prefix:'.ljust(8)}{prefix}")

        # ==============================================

        # Iterate over the csv files.
        for file in group:

            print(f"{'File:'.ljust(8)}{file}")                          # Show file.

            df = pd.read_csv(os.path.join(args.input, file))            # Create the dataframe.

            df[per_class_metric] = df[per_class_metric].apply(          # Convert every row of the last column to np.array with floats.
                lambda x: np.array(
                    [float(i) for i in x.strip('[]').split(',')],
                    dtype=float
                )
            )

            per_class_metric_df = pd.DataFrame(                         # Expand the target column into multiple columns, one per value.
                df[per_class_metric].tolist(),
                index=df.index
            )

            per_class_metric_df.columns = [                             # Rename the columns.
                f'{per_class_metric}_{i}'
                for i in range(per_class_metric_df.shape[1])
            ]

            df.drop(columns=per_class_metric, inplace=True)             # Drop the target column from the original dataframe.

            df_expanded = pd.concat([df, per_class_metric_df],          # Concatenate the original dataframe with the expanded target column.
                                    axis=1)

            dataframes.append(df_expanded)                              # Append the expanded dataframe to the list of dataframes.

        print(f'\nDATAFRAMES:\n{dataframes}')

        # ==============================================

        df_concat = pd.concat(dataframes, axis=0)                       # Merge the dataframes by rows.
        print(f'\nCONCAT:\n{df_concat}')

        by_row_index = df_concat.groupby(df_concat.index)               # Group the dataframe by row index (epoch).

        df_means = by_row_index.mean().round(3)                         # Compute the mean and std values per column.
        df_stds = by_row_index.std().round(3)

        df_means['epoch'] = df_means.index.astype('int')                # Add the epoch column to the new dataframe and convert it to int.
        df_stds['epoch'] = df_stds.index.astype('int')

        print(f'\nMEANs:\n{df_means}')
        print(f'\nSTDs:\n{df_stds}')

        df_means.to_csv(os.path.join(args.output, f'pp_mean_{prefix}.csv'), index=False)
        df_stds.to_csv(os.path.join(args.output, f'pp_std_{prefix}.csv'), index=False)

        # Create a new DataFrame with mean+-std values wo/ col_name.
        # df_both = pd.DataFrame()
        # df_both['epoch'] = df_means['epoch']
        # for column in df_means.columns:
        #     if column != 'epoch':
        #         df_both[column] = df_means[column].astype(str) + '+-' + df_stds[column].astype(str)
        # df_both = df_both.drop(col_name, axis=1)
        # df_both.to_csv(os.path.join(args.output, f'pp_both_{prefix}.csv'), index=False)

        # Save the mean and standard deviation dataframes to CSV files.

    return 0


if __name__ == "__main__":

    # Get arguments.
    args = get_args()

    # Main function.
    sys.exit(main(args))
