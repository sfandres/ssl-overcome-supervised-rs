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
        description='Script that merges the .csv files.'
    )

    parser.add_argument('--input', '-i', required=True,    # nargs='+',
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
                      if 'multiclass' in f]

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

        prefix = file.split('_s=')[0]                                # Get the prefix.
        
        if prefix != current_prefix:                                 # If the prefix is different from the previous file, start a new group.
            if current_group:
                arranged_files.append(current_group)                 # Start a new group with the current file.
            current_group = [file]
            current_prefix = prefix
        else:
            current_group.append(file)                               # Add the file to the current group.

    if current_group:
        arranged_files.append(current_group)                         # Add the last group to the arranged_files list.

    for i, group in enumerate(arranged_files):                       # Print the arranged files.
        print()
        for file in group:
            print(f'{i} --> {file}')


    for group in arranged_files:                       # Print the arranged files.

        dataframes = []
        last_rows = []

        # Create the col for the last metric.
        col_name = 'f1_per_class'

        # Iterate over the csv files.
        for file in group:

            # Show file.
            print(f"{'File:'.ljust(8)}"
                f"{file}")

            # Get prefix.
            prefix = file.split('_s=')[0]
            print(f"{'Prefix:'.ljust(8)}"
                f"{prefix}")

            # Create the dataframe.
            df = pd.read_csv(os.path.join(args.input, file))

            # Convert the last row of the last column to list and then np.array.
            column_list_str = df.iloc[-1, -1]
            list_floats = [float(x) for x in column_list_str.strip('[]').split(',')]
            array_floats = np.array(list_floats, dtype=float)

            # Remove the last column (no longer needed).
            df = df.drop(col_name, axis=1)

            # Save the last metric and dataframe.
            last_rows.append(array_floats)
            dataframes.append(df)

        # Merge the dataframes based on columns.
        df_concat = pd.concat(dataframes, axis=0)

        # Group by index (epoch) and compute mean by row.
        by_row_index = df_concat.groupby(df_concat.index)
        df_means = by_row_index.mean().round(3)
        df_means['epoch'] = df_means.index
        df_stds = by_row_index.std().round(3)
        df_stds['epoch'] = df_stds.index

        # Compute the mean and std values per column.
        avg_final_metric = np.round(np.mean(last_rows, axis=0), 3)
        formatted_avg_final_metric = ", ".join([str(num) for num in avg_final_metric])
        formatted_avg_final_metric = '[' + formatted_avg_final_metric + ']'

        std_final_metric = np.round(np.std(last_rows, axis=0), 3)
        formatted_std_final_metric = ", ".join([str(num) for num in std_final_metric])
        formatted_std_final_metric = '[' + formatted_std_final_metric + ']'

        # Add a single value at the last row of the dfs.
        df_means[col_name] = ''
        df_means.at[df_means.index[-1], col_name] = formatted_avg_final_metric
        df_stds[col_name] = ''
        df_stds.at[df_stds.index[-1], col_name] = formatted_std_final_metric

        # Fix epoch's column.
        df_means['epoch'] = df_means['epoch'].astype('int')
        df_stds['epoch'] = df_means['epoch'].astype('int')

        # Create a new DataFrame with mean +- std values
        df_both = pd.DataFrame()

        df_both['epoch'] = df_means['epoch']

        for column in df_means.columns:
            if column != 'epoch':
                df_both[column] = df_means[column].astype(str) + ' +- ' + df_stds[column].astype(str)

        df_both = df_both.drop(col_name, axis=1)

        # Save the mean and standard deviation dataframes to CSV files.
        df_means.to_csv(os.path.join(args.output, f'pp_mean_{prefix}.csv'), index=False)
        df_stds.to_csv(os.path.join(args.output, f'pp_std_{prefix}.csv'), index=False)
        df_both.to_csv(os.path.join(args.output, f'pp_both_{prefix}.csv'), index=False)


if __name__ == "__main__":

    # Get arguments.
    args = get_args()

    # Main function.
    sys.exit(main(args))