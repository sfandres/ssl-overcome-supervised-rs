

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys


def get_args() -> argparse.Namespace:
    """
    Parse and retrieve command-line arguments.

    Returns:
        An 'argparse.Namespace' object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Script that creates the figures for the paper.')

    parser.add_argument('--input_df_means_path', '-i', required=True,    # nargs='+',
                        help='path to the input dataframe with means.')

    parser.add_argument('--output', '-o', default='./',
                        help='path to the folder where the figure will be saved.')

    parser.add_argument('--metric', '-m', required=True,
                        choices=['f1_per_class', 'rmse_per_class'],
                        help='parameter to be displayed in the y-axis.')

    parser.add_argument('--bar', '-b', choices=['both', 'best', 'diff'],
                        help='type of bar plot.')

    parser.add_argument('--ref', '-r', choices=['Random', 'ImageNet'],
                        help='model to compare with Barlow Twins.')

    parser.add_argument('--save_fig', '-sf', type=str, choices=['png', 'pdf'],
                        help='format of the output image (default: png).')

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
        args_dict = vars(args)
        for arg_name in args_dict:
            arg_name_col = f'{arg_name}:'
            print(f'{arg_name_col.ljust(20)} {args_dict[arg_name]}')

    df = pd.read_csv(args.input_df_means_path)
    # df_stds = pd.read_csv(args.input_df_means_path.replace('means', 'stds'))

    # Identify the columns that represent RMSE per class.
    rmse_columns = [col for col in df.columns if 'rmse_per_class' in col]

    # Filter the DataFrame to only include the RMSE columns and the models with FT in the label.
    filtered_df = df[['train_ratio', 'label'] + rmse_columns]
    filtered_df = filtered_df[filtered_df['label'].str.contains('FT')]
    if args.verbose:
        print(f'\nFILTERED DF:\n{filtered_df}')

    # Melt the DataFrame.
    melted_df = filtered_df.melt(id_vars=['train_ratio', 'label'], var_name='rmse_class', value_name='rmse')

    # Extract class number.
    melted_df['rmse_class'] = melted_df['rmse_class'].str.extract('(\d+)$').astype(int)
    if args.verbose:
        print(f'\nMELTED DF:\n{melted_df}')

    # Find the best model per train_ratio and rmse_class.
    best_models = melted_df.loc[melted_df.groupby(['train_ratio', 'rmse_class'])['rmse'].idxmin()]
    if args.verbose:
        print(f'\nBEST MODELS:\n{best_models}')

    # Create a color mapping for each model.
    unique_labels = best_models['label'].unique()
    colors = ['blue', 'orange', 'green']
    color_mapping = dict(zip(unique_labels, colors))

    # Map colors to the best models.
    best_models['color'] = best_models['label'].map(color_mapping)

    # Plot the bar graph.
    fig, ax = plt.subplots(figsize=(18, 6))

    train_ratios = best_models['train_ratio'].unique()
    num_classes = best_models['rmse_class'].nunique()
    bar_width = 0.05  # Width of each bar
    bar_spacing = 0.025  # Space between bars within the same train ratio

    for i, train_ratio in enumerate(train_ratios):
        subset = best_models[best_models['train_ratio'] == train_ratio]
        for j, (index, row) in enumerate(subset.iterrows()):
            bar_position = i + j * (bar_width + bar_spacing) - (num_classes / 2) * (bar_width + bar_spacing)
            plt.bar(bar_position, row['rmse'], width=bar_width, color=color_mapping[row['label']], zorder=3)
            plt.text(bar_position, -0.01, f'{row["rmse_class"]}', ha='center', va='top', fontsize=10)

    # Create a custom legend.
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_mapping[label]) for label in unique_labels]
    plt.legend(handles, unique_labels, title="Model") #, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add additional information to the plot.
    plt.xticks(np.arange(len(train_ratios)), train_ratios)
    plt.xlabel('Train ratio (%)')
    plt.ylim(-0.04, 0.4)
    plt.ylabel('RMSE per class')
    plt.grid(axis='y', color='gainsboro', linestyle='-', linewidth=0.25, zorder=0)
    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.show()

    return 0





if __name__ == '__main__':
    args = get_args()                                                                                       # Parse and retrieve command-line arguments.
    sys.exit(main(args))                                                                                    # Execute the main function.