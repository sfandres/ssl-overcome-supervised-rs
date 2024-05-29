

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys


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

    # parser.add_argument('--bar', '-b', choices=['both', 'best', 'diff'],
    #                     help='type of bar plot.')

    parser.add_argument('--ref', '-r', choices=['Random', 'ImageNet'], default='ImageNet',
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

    # Configure matplotlib.
    set_plt()

    # Get task from first item and set target metric and reference.
    task = args.input_df_means_path.split('/')[-1].split('_')[1]
    if task == 'multiclass':
        metric = 'f1_per_class'
        ylabel = 'F1 per class'
    elif task == 'multilabel':
        metric = 'rmse_per_class'
        ylabel = 'RMSE per class'
    print(f"{'Task:'.ljust(16)}{task}") if args.verbose else None

    # Read the input DataFrames.
    df = pd.read_csv(args.input_df_means_path)
    # df_stds = pd.read_csv(args.input_df_means_path.replace('means', 'stds'))

    # Identify the columns that represent the target metric per class (only test columns are considered).
    per_class_columns = [col for col in df.columns if f'test_{metric}' in col]
    if args.verbose:
        print(f'\nPER CLASS COLUMNS:\n{per_class_columns}')

    # Filter the DataFrame to only include the target metric columns and the models with FT in the label.
    filtered_df = df[['epoch', 'train_ratio', 'label'] + per_class_columns]
    filtered_df = filtered_df[filtered_df['label'].str.contains('FT')]
    filtered_df = filtered_df[filtered_df['label'].str.contains('Barlow') | filtered_df['label'].str.contains(args.ref)]
    if args.verbose:
        print(f'\nFILTERED DF (ref: {args.ref}):\n{filtered_df}')

    # Melt the DataFrame.
    melted_df = filtered_df.melt(id_vars=['epoch', 'train_ratio', 'label'], var_name='class', value_name=metric)

    # Extract class number.
    melted_df['class'] = melted_df['class'].str.extract('(\d+)$').astype(int)
    if args.verbose:
        print(f'\nMELTED DF:\n{melted_df}')

    # Sort the DataFrame by train_ratio and class.
    melted_df = melted_df.sort_values(by=['label', 'train_ratio', 'class'], ascending=False)
    # melted_df.to_csv('melted_df.csv', index=False)

    # Find the best model per train_ratio and class.
    if metric == 'rmse_per_class':
        best_models = melted_df.loc[melted_df.groupby(['train_ratio', 'class'])[metric].idxmin()]
    else:
        best_models = melted_df.loc[melted_df.groupby(['train_ratio', 'class'])[metric].idxmax()]
    if args.verbose:
        print(f'\nBEST MODELS:\n{best_models}')

    # Create a color mapping for each model.
    unique_labels = sorted(best_models['label'].unique(), reverse=True)
    colors = ['blue', 'orange', 'green']
    color_mapping = dict(zip(unique_labels, colors))

    # Map colors to the best models.
    best_models['color'] = best_models['label'].map(color_mapping)

    # Plot the bar graph.
    fig, ax = plt.subplots(figsize=(18, 6))

    train_ratios = best_models['train_ratio'].unique()
    num_classes = best_models['class'].nunique()
    bar_width = 0.05  # Width of each bar
    bar_spacing = 0.025  # Space between bars within the same train ratio

    for i, train_ratio in enumerate(train_ratios):
        subset = best_models[best_models['train_ratio'] == train_ratio]
        for j, (index, row) in enumerate(subset.iterrows()):
            bar_position = i + j * (bar_width + bar_spacing) - (num_classes / 2) * (bar_width + bar_spacing)
            plt.bar(bar_position, row[metric], width=bar_width, color=color_mapping[row['label']], zorder=3)
            plt.text(bar_position, -0.01, f"{row['class']}", ha='center', va='top', fontsize=MARKER_SIZE)

    # Create a custom legend.
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_mapping[label]) for label in unique_labels]
    plt.legend(handles, unique_labels, title='Model') #, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add additional information to the plot.
    plt.xticks(np.arange(len(train_ratios)), train_ratios)
    plt.xlabel('Train ratio (%)')
    y_max = max(best_models[metric]) + max(best_models[metric]) * 0.3
    plt.ylim(-0.07, y_max)
    plt.ylabel(ylabel)
    plt.grid(axis='y', color='gainsboro', linestyle='-', linewidth=0.25, zorder=0)
    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()

    # Save figure or show.
    if args.save_fig:
        save_path = os.path.join(
            args.output,
            f'exp_{task}_m={metric}_r={args.ref}.{args.save_fig}'
        )
        fig.savefig(save_path, bbox_inches='tight')
        print(f'Figure saved at {save_path}')
    else:
        plt.show()

    return 0


if __name__ == '__main__':
    args = get_args()                                                                                       # Parse and retrieve command-line arguments.
    sys.exit(main(args))                                                                                    # Execute the main function.