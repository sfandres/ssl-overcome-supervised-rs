"""Creates the diferent datasets according to the target ratios.

Usage: dataset_split.py [-h] [--output_ini_name OUTPUT_INI_NAME] [--train_ratios TRAIN_RATIOS [TRAIN_RATIOS ...]]
                        [--val_ratio VAL_RATIO]
                        input_path output_path

Python script that divides the target dataset into training, validation, and test datasets according to custom
splits.

positional arguments:
  input_path            path to the input directory where the raw dataset is stored (one class per folder).
  output_path           path to the output directory where the new datasets will be stored.

options:
  -h, --help            show this help message and exit
  --output_ini_name OUTPUT_INI_NAME, -n OUTPUT_INI_NAME
                        beginning of each folder's name (default = 'NewDataset').
  --train_ratios TRAIN_RATIOS [TRAIN_RATIOS ...], -trs TRAIN_RATIOS [TRAIN_RATIOS ...]
                        train ratios of the splits per unit separated by spaces (example: .9 .8 .7).
  --val_ratio VAL_RATIO, -vr VAL_RATIO
                        val ratio of the splits per unit (default = 0.25).

Author:
    A.J. Sanchez-Fernandez - 15/03/2023
"""


import sys
import os
import argparse
import ast
import splitfolders
sys.path.append("../")
from utils.computation import Experiment


def create_arg_parser():
    """Creates and returns the ArgumentParser object."""

    # Parser creation and description.
    parser = argparse.ArgumentParser(
        description=('Python script that divides the target dataset into training, '
                     'validation, and test datasets according to custom splits.')
    )

    parser.add_argument(
        'input_path',
        type=str,
        help='path to the input directory where the raw dataset is stored (one class per folder).'
    )

    parser.add_argument(
        'output_path',
        type=str,
        help='path to the output directory where the new datasets will be stored.'
    )

    parser.add_argument(
        '--output_ini_name',
        '-n',
        type=str,
        default='NewDataset',
        help="beginning of each folder's name (default = 'NewDataset')."
    )

    parser.add_argument(
        '--train_ratios',
        '-trs',
        nargs='+',
        type=str,
        default='[.9]',
        help="train ratios of the splits per unit separated by spaces (example: .9 .8 .7)."
    )

    parser.add_argument(
        '--val_ratio',
        '-vr',
        type=float,
        default=.25,
        help="val ratio of the splits per unit (default = 0.25)."
    )

    return parser

def main():
    """"Main function."""

    # Create parser and get arguments.
    parser = create_arg_parser()
    args = parser.parse_args(sys.argv[1:])

    # Show info.
    initial_dir_dataset = args.input_path
    print(f'\nPath to the input raw data:\t{initial_dir_dataset}')

    output_dir_datasets = args.output_path
    print(f'Path to the output split data:\t{output_dir_datasets}')

    target_dataset_name = args.output_ini_name
    print(f'Output folder name:\t\t{target_dataset_name}')

    # Experiment class for reproducibility.
    exp = Experiment()
    exp.reproducibility()
    print(f'\nSeed used:\t   {exp.seed}')

    # Target division according to the train dataset.
    # Get train ratios from command line arguments.
    train_ratios = args.train_ratios
    if isinstance(train_ratios, list):
        train_ratios = [float(num) for num in train_ratios]
    else:
        train_ratios = [float(num) for num in ast.literal_eval(train_ratios)]
    print(f'Train ratios:\t   {train_ratios}')

    val_ratio = args.val_ratio
    print(f'Validation ratio:  {val_ratio}\n')

    # Iterate over the train splits.
    ratios = []
    for i, train_s in enumerate(train_ratios):

        # Rest: 25% validation and 75% test.
        val_s = round((1 - train_s) * val_ratio, 4)
        test_s = round(1 - train_s - val_s, 4)

        # Stats.
        sum_s = train_s + val_s + test_s
        print(f'[{i}] --> train: {train_s:.3f}   '
            f'val: {val_s:.4f}   '
            f'test: {test_s:.4f}   '
            f'sum: {sum_s}')

        # Save the ratio in a list.
        ratios.append((train_s,
                    val_s,
                    test_s))

    # Print the list of ratios to be used.
    print(f'{ratios}\n')

    # Split with a ratio.
    # To only split into training and validation set,
    # set a tuple to `ratio`, i.e, `(.8, .2)`.
    # ratio = (.7, .1, .2)  # (.01, .01, .98)

    # Iterate over the ratios.
    for ratio in ratios:

        # Changing the format of the ratio to an appropriate one.
        str_ratio = f'({ratio[0]:.3f},{ratio[1]:.4f},{ratio[2]:.4f})'
        
        # Creating dataset's name.
        dataset_name = (f'{target_dataset_name}'
                        f'-ratio={str_ratio}'
                        f'-seed={exp.seed}')

        print(f'Building dataset: {dataset_name}...')

        # Split the dataset (default values).
        splitfolders.ratio(initial_dir_dataset,
                        output=os.path.join(output_dir_datasets,
                                            dataset_name),
                        seed=exp.seed,
                        ratio=ratio,
                        group_prefix=None,
                        move=False)

        print('Successfully created!\n')

    return 0


# Call the main function to execute the program.
if __name__ == '__main__':
    sys.exit(main())
