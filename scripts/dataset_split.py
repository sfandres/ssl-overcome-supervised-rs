"""Creates the diferent datasets according to the target ratios.

Usage:
    ./stationinfo.py <input_dir> --output_dir <output_dir> --output_ini_name <output_ini_name>

Author:
    A.J. Sanchez-Fernandez - 14/02/2023
"""


import sys
sys.path.append("../")
from utils.computation import Experiment
import splitfolders
import argparse
import sys
import os


def create_arg_parser():
    """Creates and returns the ArgumentParser object."""

    # Parser creation and description.
    parser = argparse.ArgumentParser(
        description=('Python script that divides the target dataset into training, '
                     'validation, and test datasets according to custom splits.')
    )

    parser.add_argument(
        'input_dir',
        type=str,
        help='Path to the input directory where the raw dataset is stored.'
    )

    parser.add_argument(
        'output_dir',
        type=str,
        help='Path to the output directory where the new datasets will be stored.'
    )

    parser.add_argument(
        '--output_ini_name',
        type=str,
        default='NewDataset',
        help="Beginning of each folder's name (default = NewDataset)."
    )

    return parser


# Parser (get arguments).
if __name__ == "__main__":

    # Create parser and get arguments.
    parser = create_arg_parser()
    args = parser.parse_args(sys.argv[1:])

# Show info.
initial_dir_dataset = args.input_dir
print(f'\nPath to the input raw data: {initial_dir_dataset}')

output_dir_datasets = args.output_dir
print(f'\nPath to the output split data: {output_dir_datasets}')

target_dataset_name = args.output_ini_name
print(f'\nOutput folder name: {target_dataset_name}')

# Experiment class for reproducibility.
exp = Experiment()
exp.reproducibility()
print(f'\nSeed used: {exp.seed}')

# Target division according to the train dataset.
train = [.98, .9, .7, .5, .3, .1, .02]
ratios = []

# Iterate over the train splits.
for i, train_s in enumerate(train):

    # Rest: 30% validation and 70% test.
    val_s = round((1 - train_s) * .3, 4)
    test_s = round(1 - train_s - val_s, 4)

    # Stats.
    sum_s = train_s + val_s + test_s
    print(f'{i}- train: {train_s:.3f}   '
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

# Balanced dataset.
# Split val/test with a fixed number of items, e.g. `(100, 100)`, for each set.
# To only split into train-val set, use a single number to `fixed`, i.e., `10`.
# Set 3 values, e.g. `(300, 100, 100)`, to limit the number of training values.

# fixed = (100, 150)
# dataset_name = f'datasets/Sentinel2GlobalLULC_full' \
#                f'-fixed={fixed}' \
#                f'-seed={SEED}'

# splitfolders.fixed(data_dir_initial,
#                    output=dataset_name,
#                    seed=SEED,
#                    fixed=fixed,
#                    oversample=False,  # Does not duplicate train samples.
#                    group_prefix=None,
#                    move=False)  # Default values.