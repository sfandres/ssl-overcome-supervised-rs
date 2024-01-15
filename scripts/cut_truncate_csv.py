import os
import pandas as pd
import sys
import argparse


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

    parser.add_argument('--input_dir', '-i', required=True,
                        help='folder where the csv file(s) are stored.')

    parser.add_argument('--output_dir', '-o', required=True,
                        help='folder to output the new csv file(s).')

    return parser.parse_args(sys.argv[1:])


# Function to copy and truncate CSV files
def copy_and_truncate_csv(source_path, destination_path):
    # Read the CSV file
    df = pd.read_csv(source_path)

    # Truncate to the first 100 rows (epochs 0 to 99)
    truncated_df = df.head(100)

    # Write the truncated data to the destination file
    truncated_df.to_csv(destination_path, index=False)


# Function to process the files in the source directory
def process_files(source_dir, destination_dir):
    # Iterate through the source directory
    for root, dirs, files in os.walk(source_dir):
        # Create corresponding subdirectories in the destination directory
        relative_path = os.path.relpath(root, source_dir)
        destination_subdir = os.path.join(destination_dir, relative_path)
        os.makedirs(destination_subdir, exist_ok=True)

        # Process each file in the current directory
        for file in files:
            source_path = os.path.join(root, file)
            destination_path = os.path.join(destination_subdir, file)

            # Copy and truncate the CSV file
            copy_and_truncate_csv(source_path, destination_path)

    print("Copying and truncating completed.")


# Main function
def main(args):

    for root, dirs, files in os.walk(args.input_dir):
        # Create corresponding subdirectories in the destination directory
        relative_path = os.path.relpath(root, args.input_dir)
        destination_subdir = os.path.join(args.output_dir, relative_path)
        os.makedirs(destination_subdir, exist_ok=True)

        # Process each file in the current directory
        for file in files:
            source_path = os.path.join(root, file)
            destination_path = os.path.join(destination_subdir, file)

            # Copy and truncate the CSV file
            copy_and_truncate_csv(source_path, destination_path)

    print("Copying and truncating completed.")

    return 0


if __name__ == "__main__":

    # Get arguments.
    args = get_args()

    # Main function.
    sys.exit(main(args))
