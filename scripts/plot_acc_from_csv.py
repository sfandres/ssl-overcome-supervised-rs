"""Prints the plot of the accuracy values
   obtained during training from an csv file.

Author:
    A.J. Sanchez-Fernandez - 31/05/2023
"""


import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def get_args() -> argparse.Namespace:
    """
    Parse and retrieve command-line arguments.

    Returns:
        An 'argparse.Namespace' object containing the parsed arguments.
    """

    # Get arguments.
    parser = argparse.ArgumentParser(
        description='Script that plots the accuracy values from .csv file.'
    )

    parser.add_argument('--input', '-i', nargs='+', required=True,
                        help='csv file(s) to plot.')

    return parser.parse_args(sys.argv[1:])


def main(args):

    # Iterate over each CSV file.
    for filename in args.input:

        # Read the CSV file into a pandas DataFrame.
        df = pd.read_csv(filename)
                
        # Extract the header values.
        headers = list(df.columns)
        x_label = headers[0]
        y_label = headers[2]

        # Extract the first and second columns.
        x = df[x_label]
        y = df[y_label]

        # Plot the data.
        plt.plot(x, y, label=filename)
    
    # Add labels and title to the plot.
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title('CSV Data')
    
    # Display the legend and show the plot.
    plt.legend()
    plt.show()

    return 0


if __name__ == "__main__":

    # Get arguments.
    args = get_args()

    # Main function.
    sys.exit(main(args))
