"""Prints the plot of the accuracy values
   obtained during training from an csv file.

Usage: plot_loss_lr_from_output.py [-h] --input_file INPUT_FILE [--graph {matplotlib,plotly}]

Script that plots the average losses and learning rates from .out file.

options:
  -h, --help            show this help message and exit
  --input_file INPUT_FILE, -i INPUT_FILE
                        path to input file (.out).
  --graph {matplotlib,plotly}, -g {matplotlib,plotly}
                        graphing library to use for plotting (default: matplotlib).

Author:
    A.J. Sanchez-Fernandez - 30/05/2023
"""


import matplotlib.pyplot as plt
import sys
import argparse
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def get_args() -> argparse.Namespace:
    """
    Parse and retrieve command-line arguments.

    Returns:
        An 'argparse.Namespace' object containing the parsed arguments.
    """

    # Get arguments.
    parser = argparse.ArgumentParser(
        description='Script that plots the average losses and learning rates from .out file.'
    )

    # General arguments.
    parser.add_argument('--input_file', '-i', type=str, required=True,
                        help='path to input file (.out).')

    parser.add_argument('--graph', '-g', type=str, default='matplotlib',
                        choices=['matplotlib', 'plotly'],
                        help='graphing library to use for plotting (default: matplotlib).')

    return parser.parse_args(sys.argv[1:])


def main(args):

    # Initialization.
    train_losses = {}      # Dictionary to store train losses per epoch.
    learning_rates = []    # List to store learning rates.
    first_epoch = 0        # Initialize the first epoch.
    last_epoch = 0         # Initialize the last epoch.

    # Read the output file.
    with open(args.input_file, 'r') as file:
        for line in file:
            if 'Train loss:' in line:
                # Split the line using the '|' delimiter.
                parts = line.split('|')

                # Extract the GPU, epoch, and train loss values from the line.
                gpu = int(parts[0].strip().split(':')[1].strip('[]'))
                epoch = int(parts[1].strip().split(':')[1].strip().strip('[]'))
                loss = float(parts[2].split(':')[1].strip())

                # Store the train loss in the dictionary under the corresponding epoch key.
                train_losses.setdefault(epoch, []).append(loss)

                # Update the first and last epoch values.
                first_epoch = min(first_epoch, epoch)
                last_epoch = max(last_epoch, epoch)
            
            elif 'Learning rate:' in line:
                # Split the line using the ':' delimiter.
                parts = line.split(':')

                # Check if the learning rate value has already been saved.
                lr = float(parts[1].strip())
                if lr not in learning_rates:
                    learning_rates.append(lr)

    # Create a range of epochs.
    epochs = range(first_epoch, last_epoch + 1)

    # Calculate average loss per epoch.
    average_losses = [sum(train_losses.get(epoch, [])) / 4 for epoch in epochs]

    # The last lr value has appeared and should not be counted.
    if len(learning_rates) - len(epochs) == 1:
        learning_rates = learning_rates[:-1]

    # Graph.
    if args.graph == 'matplotlib':

        # Create subplots for average losses and learning rates.
        fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

        # Plot average losses over epochs.
        axs[0].plot(epochs, average_losses, marker='o')
        axs[0].set_ylabel('Average train loss')
        axs[0].set_title(args.input_file)
        axs[0].grid(True)

        # Plot learning rates over epochs.
        axs[1].plot(epochs, learning_rates, marker='o', color='orange')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Learning rate')
        axs[1].grid(True)

        # Show the figure.
        plt.tight_layout()
        plt.show()

    elif args.graph == 'plotly':

        # Create subplots with 2 rows and 1 column.
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.02, subplot_titles=(args.input_file, ))

        # Add trace for average losses.
        fig.add_trace(go.Scatter(x=list(epochs), y=average_losses,
                                 mode='lines+markers', name='Average train loss'),
                                 row=1, col=1)

        # Add trace for learning rates.
        fig.add_trace(go.Scatter(x=list(epochs), y=learning_rates,
                                 mode='lines+markers', name='Learning rate',
                                 line=dict(color='orange')), row=2, col=1)

        # Update layout.
        fig.update_xaxes(title_text='Epoch', row=2, col=1)
        fig.update_yaxes(title_text='Average train loss', row=1, col=1)
        fig.update_yaxes(title_text='Learning rate', row=2, col=1)
        # fig.update_yaxes(tickformat="none")

        # Show the figure.
        fig.show()

    return 0


if __name__ == "__main__":

    # Get arguments.
    args = get_args()

    # Main function.
    sys.exit(main(args))
