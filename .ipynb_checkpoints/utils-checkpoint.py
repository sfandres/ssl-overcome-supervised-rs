import torch
import os


def get_mean_std_dataloader(loader):
    """
    Takes the dataloader and computes
    the mean and std of the entire dataset.
    
    Args:
        loader: PyTorch DataLoader.

    Returns:
        mean: mean of the samples per dimension.
        std: standard deviation of the samples per dimension.
    """

    # Initialization.
    channels_sum, channels_squares_sum, num_batches = 0, 0, 0

    # Loop over the batches.
    for data, _ in loader:

        # Compute the mean in the given dimensions (not channel).
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squares_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches +=1

    # Final computation.
    mean = channels_sum / num_batches
    std = (channels_squares_sum / num_batches - mean**2)**0.5

    return mean, std


def listdir_fullpath(directory):
    """
    Takes the directory and creates a list of
    the subdirectories with the root path included.

    Args:
        directory: the target directory.

    Returns:
        A list of the subdirectories with the full path. 
    """

    return sorted([os.path.join(directory, x)
                   for x in os.listdir(directory)])
