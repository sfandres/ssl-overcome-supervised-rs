"""Useful functions for graphs.

Usage:
    -

Author:
    A.J. Sanchez-Fernandez - 16/02/2023
"""



import matplotlib as plt
import numpy as np


def simple_bar_plot(ax, data_x, label_x, data_y, label_y, **param_dict):
    """
    Function that returns a simple bar plot.
    https://matplotlib.org/3.5.0/tutorials/introductory/usage.html#the-object-oriented-interface-and-the-pyplot-interface

    Parameters
    ----------
    ax : Axes
        The axes to draw to

    data_x : array
        The x data

    label_x : string
        The x axis label

    data_y : array
        The y data

    label_y : string
        The y axis label

    param_dict : dict
        Dictionary of keyword arguments to pass to ax.plot

    Returns
    -------
    out : list
        List of artists added
    """
    
    # Global options;
    # it could be changed when calling the function.
    # ** means "treat the key-value pairs in the dictionary
    # as additional named arguments to this function call."
    param_dict = {
        'width': .4,
        'zorder': 3,
        'color': '#005f8a',
        'edgecolor': 'black',
        'linewidth': .25,
        'label': 'r'
    }

    # Set position of bar on X axis.
    x_pos = np.arange(len(data_y))

    # Main plot.
    out = ax.bar(x_pos,
                 data_y,
                 **param_dict)

    # Set y limits.
    ax.set_ylim(0, np.max(data_y) + np.max(data_y) * 0.1)

    # Add the numbers on top of each bar.
    for x in range(len(x_pos)):
        ax.text(x=x,
                y=data_y[x],
                s=r'{}'.format(round(data_y[x])),
                ha='center',
                va='bottom')

    # Add xticks on the middle of the group bars.
    ax.set_xticks([r for r in range(len(data_x))])

    # Remaining options.
    ax.set_xlabel(label_x, labelpad=15)
    ax.set_ylabel(label_y, labelpad=15)

    return out


def plot_one_random_sample_dataloader(ax, dataloaders, mean, std, class_names, split, **param_dict):
    """
    Function that shows a random sample from the dataloader.

    Parameters
    ----------
    ax : Axes
        The axes to draw to

    dataloader : PyTorch dataloader
        Dataloader

    mean : dict
        Mean values of the normalization of the samples

    std : dict
        Std values of the normalization of the samples

    class_names : list
        Map the idx into labels

    split : str
        Train, val, or test

    param_dict : dict
        Dictionary of keyword arguments to pass to ax.plot

    Returns
    -------
    out : list
        List of artists added
    """

    # Accessing Data and Targets in a PyTorch DataLoader.
    for i, (images, labels) in enumerate(dataloaders[split]):
        print(f'Batch-->images: {images.shape}')
        print(f'Batch-->labels: {labels.shape}')
        label = labels[0]
        img = images[0].numpy().transpose((1, 2, 0))
        img = std[split] * img + mean[split]
        img = np.clip(img, 0, 1)
        break

    # Set title.
    ax.set_title(f'Label: {int(label)} ({class_names[label]})')
    ax.set_aspect('auto')

    # Main plot.
    out = ax.imshow(img)

    return out
