import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import math
import torch
import random
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px


class Experiment:
    """
    Experimental setup.

    Attributes:
        batch_size: Size of the batch of samples to train.
        epochs: Number of epochs to train the model.
        input_size: Size of the input images.
        num_workers: Number of threads for the dataloader.
        seed: Seed for reproducibility purposes.
    """
    def __init__(self, batch_size=128, epochs=10,
                 input_size=224, num_workers=8, seed=42):
        """Inits Experiment with default parameters."""
        self.batch_size = batch_size
        self.epochs = epochs
        self.input_size = input_size
        self.num_workers = num_workers
        self.seed = seed

    def reproducibility(self):
        """Makes the experiments reproducible."""
        # Seed torch and numpy.
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Enable CUDNN deterministic mode.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Issues a warning if it is not met.
        torch.use_deterministic_algorithms(True)
        
        self.g = torch.Generator()
        self.g.manual_seed(self.seed)
        
        # Enable deterministic behavior using external GPU.
        # %env CUBLAS_WORKSPACE_CONFIG=:4096:8
        # %env CUBLAS_WORKSPACE_CONFIG=:16:8
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
        
    def seed_worker(self, worker_id):
        """Seed for the workers of the dataloaders (reproducibility)."""
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)


def simple_bar_plot(data_x, data_y, x_axis_label=r'x axis label',
                    y_axis_label=r'y axis label',
                    plt_name='simple_bar_plot',
                    fig_size=(15, 5), save=False):
    """
    Takes the "data_x" and "data_y" 
    and creates a bar plot.

    Args:
        data_x: data x axis.
        data_y: height of the bars.
        x_axis_label: label of the x axis.
        y_axis_label: label of the y axis.
        plt_name: name of the plot.
        fig_size: size of the figure.
        save: whether the plot is saved.

    Returns:
        none
    """

    # Barplot.
    fig, ax = plt.subplots(figsize=fig_size)

    # Set bar width.
    barWidth = 0.4

    # Set y values.
    y = data_y

    # Set position of bar on X axis.
    x_pos = np.arange(len(y))

    # Make the plot.
    plt.bar(x_pos,
            y,
            width=barWidth,
            zorder=3,
            color='#005f8a',
            edgecolor='black',
            linewidth=.25,
            label=r'')

    plt.ylim(0, np.max(y) + np.max(y) * 0.1)

    # Add the numbers on top of each bar.
    for x in range(len(x_pos)):
        plt.text(x=x,
                 y=y[x],
                 s=r'{}'.format(round(y[x])),
                 ha='center',
                 va='bottom',
                 fontsize=13)

    # Add xticks on the middle of the group bars.
    plt.xticks([r for r in range(len(data_x))], data_x, fontsize=15)
    plt.yticks(fontsize=15)
                 
    # Set grid.
    # plt.grid(axis='y',
    #          color='gainsboro',
    #          linestyle='-',
    #          linewidth=0.2,
    #          zorder=0)

    # Remaining options.
    plt.xlabel(x_axis_label, labelpad=15, fontsize=17)
    plt.ylabel(y_axis_label, labelpad=15, fontsize=17)

    # Adjust margins.
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.15)

    # Adjust X ticks.
    # fig.autofmt_xdate()

    # Plot.
    plt.show()

    # Save resulting figure in two formats.
    if save:
        fig.savefig(f'figures/{plt_name}.png',
                    bbox_inches='tight')
        fig.savefig(f'figures/{plt_name}.pdf',
                    bbox_inches='tight')


def get_mean_std_dataloader(dataloader):
    """
    Takes the "dataloader" and computes
    the mean and std of the entire dataset.

    Args:
        dataloader: PyTorch DataLoader.

    Returns:
        mean: mean of the samples per dimension.
        std: standard deviation of the samples per dimension.
    """

    # Initialization.
    channels_sum, channels_squares_sum, num_batches = 0, 0, 0

    # Loop over the batches.
    for data, _ in dataloader:

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
    Takes the "directory" and creates a list of
    the subdirectories with the root path included.

    Args:
        directory: the target directory.

    Returns:
        A list of the subdirectories with the full path. 
    """

    return sorted([os.path.join(directory, x)
                   for x in os.listdir(directory)])


def create_confusion_matrix(model, dataloader, device, class_names):
    """
    Takes the "model" and "dataloader" and computes
    the confusion matrix required to calculate and
    plot the F1-score using the "device".

    Args:
        model: trained model.
        dataloader: PyTorch DataLoader.
        device: 'cpu' or 'cuda' for PyTorch.

    Returns:
        conf_mat: resulting confusion matrix.
        class_accuracy: accuracy results per class.
    """

    # Initialize the prediction and label lists (tensors).
    pred_list = torch.zeros(0, dtype=torch.long, device=device)
    label_list = torch.zeros(0, dtype=torch.long, device=device)

    # Since we're not training, we don't need to calculate
    # the gradients for our outputs with torch.no_grad():
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader):

            # Dataset.
            inputs, labels = data[0].to(device), data[1].to(device)

            # Predict labels.
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            # Append batch prediction results.
            pred_list = torch.cat([pred_list, predicted.view(-1)])
            label_list = torch.cat([label_list, labels.view(-1)])

    # Copy back to cpu.
    pred_list = pred_list.to('cpu')
    label_list = label_list.to('cpu')

    # Confusion matrix and per-class accuracy.
    conf_mat = confusion_matrix(label_list.numpy(), pred_list.numpy())
    class_accuracy = 100 * conf_mat.diagonal() / conf_mat.sum(1)
    
    # Create dataframe from confusion matrix.
    df_conf_mat = pd.DataFrame(conf_mat,
                               index=class_names,
                               columns=class_names).astype(int)

    # Plot seaborn confusion matrix.
    fig = plt.figure(figsize=(22, 12))
    sns_plot = sns.heatmap(df_conf_mat,
                           annot=True,
                           xticklabels=class_names,
                           yticklabels=class_names,
                           fmt="d")
    plt.show()

    # Print the classification report.
    print(classification_report(pred_list,
                                label_list,
                                target_names=class_names))

    return conf_mat, class_accuracy


def pca_computation_and_plot(embeddings, labels, plot='all'):
    """
    Takes the "embeddings" and "labels" and
    computes and plots the pca method.

    Args:
        embeddings: generated by the model.
        labels: labels of the samples.
        plot: "2d", "3d", "23d", "3d-plotly", "all".

    Returns:
        none
    """

    # PCA computation.
    pca = PCA(n_components=3)
    pca_results = pca.fit_transform(embeddings)
    print(f'Explained variation per principal component: \
          {pca.explained_variance_ratio_}')

    # Create dataframe with the resulting data.
    df = pd.DataFrame()
    df['pca_x'] = pca_results[:, 0]
    df['pca_y'] = pca_results[:, 1]
    df['pca_z'] = pca_results[:, 2]
    df['labels'] = labels

    # 2-D plot.
    if plot == '3d' or plot == "23d" or plot == 'all':
        plt.figure(figsize=(10, 10))
        sns.scatterplot(
            x='pca_x',
            y='pca_y',
            hue='labels',
            palette=sns.color_palette('hls', 29),
            data=df,
            legend='full',
            alpha=0.9
        )
        plt.show()

    # 3-D plot with matplotlib.
    if plot == '3d' or plot == "23d" or plot == 'all':
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(
            xs=df['pca_x'],
            ys=df['pca_y'],
            zs=df['pca_z'],
            c=df['labels'],
            cmap='tab10'
        )
        ax.set_xlabel('pca_x')
        ax.set_ylabel('pca_y')
        ax.set_zlabel('pca_z')
        plt.show()

    # 3-D plot with pyplot.
    if plot == '3d-plotly' or plot == 'all':
        fig = px.scatter_3d(df, x='pca_x',
                            y='pca_y', z='pca_z',
                            color='labels',
                            width=1000, height=800)  # symbol='labels'

        fig.update_traces(marker=dict(size=3))

        # Move colorbar.
        # fig.update_layout(coloraxis_colorbar=dict(yanchor="top", y=1, x=0,
        #                                           ticks="outside",
        #                                           ticksuffix=""))

        fig.show()