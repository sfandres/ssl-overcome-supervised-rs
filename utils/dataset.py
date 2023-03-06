"""Useful functions to manage the datasets.

Usage:
    -

Author:
    A.J. Sanchez-Fernandez - 02/03/2023
"""


import os
import ast
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image

#--------------------------
# LOADING DATASETS
#--------------------------

class AndaluciaDataset(Dataset):
    """Sentinel2AndaluciaLULC dataset."""

    def __init__(self, root_dir, level, split, transform=None, target_transform=None, verbose=True):
        """
        Args:
            root_dir (str): Root (parent) directory.
            level (str): Level of the dataset (N1 or N2).
            split (str): Train, validation or test splits.
            transform (callable, optional): Optional transform to be
            applied on an image.
            target_transform (callable, optional): Optional transform to be
            applied on a label.
            verbose (callable, optional): Enables show info.
        """
        # Build paths.
        self.root_dir = root_dir
        self.level_dir = os.path.join(self.root_dir, level)
        self.images_dir = os.path.join(self.level_dir,
                                       os.path.join('RGB Images', split))

        # Get csv file.
        self.csv_file_path = os.path.join(self.level_dir,
                                          os.path.join('CSV',
                                                       f'csv_{split}.csv'))
        self.csv_dataset_info = pd.read_csv(self.csv_file_path)

        # Get transforms if applicable.
        self.transform = transform
        self.target_transform = target_transform

        # Get classes and mapping.
        classes = self._find_classes(level)
        self.classes, self.class_to_idx, self.idx_to_class = classes

        # Print info.
        if verbose:
            self._show_info()

    def _show_info(self):
        """ Shows data regarding the dataset. """
        print('-----------Andalucia dataset info-----------')
        print(f'Root/Parent folder: {self.root_dir}')
        print(f'Level folder:       {self.level_dir}')
        print(f'Images folder:      {self.images_dir}')
        print(f'Path to csv file:   {self.csv_file_path}')
        print('Instance variables: .classes, .class_to_idx, .idx_to_class')
        print(f'Number of samples:  {self.__len__()}')
        print(f'Number of classes:  {len(self.classes)}')
        print('Labels:')
        print(self.classes)
        print(f'--------------------------------------------\n')

    def _find_classes(self, level):
        """
        Finds the class folders in a dataset.
        Args:
            level (str): Level of the dataset (N1 or N2).
        Returns:
            tuple: (classes, class_to_idx, idx_to_class) where classes are
            relative to (dir), and class_to_idx and idx_to_class are dictionaries.
        """
        # Get csv with class names.
        path_to_csv_dict = os.path.join(self.root_dir,
                                        'N1_and_N2_Dictionnary.xlsx')
        csv_dict = pd.read_excel(path_to_csv_dict, header=None)

        # Get the classes.
        if level == 'Level_N1':
            classes = csv_dict.iloc[2:6, 1].to_numpy()
        elif level == 'Level_N2':
            classes = csv_dict.iloc[10:21, 1].to_numpy()

        # Create the dictionaries.
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        idx_to_class = {v: k for k, v in class_to_idx.items()}

        return classes, class_to_idx, idx_to_class

    def __len__(self):
        """ Returns the number of samples. """
        return len(self.csv_dataset_info)

    def __getitem__(self, idx):
        """
        Supports the indexing of samples.
        Args:
            idx (int): index of the sample.
        Returns:
            sample (dict): sample returned.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get image name (with idx and column 'filename').
        img_name = os.path.join(self.images_dir,
                                self.csv_dataset_info.iloc[idx, 3])

        # Load image.
        img = Image.open(img_name)

        # Get ground-truth abundances (multi-label).
        abundances = self.csv_dataset_info.iloc[idx, 5:15]
        abundances = torch.FloatTensor(abundances)

        # Apply transforms if requested.
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            abundances = self.target_transform(abundances)

        return img, abundances


#--------------------------
# OTHER USEFUL FUNCTIONS
#--------------------------

def list_subdirs_fullpath(input_path):
    """
    Takes the "input_path" and creates a list of
    the subdirectories with the root path included.

    Args:
        input_path: the target directory.

    Returns:
        A sorted list of the subdirectories with the full path. 
    """

    return sorted([os.path.join(input_path, x)
                   for x in os.listdir(input_path)])


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


def load_mean_std_values(input_path):
    """
    Takes the "input_path" and loads the mean
    and std values from the txt file.

    Args:
        input_path: path to the directory where the dataset is saved.

    Returns:
        mean: dictionary holding the mean of the samples
            per dimension and split.
        std: dictionary holding the standard deviation of
            the samples per dimension and split.
    """

    # Initialization.
    splits = ['train', 'val', 'test']
    filename = 'dataset_mean_std.txt'
    mean, std = {}, {}

    # Create path to the txt file.
    filepath = os.path.join(input_path, filename)

    # Read the values by transforming from str to list
    # and reading only the target characters.
    with open(filepath) as f:
        lines = f.readlines()
        mean['train'] = ast.literal_eval(lines[1][7:-2])
        std['train'] = ast.literal_eval(lines[2][7:-2])
        mean['val'] = ast.literal_eval(lines[4][7:-2])
        std['val'] = ast.literal_eval(lines[5][7:-2])
        mean['test'] = ast.literal_eval(lines[7][7:-2])
        std['test'] = ast.literal_eval(lines[8][7:-2])

    return mean, std


def load_dataset_based_on_ratio(input_path, name, ratio):
    """
    Takes the "input_path", as well as the target "name"
    and "ratio" of the dataset and returns the full path,
    mean and std values.

    Args:
        input_path: directory where the datasets are stored.
        name: name of the target dataset (directory's name as well).
        ratio: ratio of the target dataset according to train, val, and test.

    Returns:
        split_path: full path to the target split.
        mean: dictionary holding the mean of the samples.
        std: dictionary holding the standard deviation of the samples.
    """

    # Get target dataset directory.
    dataset_path = os.path.join(input_path, name)
    print(f'Path to dataset folder: {dataset_path}')

    # Get the splits in the dataset directory.
    splits_path = list_subdirs_fullpath(dataset_path)

    # Iterate over the splits.
    for split_path in splits_path:
        
        # Get the current ratio.
        split_ratio = split_path[
            split_path.index("("):split_path.index(")")+1
        ]

        # Compare both ratios.
        if split_ratio == ratio:
            found = True
            break

    # Handling errors: finding the path to the target split.
    try:
        print(f'Dataset found based on the given ratio: {found}')
        print(f'\nPath to split folder: {split_path}')
    except:
        raise Exception('Dataset split not found based on the given ratio. '
                        'Please check the ratio.')

    # Handling errors: loading mean and std from file.
    try:
        mean, std = load_mean_std_values(split_path)
        print(f'Mean loaded from .txt: {mean}')
        print(f'Std loaded from .txt: {std}')
    except:
        raise Exception('Error loading the mean and std files for the target dataset. '
                        'Please check the files.')

    return split_path, mean, std


def show_one_batch(axes, num_cols, dataloader, idx_to_class, batch_id=0, param_dict={}):
    """
    Takes the "dataloader" and creates a figure using "axes",
    "num_cols", and "param_dict" displaying all the images in
    the "batch_id" batch. Each subplot shows the class name with
    the greatest abundance using the "idx_to_class" dict.

    Example:
        show_one_batch(axes, dataloader, dataset.idx_to_class,
                       batch_id=0, param_dict={'alpha':0.25})

    Args:
        axes (matplotlib.axes): An Axes object encapsulates all
        the elements of an individual (sub-)plot in the figure.
        num_cols (int): Number of columns in the figure.
        dataloader (PyTorch dataloader): The dataloader.
        idx_to_class (dict): Dictionary that maps ids to class' names. 
        batch_id (int, optional): Item (batch) to be displayed.
        param_dict (dict, optional): Dictionary of kwargs to
        pass to ax.plot.

    Returns:
        out (list): List of artists added.
    """
    
    # Iterate over batches.
    for i, batch in enumerate(dataloader):

        # Batch is a tuple of inputs and targets.
        inputs, labels = batch

        # Display images (only target batch).
        if i == batch_id:
            for j in range(dataloader.batch_size):

                # Get image and label from batch.
                image = inputs[j]
                label = labels[j]

                # Get class name with the greatest abundance (multi-label for now).
                class_name = idx_to_class[int(torch.argmax(label))]

                # Convert image from tensor to numpy array.
                image = torch.permute(image, (1, 2, 0))  # [C, H, W] -> [H, W, C]

                # Display image in subplot.
                ax = axes[j // num_cols, j % num_cols]
                ax.set_title(f'Max: {class_name}')
                ax.axis('off')
                out = ax.imshow(image, **param_dict)

            return out
