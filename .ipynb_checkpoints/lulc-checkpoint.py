import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory

# Setting seed value
# from https://stackoverflow.com/a/52897216
# generated randomly by running `random.randint(0, 100)` once
SEED = 42
# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value.
# os.environ['PYTHONHASHSEED'] = str(SEED)
# 2. Set the `python` built-in pseudo-random generator at a fixed value.
# random.seed(SEED)
# 3. Set the `numpy` pseudo-random generator at a fixed value.
np.random.seed(SEED)
# 4. Set the `tensorflow` pseudo-random generator at a fixed value.
tf.random.set_seed(SEED)
# 5. More just in case.
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()


def tf_data_loader(path_dir, split_train_val=0.3, shuffle=True, ratio_val_test=3):
    """
    Loads the dataset in TensorFlow format.

    Parameters
    ----------
    split_train_val : int, optional
        Ratio between train and test datasets (percentage for validation and test together).
    shuffle : bool, optional
        Whether the dataset must be shuffled.
    ratio_val_test : int, optional
        Ratio between validation and test after split_train_val.

    Returns
    -------
    lists
        Three tf.data.Dataset objects containing each dataset.
        One list with the class names.
    """
    
    # TensorFlow setup.
    print(tf.__version__)
    # os.environ["CUDA_VISIBLE_DEVICES"] = '-1'    # Enabled: Disable GPU.
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
           
    # Samples info.
    BATCH_SIZE = 32
    IMG_SIZE = (224, 224)

    # Train set.
    train_dataset = image_dataset_from_directory(path_dir,
                                                 label_mode='categorical',
                                                 batch_size=BATCH_SIZE,
                                                 image_size=IMG_SIZE,
                                                 shuffle=shuffle,
                                                 seed=42,
                                                 validation_split=split_train_val,
                                                 subset='training')

    # Validation set.
    validation_dataset = image_dataset_from_directory(path_dir,
                                                      label_mode='categorical',
                                                      batch_size=BATCH_SIZE,
                                                      image_size=IMG_SIZE,
                                                      shuffle=shuffle,
                                                      seed=42,
                                                      validation_split=split_train_val,
                                                      subset='validation')
    
    # Test set.
    ratio_val_test = 3
    val_batches = tf.data.experimental.cardinality(validation_dataset)             # Number of batches in val and test datasets.
    test_dataset = validation_dataset.take(val_batches // ratio_val_test)          # Test dataset.
    validation_dataset = validation_dataset.skip(val_batches // ratio_val_test)    # Validation dataset.
    validation_dataset, test_dataset = test_dataset, validation_dataset            # Swap val and test datasets.
    
    # Dealing with the different classes.
    class_names = train_dataset.class_names
    n_classes = len(class_names)

    # Show some stats.
    print('Classes: ', class_names)
    print('Number of classes: ', n_classes)
    
    # Example.
    plt.figure(figsize=(15, 10))
    for images, labels in train_dataset.take(1):    # Take a batch.
        for i in range(9):                          # Take nine images.
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            # plt.title(class_names[labels[i]])                # No one-hot.
            plt.title(class_names[np.argmax(labels[i])])       # One-hot.
            plt.axis("off")
    
    # Preprocess: batches.
    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
    
    # Checking if dataset is balanced.
    target_dataset = [train_dataset,
                  validation_dataset,
                  test_dataset]

    ds_labels = {}
    ds_samples = {}

    for ds, i in zip(target_dataset, range(len(target_dataset))):
        labels = []
        for x, y in ds:
            # If one hot encoded, then apply argmax.
            labels.append(np.argmax(y, axis=-1))

            # Not one hot encoded.
            # labels.append(y.numpy())

        # Concatenate asuming dataset was batched.
        labels = np.concatenate(labels, axis=0)

        # Count unique labels.
        ds_labels[i], ds_samples[i] = np.unique(labels, return_counts=True)
    
    # Set up the subplot of train-val-test split.
    fig, ax = plt.subplots(1, 3, sharex='col', sharey='row')

    # Set width and height.
    fig.set_figwidth(22)
    fig.set_figheight(10)

    dataset_labels = ['Train',
                      'Validation',
                      'Test']

    for i in range(len(target_dataset)):
        bars = ax[i].barh(class_names, ds_samples[i])    # class_names
        ax[i].bar_label(bars)
        ax[i].set_title(dataset_labels[i])
        ax[i].set_xlim(0, max(ds_samples[i]) + 2000/(1+i*2))

    # Hide x labels and tick labels for top plots
    # and y ticks for right plots.
    for axs in ax.flat:
        axs.label_outer()
        axs.set(xlabel='N samples')
        axs.grid(color='gray', linestyle=':', linewidth=.5)
        # axs.legend(loc='best')

    # Show stats.
    training_samples = len(train_dataset) * BATCH_SIZE
    validation_samples = len(validation_dataset) * BATCH_SIZE
    test_samples = len(test_dataset) * BATCH_SIZE
    total_samples = training_samples + validation_samples + test_samples
    print('Training batches: %d' % tf.data.experimental.cardinality(train_dataset))
    print('Training samples: ' + str(training_samples) + ' (' + str(100*training_samples/total_samples) + '%)')

    print('\nValidation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
    print('Validation samples: ' + str(validation_samples) + ' (' + str(100*validation_samples/total_samples) + '%)')

    print('\nTest batches: %d' % tf.data.experimental.cardinality(test_dataset))
    print('Test samples: ' + str(test_samples) + ' (' + str(100*test_samples/total_samples) + '%)')

    print('\nTotal approx.: ' + str(total_samples))
    
    return (train_dataset, validation_dataset, test_dataset), class_names
