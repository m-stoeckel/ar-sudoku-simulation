import warnings

import keras
import numpy as np
from matplotlib import pyplot as plt

from digit import CuratedCharactersDataset
from digit.dataset import CharacterDataset, PrerenderedDigitDataset, ClassSeparateMNIST, ConcatDataset
from image.image_transforms import RandomPerspectiveTransform


@DeprecationWarning
class DigitDataGenerator(keras.utils.Sequence):

    def __init__(
            self,
            dataset: CharacterDataset,
            batch_size=32,
            shuffle=True,
            **kwargs
    ):
        """Initialization"""
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.machine_indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(self.machine_indices)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.dataset) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indices = self.machine_indices[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        x, y = self.__data_generation(indices)
        x = x.astype(np.float32)

        return x[:, :, :, np.newaxis], y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.machine_indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(self.machine_indices)

    def __data_generation(self, indices):
        """Generates data containing batch_size samples"""
        x = self.dataset.train_x[indices]
        y = self.dataset.train_y[indices]

        return x, keras.utils.to_categorical(y, num_classes=9)


class BalancedDataGenerator(keras.utils.Sequence):
    def __init__(
            self,
            machine_digits: CharacterDataset,
            handwritten_digits: CharacterDataset,
            out_dataset: CharacterDataset,
            batch_size=32,
            shuffle=True,
            flatten=False,
            resolution=28
    ):
        self.machine_dataset = machine_digits
        self.handwritten_dataset = handwritten_digits
        self.out_dataset = out_dataset

        self.shuffle = shuffle
        self.flatten = flatten
        self.resolution = resolution
        self.batch_size = batch_size
        if self.batch_size % 3 != 0:
            warnings.warn("The batch size should be divisible by three!")

        self.num_classes = 20

        self.machine_indices = np.arange(self.machine_dataset.train_y.shape[0])
        self.handwritten_indices = np.arange(self.handwritten_dataset.train_y.shape[0])
        self.out_indices = np.arange(self.out_dataset.train_y.shape[0])

        if self.shuffle:
            np.random.shuffle(self.machine_indices)
            np.random.shuffle(self.handwritten_indices)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(len(self.machine_dataset) * 2 / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data
        :param index: The batch number.
        :return: Returns a tuple of a 4-dimensional ndarray and the class-categorical label ndarray
        """
        # Generate indexes of the batch
        mini_batch_size = int(self.batch_size / 3)
        mini_batch_size_last = self.batch_size - 2 * mini_batch_size

        machine_indices = self.machine_indices[index * mini_batch_size:(index + 1) * mini_batch_size]
        handwritten_indices = self.handwritten_indices[index * mini_batch_size:(index + 1) * mini_batch_size]
        out_indices = self.out_indices[index * mini_batch_size_last:(index + 1) * mini_batch_size_last]

        # Generate data
        xd, yd = self.__machine_data_generation(machine_indices)
        xm, ym = self.__handwritten_data_generation(handwritten_indices)
        xo, yo = self.__out_data_generation(out_indices)

        # Stack data and convert images to float
        x = np.vstack((xd, xm, xo)).astype(np.float32)
        y = np.vstack((yd, ym, yo))

        # Scale x to 0..1
        x /= 255.

        if self.flatten:
            x = x.reshape(-1, self.resolution ** 2)
        else:
            x = x[:, :, :, np.newaxis]

        return x, y

    def on_epoch_end(self):
        self.machine_indices = np.arange(len(self.machine_dataset))
        self.handwritten_indices = np.arange(len(self.handwritten_dataset.train_y))
        if self.shuffle:
            np.random.shuffle(self.machine_indices)
            np.random.shuffle(self.handwritten_indices)

    def __machine_data_generation(self, indices, ):
        """
        Generates data containing batch_size samples. Machine written digits have class <digit>.
        :param indices: The indices to select
        :return: A tuple of a digit array and a class categorical array
        """
        X = self.machine_dataset.train_x[indices]
        y = self.machine_dataset.train_y[indices]

        return X, keras.utils.to_categorical(y, num_classes=self.num_classes)

    def __handwritten_data_generation(self, indices, ):
        """
        Generates data containing batch_size samples. MNIST digits have class <digit> + 9.
        :param indices: The indices to select
        :return: A tuple of a digit array and a class categorical array
        """
        X = self.handwritten_dataset.train_x[indices]
        y = self.handwritten_dataset.train_y[indices]

        return X, keras.utils.to_categorical(y, num_classes=self.num_classes)

    def __out_data_generation(self, indices, ):
        """
        Generates data containing batch_size samples. MNIST digits have class <digit> + 9.
        :param indices: The indices to select
        :return: A tuple of a digit array and a class categorical array
        """
        X = self.out_dataset.train_x[indices]
        y = self.out_dataset.train_y[indices]

        return X, keras.utils.to_categorical(y, num_classes=self.num_classes)


def test_generator():
    prerendered_dataset = PrerenderedDigitDataset(digits_path="../datasets/digits/")
    prerendered_dataset.add_transforms(RandomPerspectiveTransform())
    # dataset.add_transforms(RandomPerspectiveTransformX())
    # dataset.add_transforms(RandomPerspectiveTransformY())
    prerendered_dataset.apply_transforms(keep=False)
    prerendered_dataset.resize(28)

    # handwritten non-digits
    curated_out = CuratedCharactersDataset(
        digits_path="../datasets/curated/",
        load_chars="0abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.!?"
    )
    curated_out.add_transforms(RandomPerspectiveTransform())
    curated_out.apply_transforms(keep=False)
    curated_out.resize(28)

    # handwritten digits
    curated_digits = CuratedCharactersDataset(digits_path="../datasets/curated/", load_chars="123456789")
    curated_digits.add_transforms(RandomPerspectiveTransform())
    curated_digits.apply_transforms(keep=False)
    curated_digits.resize(28)

    # mnist digits
    mnist = ClassSeparateMNIST()
    concat_dataset = ConcatDataset([mnist, curated_digits])

    batch_size = 12
    d = BalancedDataGenerator(
        prerendered_dataset, concat_dataset, curated_out,
        batch_size=batch_size,
        shuffle=True,
        resolution=28
    )
    img_l = []
    for i in range(batch_size):
        X, y = d[i]
        img_l.append(np.hstack([img for img in X.squeeze()]))
    plt.figure(figsize=(4, 4))
    plt.imshow(np.vstack(img_l), cmap="gray")
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    test_generator()