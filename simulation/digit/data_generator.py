import warnings
from typing import Tuple

import keras
import numpy as np

from simulation.digit.dataset import CharacterDataset


@DeprecationWarning
class DigitDataGenerator(keras.utils.Sequence):

    def __init__(
            self,
            dataset: CharacterDataset,
            batch_size=32,
            shuffle=True
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
    TRUNCATE = 0
    REPEAT = 1

    def __init__(
            self,
            machine_digits: (np.ndarray, np.ndarray),
            handwritten_digits: (np.ndarray, np.ndarray),
            out_dataset: (np.ndarray, np.ndarray),
            batch_size=32,
            shuffle=True,
            flatten=False,
            data_align=TRUNCATE
    ):
        """
        TODO: Comment
        :param machine_digits:
        :type machine_digits:
        :param handwritten_digits:
        :type handwritten_digits:
        :param out_dataset:
        :type out_dataset:
        :param batch_size:
        :type batch_size:
        :param shuffle:
        :type shuffle:
        :param flatten:
        :type flatten:
        :param data_align:
        :type data_align:
        """
        self.machine_dataset = machine_digits
        self.handwritten_dataset = handwritten_digits
        self.out_dataset = out_dataset

        self.shuffle = shuffle
        self.flatten = flatten
        self.batch_size = batch_size
        self.data_align = data_align
        if self.batch_size % 3 != 0:
            warnings.warn("The batch size should be divisible by three!")

        self.num_classes = 20

        self.machine_len = self.machine_dataset[0].shape[0]
        self.handwritten_len = self.handwritten_dataset[0].shape[0]
        self.out_len = self.out_dataset[0].shape[0]

        if not (self.machine_len == self.handwritten_len == self.out_len):
            s_data_align = "truncating larger" if self.data_align == self.TRUNCATE else "repeating smaller"
            print(f"Dataset sizes are different, {s_data_align} datasets: "
                  f"machine_len={self.machine_len}, handwritten_len={self.handwritten_len}, out_len={self.out_len}")

        self.on_epoch_end()

    @property
    def max_len(self):
        return max(self.handwritten_len, self.machine_len, self.out_len)

    @property
    def min_len(self):
        return min(self.handwritten_len, self.machine_len, self.out_len)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        if self.data_align == self.TRUNCATE:
            return int(np.ceil(self.min_len * 3 / self.batch_size))
        else:
            return int(np.ceil(self.max_len * 3 / self.batch_size))

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate one batch of data

        :param index: The batch number.
        :type index: int
        :return: Returns a tuple of a 4-dimensional ndarray and the class-categorical label ndarray
        :rtype: Tuple[np.ndarray, np.ndarray]
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
            shape = x.shape
            x = x.reshape(-1, shape[1] * shape[2])
        else:
            x = x[:, :, :, np.newaxis]

        return x, y

    def on_epoch_end(self):
        if self.data_align == self.TRUNCATE:
            self.machine_indices = np.arange(self.machine_len)
            self.handwritten_indices = np.arange(self.handwritten_len)
            self.out_indices = np.arange(self.out_len)
        else:
            self.machine_indices = np.arange(self.max_len) % self.machine_len
            self.handwritten_indices = np.arange(self.max_len) % self.handwritten_len
            self.out_indices = np.arange(self.max_len) % self.out_len

        if self.shuffle:
            np.random.shuffle(self.machine_indices)
            np.random.shuffle(self.handwritten_indices)
            np.random.shuffle(self.out_indices)

    def __machine_data_generation(self, indices):
        """
        Generates data containing batch_size samples. Machine written digits have class <digit>.
        :param indices: The indices to select
        :return: A tuple of a digit array and a class categorical array
        """
        X = self.machine_dataset[0][indices]
        y = self.machine_dataset[1][indices]

        return X, keras.utils.to_categorical(y, num_classes=self.num_classes)

    def __handwritten_data_generation(self, indices):
        """
        Generates data containing batch_size samples. MNIST digits have class <digit> + 9.
        :param indices: The indices to select
        :return: A tuple of a digit array and a class categorical array
        """
        X = self.handwritten_dataset[0][indices]
        y = self.handwritten_dataset[1][indices]

        return X, keras.utils.to_categorical(y, num_classes=self.num_classes)

    def __out_data_generation(self, indices):
        """
        Generates data containing batch_size samples. MNIST digits have class <digit> + 9.
        :param indices: The indices to select
        :return: A tuple of a digit array and a class categorical array
        """
        X = self.out_dataset[0][indices]
        y = self.out_dataset[1][indices]

        return X, keras.utils.to_categorical(y, num_classes=self.num_classes)


class SimpleDataGenerator(keras.utils.Sequence):
    def __init__(
            self,
            *datasets: (np.ndarray, np.ndarray),
            batch_size=32,
            shuffle=True,
            flatten=False,
            to_simple_digit=False
    ):
        """
        TODO: Comment
        :param machine_digits:
        :type machine_digits:
        :param handwritten_digits:
        :type handwritten_digits:
        :param out_dataset:
        :type out_dataset:
        :param batch_size:
        :type batch_size:
        :param shuffle:
        :type shuffle:
        :param flatten:
        :type flatten:
        """
        self.data = np.vstack(tuple([dataset[0] for dataset in datasets]))
        self.labels = np.hstack(tuple([dataset[1] for dataset in datasets]))

        self.shuffle = shuffle
        self.flatten = flatten
        self.batch_size = batch_size

        if to_simple_digit:
            indices = np.logical_and(self.labels > 9, self.labels != 10)
            self.labels[indices] -= 10
            self.num_classes = 11
        else:
            self.num_classes = 20

        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(self.data.shape[0] / self.batch_size))

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate one batch of data

        :param index: The batch number.
        :type index: int
        :return: Returns a tuple of a 4-dimensional ndarray and the class-categorical label ndarray
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        # Generate indices of the batch
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        x, y = self.__data_generation(indices)

        # Convert images to float
        x = x.astype(np.float32)

        # Scale x to 0..1
        x /= 255.

        if self.flatten:
            shape = x.shape
            x = x.reshape(-1, shape[1] * shape[2])
        else:
            x = x[:, :, :, np.newaxis]

        return x, y

    def on_epoch_end(self):
        self.indices = np.arange(self.data.shape[0])
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_generation(self, indices):
        """
        Generates data containing batch_size samples. MNIST digits have class <digit> + 9.
        :param indices: The indices to select
        :return: A tuple of a digit array and a class categorical array
        """
        X = self.data[indices]
        y = self.labels[indices]

        return X, keras.utils.to_categorical(y, num_classes=self.num_classes)
