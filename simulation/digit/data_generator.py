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


class BaseDataGenerator(keras.utils.Sequence):
    def get_data(self):
        pass

    def get_labels(self):
        pass


class BalancedDataGenerator(BaseDataGenerator):
    def __init__(
            self,
            *datasets: Tuple[np.ndarray, np.ndarray],
            batch_size=32,
            shuffle=True,
            flatten=False,
            truncate=True,
            num_classes=20
    ):
        """

        :param datasets:
        :type datasets:
        :param batch_size:
        :type batch_size:
        :param shuffle:
        :type shuffle:
        :param flatten:
        :type flatten:
        :param truncate:
        :type truncate:
        """
        self.datasets = [dataset[0] for dataset in datasets]
        self.labels = [dataset[1] for dataset in datasets]
        self.lengths = [dataset[1].shape[0] for dataset in datasets]

        self.shuffle = shuffle
        self.flatten = flatten
        self.truncate = truncate
        self.batch_size = batch_size

        if batch_size % self.num_datasets != 0:
            warnings.warn("The batch size should be divisible by the number of datasets!")
        self.mini_batch_size = int(batch_size / self.num_datasets)
        self.last_mini_batch_size = batch_size - (self.num_datasets - 1) * self.mini_batch_size

        self.num_classes = num_classes

        if not np.alltrue(self.lengths == self.lengths[0]):
            s_data_align = "truncating larger" if self.truncate else "repeating smaller"
            print(f"Dataset sizes are different, {s_data_align} datasets.")

        self.on_epoch_end()

    @property
    def num_datasets(self):
        return len(self.lengths)

    @property
    def max_len(self):
        return max(self.lengths)

    @property
    def min_len(self):
        return min(self.lengths)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        if self.truncate:
            return int(np.ceil(self.min_len * self.num_datasets / self.batch_size))
        else:
            return int(np.ceil(self.max_len * self.num_datasets / self.batch_size))

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate one batch of data

        :param index: The batch number.
        :type index: int
        :return: Returns a tuple of a 4-dimensional ndarray and the class-categorical label ndarray
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        # Generate indexes of the batch
        indices = [dataset_indices[index * self.mini_batch_size:(index + 1) * self.mini_batch_size]
                   for dataset_indices in self.indices]
        indices[-1] = self.indices[-1][index * self.last_mini_batch_size:(index + 1) * self.last_mini_batch_size]

        # Generate data
        xs, ys = self._data_generation(indices)

        # Stack data and convert images to float
        x = np.vstack(xs).astype(np.float32)
        y = np.hstack(ys)

        # Scale x to 0..1
        x /= 255.

        if self.flatten:
            shape = x.shape
            x = x.reshape(-1, shape[1] * shape[2])
        else:
            x = x[:, :, :, np.newaxis]

        return x, y

    def on_epoch_end(self):
        if self.truncate:
            self.indices = [np.arange(len(dataset)) for dataset in self.datasets]
        else:
            self.indices = [np.arange(self.max_len) % len(dataset) for dataset in self.datasets]

        if self.shuffle:
            for dataset_indices in self.indices:
                np.random.shuffle(dataset_indices)

    def _data_generation(self, indices):
        xs, ys = [], []
        for i in range(self.num_datasets):
            xs.append(self.datasets[i][indices[i]])
            ys.append(keras.utils.to_categorical(self.labels[i][indices[i]], num_classes=self.num_classes))
        return tuple(xs), tuple(ys)

    def get_data(self):
        return np.vstack(tuple([dataset[0] for dataset in self.datasets]))

    def get_labels(self):
        return np.hstack(tuple([dataset[1] for dataset in self.datasets]))


class ToBinaryGenerator(BalancedDataGenerator):
    def __init__(
            self,
            *datasets: (np.ndarray, np.ndarray),
            class_to_match: int = 0,
            **kwargs
    ):
        """

        :param datasets:
        :type datasets:
        :param class_to_match:
        :type class_to_match:
        :param kwargs:
        :type kwargs:
        """
        self.data = np.vstack(tuple([dataset[0] for dataset in datasets]))
        self.all_labels = np.hstack(tuple([dataset[1] for dataset in datasets]))

        # Split all datasets into matching and other data
        matched_indices = self.all_labels == class_to_match
        other_indices = self.all_labels != class_to_match

        # Assign new binary labels
        self.all_labels[matched_indices] = 1
        self.all_labels[other_indices] = 0

        matched_data = (self.data[matched_indices], self.all_labels[matched_indices])
        other_data = (self.data[other_indices], self.all_labels[other_indices])
        super().__init__(matched_data, other_data, num_classes=1, **kwargs)

    def _data_generation(self, indices):
        xs, ys = [], []
        for i in range(self.num_datasets):
            xs.append(self.datasets[i][indices[i]])
            ys.append(self.labels[i][indices[i]])
        return xs, ys

    def get_data(self):
        return self.data

    def get_labels(self):
        return self.all_labels


class SimpleDataGenerator(BaseDataGenerator):
    def __init__(
            self,
            *datasets: (np.ndarray, np.ndarray),
            batch_size=32,
            shuffle=True,
            flatten=False,
            to_simple_digit=False,
            no_zero=False
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
            indices = self.labels != 10
            self.data = self.data[indices]
            self.labels = self.labels[indices]
            if no_zero:
                self.labels -= 1
                self.num_classes = 9
            else:
                self.num_classes = 10
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
        x, y = self._data_generation(indices)

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

    def _data_generation(self, indices):
        """
        Generates data containing batch_size samples. MNIST digits have class <digit> + 9.
        :param indices: The indices to select
        :return: A tuple of a digit array and a class categorical array
        """
        x = self.data[indices]
        y = self.labels[indices]

        return x, keras.utils.to_categorical(y, num_classes=self.num_classes)

    def get_data(self):
        return self.data

    def get_labels(self):
        return self.labels
