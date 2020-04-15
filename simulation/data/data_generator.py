import warnings
from abc import abstractmethod, ABCMeta
from typing import Tuple, Iterable, Union, List

import numpy as np
import tensorflow.keras as keras


class BaseDataGenerator(keras.utils.Sequence, metaclass=ABCMeta):
    """
    Abstract base class for all character data generators.
    """

    @abstractmethod
    def get_data(self):
        """
        Get all data from this generator as a single array. The array contains normalized floats.

        Returns:
            :py:class:`numpy.ndarray`: All data of this generator as a float array.
        """
        pass

    @abstractmethod
    def get_labels(self):
        """
        Get all labels of this generator in a single array.

        Returns:
            :py:class:`numpy.ndarray`: All labels of this generator as an integer array.
        """
        pass


class BalancedDataGenerator(BaseDataGenerator):
    """
    This generator balances each of its input datasets. There are two strategies for balancing:
    
    * truncate: trims all datasets to the length of the shortest dataset for each epoch,
    * repeat: all datasets shorter than the longest dataset will be repeated during each epoch.

    """

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
        

        Args:
            datasets: A sequence of tuples, one for each dataset containing (data, labels).
            batch_size(int, optional): The batch size. Should be divisible by the number of datasets.
                (Default value = 32)
            shuffle(bool, optional): If True, shuffle the datasets at the end of each epoch. (Default value = True)
            flatten(bool, optional): If True, flatten the datasets. (Default value =  False)
            truncate(bool, optional): If True, the datasets will be truncated to the length of the shortest dataset.
                Else, they will be repeated. (Default value = True)
            num_classes(int, optional): The number of classes in the datasets. If None, will be inferred from the data.
                (Default value = 20)

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

        if num_classes is not None:
            self.num_classes = num_classes
        else:
            self.num_classes = len(np.unique(self.get_labels()))

        if not np.alltrue(self.lengths == self.lengths[0]):
            s_data_align = "truncating larger" if self.truncate else "repeating smaller"
            print(f"Dataset sizes are different, {s_data_align} datasets.")

        self.on_epoch_end()

    @property
    def num_datasets(self) -> int:
        """The number of datasets."""
        return len(self.lengths)

    @property
    def max_len(self) -> int:
        """The maximum length of all datasets."""
        return max(self.lengths)

    @property
    def min_len(self) -> int:
        """The minimum length of all datasets."""
        return min(self.lengths)

    def __len__(self) -> int:
        """Denotes the number of batches per epoch"""
        if self.truncate:
            return int(np.ceil(self.min_len * self.num_datasets / self.batch_size))
        else:
            return int(np.ceil(self.max_len * self.num_datasets / self.batch_size))

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate one batch of data. If the batch size is not divisible by the number of datasets, the **last** dataset
        will be used to fill the batch.

        Args:
            index(int): The batch number.

        Returns:
            Returns a tuple of a 4-dimensional array and the class label array.

        """
        # Generate indices of the batch
        indices = [dataset_indices[index * self.mini_batch_size:(index + 1) * self.mini_batch_size]
                   for dataset_indices in self.indices]
        indices[-1] = self.indices[-1][index * self.last_mini_batch_size:(index + 1) * self.last_mini_batch_size]

        # Generate data
        xs, ys = self._data_generation(indices)

        # Stack data, convert images to float and scale to 0..1
        x = np.vstack(xs).astype(np.float32) / 255.
        y = np.hstack(ys)

        if self.flatten:
            shape = x.shape
            x = x.reshape(-1, shape[1] * shape[2])
        else:
            x = x[:, :, :, np.newaxis]

        return x, y

    def on_epoch_end(self):
        """
        Shuffles the datasets and chooses new indices according to the balancing strategy.
        
        If :py:attr:`truncate` is True, each dataset will get ordinary indices, some of which will not be selected in
        batch generation.
        
        If :py:attr:`truncate` is False however, each dataset will get indices for the length of the
        *longest* dataset. Too large indices are pruned using the modulo operator.
        
        Returns:
             None

        """
        if self.truncate:
            self.indices = [np.arange(len(dataset)) for dataset in self.datasets]
        else:
            self.indices = [np.arange(self.max_len) % len(dataset) for dataset in self.datasets]

        if self.shuffle:
            for dataset_indices in self.indices:
                np.random.shuffle(dataset_indices)

    def _data_generation(self, indices: List[np.ndarray]) -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]]:
        """
        Helper function to generate one batch of data from multiple datasets.

        Args:
            indices: The batch indices.

        Returns:
            Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]]: One batch of data as a two tuples.

        """
        xs, ys = [], []
        for i in range(self.num_datasets):
            xs.append(self.datasets[i][indices[i]])
            ys.append(self.labels[i][indices[i]])
        return tuple(xs), tuple(ys)

    def get_data(self):
        data = np.vstack(tuple([dataset[0] for dataset in self.datasets]))
        data = data[:, :, :, np.newaxis]
        data = data.astype(np.float32) / 255.
        if self.flatten:
            data = data.squeeze()
            shape = data.shape
            data = data.reshape(-1, shape[1] * shape[2])
        else:
            data = data[:, :, :, np.newaxis]
        return data

    def get_labels(self):
        return np.hstack(tuple([dataset[1] for dataset in self.datasets]))


class ToBinaryGenerator(BalancedDataGenerator):
    """
    A variant of the :py:class:`BalancedDataGenerator` which converts the labels to binary given a class or a sequence
    of classes to match. Matched classes will be given the label *1*, others *0*. This converted data will then be split
    into two datasets by class from which a :py:class:`BalancedDataGenerator` is constructed.

    """

    def __init__(
            self,
            *datasets: Tuple[np.ndarray, np.ndarray],
            class_to_match: Union[int, Iterable[int]] = 0,
            **kwargs
    ):
        """
        

        Args:
            datasets(Tuple[:py:class:`numpy.ndarray`, :py:class:`numpy.ndarray`]): A sequence of datasets to convert to
                binary.
            class_to_match(Union[int, Iterable[int]]): The class or classes to match as binary *1* class.
            kwargs(dict): Arbitrary :py:class:`BalancedDataGenerator` arguments.

        """
        self.data = np.vstack(tuple([dataset[0] for dataset in datasets]))
        self.all_labels = np.hstack(tuple([dataset[1] for dataset in datasets]))

        # Split all datasets into matching and other data
        if isinstance(class_to_match, int):
            matched_indices = self.all_labels == class_to_match
        else:
            iterator = iter(class_to_match)
            matched_indices = self.all_labels == iterator.__next__()
            for cls in iterator:
                matched_indices = np.logical_or(matched_indices, self.all_labels == cls)
        other_indices = np.logical_not(matched_indices)

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
        data = self.data.astype(np.float32) / 255.
        if self.flatten:
            data = data.squeeze()
            shape = data.shape
            data = data.reshape(-1, shape[1] * shape[2])
        else:
            data = data[:, :, :, np.newaxis]
        return data

    def get_labels(self):
        return self.all_labels


class SimpleDataGenerator(BaseDataGenerator):
    """
    A simple data generator which does not do any balancing. All input datasets are concatenated into a single array for
    both data and labels.
    
    They are also immediately converted to normalized float arrays, giving a possible performance increase in
    comparison to :py:class:`BalancedDataGenerator`.

    """

    def __init__(
            self,
            *datasets: Tuple[np.ndarray, np.ndarray],
            batch_size=32,
            shuffle=True,
            flatten=False,
            to_simple_digit=False,
            no_zero=False
    ):
        """
        

        Args:
            datasets(Tuple[:py:class:`numpy.ndarray`, :py:class:`numpy.ndarray`]): The input datasets as a sequence of
                (data, label) tuples.
            batch_size(int, optional): The batch size. (Default value = 32)
            shuffle(bool, optional): If True, shuffle the datasets at the end of each epoch. (Default value = True)
            flatten(bool, optional): If True, flatten the datasets. (Default value = False)
            to_simple_digit(bool, optional): If True, convert the dataset into a simple digit dataset, mapping all
                handwritten digits to the class of machine written digits. (Default value = False)
            no_zero(bool, optional): If True and :py:data:`to_simple_digit` is True too, remove all 0-class entries
                from the datasets. (Default value = False)

        """
        self.data = np.vstack(tuple([dataset[0] for dataset in datasets]))

        # Convert images to float and scale to 0..1
        self.data = self.data.astype(np.float32) / 255.

        if flatten:
            shape = self.data.shape
            self.data = self.data.reshape(-1, shape[1] * shape[2])
        else:
            self.data = self.data[:, :, :, np.newaxis]

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
        Generate one batch of data.

        Args:
            index(int): The batch number.

        Returns:
            Tuple[:py:class:`numpy.ndarray`, :py:class:`numpy.ndarray`]: A tuple of a 4-dimensional array and the class
                label array.

        """
        # Generate indices of the batch
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        x = self.data[indices]
        y = self.labels[indices]

        return x, y

    def on_epoch_end(self):
        self.indices = np.arange(self.data.shape[0])
        if self.shuffle:
            np.random.shuffle(self.indices)

    def get_data(self):
        return self.data

    def get_labels(self):
        return self.labels
