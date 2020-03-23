import os
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable

from p_tqdm import p_map
from sklearn.datasets import fetch_openml
from tqdm import tqdm, trange

from simulation.image.image_transforms import *
from simulation.image.image_transforms import ImageTransform

DEBUG = False

INTER_DOWN_HIGH = cv2.INTER_LANCZOS4
INTER_DOWN_FAST = cv2.INTER_AREA
INTER_UP_HIGH = cv2.INTER_CUBIC
INTER_UP_FAST = cv2.INTER_LINEAR

# Classmap: {0: OUT, 0..9: #_MACHINE, 10: EMPTY, 11..19: #_HAND}
CLASS_OUT = 0
CLASS_EMPTY = 10


def strip_file_ext(path: str):
    extension = None
    for ext in [".zip", ".tar", ".tar.gz"]:
        if path.endswith(ext):
            extension = ext
    if extension:
        return "".join(list(path)[0:-len(extension)])
    else:
        return path


def char_is_valid_number(char: Union[int, str]):
    if isinstance(char, str):
        char = ord(char)
    return char in [ord(c) for c in '123456789']


class CharacterDataset:
    digit_offset = 0

    def __init__(
            self,
            resolution,
            shuffle=True,
            fast_resize=True
    ):
        self.resolution = resolution
        self.shuffle = shuffle
        self.inter_down = INTER_DOWN_FAST if fast_resize else INTER_DOWN_HIGH
        self.inter_up = INTER_UP_FAST if fast_resize else INTER_UP_HIGH

        self.train_x: np.ndarray = np.empty(0, dtype=np.uint8)
        self.train_y: np.ndarray = np.empty(0, dtype=int)
        self.test_x: np.ndarray = np.empty(0, dtype=np.uint8)
        self.test_y: np.ndarray = np.empty(0, dtype=int)

        self.train_indices_by_number: List[np.ndarray] = [np.empty(0, dtype=int)]
        self.test_indices_by_number: List[np.ndarray] = [np.empty(0, dtype=int)]

        self.transforms: List[List[ImageTransform]] = list()

        self._load()

    def _load(self):
        pass

    def __getitem__(self, item):
        return self.train_x[item]

    def __len__(self):
        return self.train_x.shape[0]

    def get_label(self, char: Union[int, str]):
        if char_is_valid_number(char):
            return int(chr(char)) + self.digit_offset
        else:
            return CLASS_OUT

    def add_transforms(self, *transforms: ImageTransform):
        """
        Add a transform or a list of sequential transforms to this generator.

        :param transforms: Single ImageTransform or list of sequential ImageTransform
        :return: None
        """
        transforms = list(transforms)
        self.transforms.append(transforms)

    def apply_transforms(self, keep=True):
        if not self.transforms:
            return
        n_train = self.train_x.shape[0]
        n_test = self.test_x.shape[0]
        n_transforms = len(self.transforms)
        new_train_shape = self.train_x.shape * np.array([int(keep) + n_transforms, 1, 1])
        new_test_shape = self.test_x.shape * np.array([int(keep) + n_transforms, 1, 1])

        # apply transforms to train and test data
        new_train_x = np.zeros(new_train_shape, dtype=np.uint8)
        new_test_x = np.zeros(new_test_shape, dtype=np.uint8)
        if keep:
            new_train_x[:n_train] = self.train_x
            new_test_x[:n_test] = self.test_x
        for i, transforms in enumerate(tqdm(self.transforms, desc="Applying transforms", position=0), start=int(keep)):
            def _apply_transforms(img):
                for transform in transforms:
                    img = transform.apply(img)
                return img

            train_x_i = p_map(
                _apply_transforms, list(self.train_x), desc="Processing images (1/2)",
                position=1, leave=False, disable=False,
                num_cpus=os.cpu_count()
            )
            test_x_i = p_map(
                _apply_transforms, list(self.test_x), desc="Processing images (2/2)",
                position=1, leave=False, disable=False,
                num_cpus=os.cpu_count()
            )

            new_train_x[n_train * i:n_train * (i + 1)] = train_x_i
            new_test_x[n_test * i:n_test * (i + 1)] = test_x_i

        # save new data
        self.train_x = new_train_x
        self.test_x = new_test_x

        # duplicate labels
        self.train_y = np.tile(self.train_y, int(keep) + n_transforms)
        self.test_y = np.tile(self.test_y, int(keep) + n_transforms)

    def resize(self, resolution=28):
        """
        Resize all images in the data set to the given resolution.

        :param resolution: The new image width/height.
        :type resolution: int
        :return: None
        :rtype: None
        """
        if resolution == self.resolution:
            return
        interpolation = self.inter_down if resolution < self.resolution else self.inter_up
        self.train_x = self._get_resized(self.train_x, resolution, interpolation)
        self.test_x = self._get_resized(self.test_x, resolution, interpolation)
        self.resolution = resolution

    @staticmethod
    def _get_resized(data, resolution, interpolation) -> np.ndarray:
        def _do_resize(img):
            img = cv2.resize(img, (resolution, resolution), interpolation=interpolation)
            return img

        # Allocate an array for the shape with the new resolution
        num_digits = data.shape[0]
        if data.shape.__len__() == 4:
            shape = tuple([num_digits] + [resolution, resolution] + [data.shape[3]])
        else:
            shape = (num_digits, resolution, resolution)
        new_digits = np.zeros(shape, dtype=np.uint8)

        # Run the resize operation on all images in parallel
        resized_images = p_map(_do_resize, [data[i] for i in range(num_digits)],
                               desc="Resizing images",
                               num_cpus=os.cpu_count())
        # Copy the resized images to the newly allocated array
        for i, img in enumerate(resized_images):
            new_digits[i] = img

        return new_digits

    def cvtColor(self, mode=cv2.COLOR_GRAY2BGRA):
        """
        Convert all images in the dataset to the specified colorspace.
        If mode is 'cv2.COLOR_GRAY2BGRA', optimized code is used, which also assigns correct alpha values.

        :param mode: The OpenCV color space conversion code.
        :type mode: int
        :return: None
        :rtype: None
        """
        self.train_x = self._get_with_colorspace(self.train_x, mode)
        self.test_x = self._get_with_colorspace(self.test_x, mode)

    @staticmethod
    def _get_with_colorspace(data, mode) -> np.ndarray:
        if mode == cv2.COLOR_GRAY2BGRA:
            # If mode is grayscale to RGBA, use optimized code instead of ordinary cvtColor
            # Also assigns correct alpha values
            tq = tqdm(desc="Changing colorspace", total=data.shape[0])
            shape = data.shape
            # Invert the original data and save as new alpha information
            alpha = cv2.bitwise_not(data)
            # Create RGBA images using array repetition
            data = data.repeat(4).reshape(shape[0], shape[1], shape[2], 4)
            # Copy alpha values to new data array
            data[:, :, :, 3] = alpha.squeeze()
            tq.update(data.shape[0])
            return data
        else:
            # Otherwise use cvtColor in parallel
            def _do_cvtcolor(img):
                return cv2.cvtColor(img, mode)

            # Allocate an array for the shape of the new colorspace
            num_digits = data.shape[0]
            shape = list(cv2.cvtColor(data[0], mode).shape)
            shape = tuple([num_digits] + shape)
            new_digits = np.zeros(shape, dtype=np.uint8)

            # Run the color transformation in parallel
            recolored_images = p_map(_do_cvtcolor, [data[i] for i in range(num_digits)],
                                     desc="Changing colorspace",
                                     num_cpus=os.cpu_count())
            # Copy the new images into the array
            for i, img in enumerate(recolored_images):
                new_digits[i] = img

            return new_digits

    def invert(self):
        self.train_x = self._get_inverted(self.train_x)
        self.test_x = self._get_inverted(self.test_x)

    @staticmethod
    def _get_inverted(data):
        tq = tqdm(desc="Inverting images", total=data.shape[0])
        cv2.bitwise_not(data, data)
        tq.update(data.shape[0])

        return data

    def induce_alpha(self, average_color: Tuple[int] = None, alpha_zero_value: int = None,
                     max_of_channel: Union[Tuple[int], List[int]] = None, invert=True):
        """
        Induce the alpha for the images in this dataset. The images must be in RGBA format for this to work.

        :param average_color: If True, set alpha value to the average of all color channels for each pixel (default).
        :type average_color: bool
        :param alpha_zero_value: If given, set the alpha value to zero for this value and to 255 for all others.
        :type alpha_zero_value: int
        :param max_of_channel: If given, set the alpha value to the maximum value of the given color channels.
        :type max_of_channel: Iterable[int]
        :param invert: If True, invert the alpha values.
        :type invert: bool
        :return:
        :rtype:
        """
        if all(p is None for p in [average_color, alpha_zero_value, max_of_channel]):
            average_color = (0, 1, 2)
        self.train_x = self._get_with_alpha(self.train_x, average_color, alpha_zero_value, max_of_channel, invert)
        self.test_x = self._get_with_alpha(self.test_x, average_color, alpha_zero_value, max_of_channel, invert)

    @staticmethod
    def _get_with_alpha(
            data, average_color=None,
            alpha_zero_value: int = None,
            max_of_channel: Tuple[int] = None,
            invert=True
    ):
        tq = tqdm(desc="Inducing alpha", total=data.shape[0])
        if average_color is not None:
            # Compute the average across all given color channels
            alpha = np.average(data[:, :, :, average_color], axis=3)
            if invert:
                cv2.bitwise_not(alpha, alpha)
            data[:, :, :, 3] = alpha
        elif alpha_zero_value is not None:
            # Get all pixels across all images for which the alpha value is equal to the alpha zero value
            indices = np.nonzero(data[:, :, :, 3] == alpha_zero_value)
            # Create a view of these pixels and set their alpha value to zero
            view = data[indices]
            view[:, 3] = (0 if not invert else 255)
            # Set all other alpha values to 255 and apply the view data
            data[:, :, :, 3] = (255 if not invert else 0)
            data[indices] = view
        elif max_of_channel is not None:
            # Compute the maximum of all pixels for the given channels across all images and set the alpha to that value
            alpha = np.max(data[:, :, :, max_of_channel], axis=3)
            if invert:
                cv2.bitwise_not(alpha, alpha)
            data[:, :, :, 3] = alpha
        else:
            raise RuntimeError
        tq.update(data.shape[0])
        return data

    def _split(self, digits, labels):
        """
        Split the dataset into train and validation splits.

        :param digits: An array of images
        :param labels: An array of labels
        """
        all_count = digits.shape[0]
        train_count = int(all_count * 0.9)
        indices = np.arange(all_count)
        if self.shuffle:
            np.random.shuffle(indices)
        self.train_x = digits[indices[:train_count]]
        self.train_y = labels[indices[:train_count]]
        self.test_x = digits[indices[train_count:]]
        self.test_y = labels[indices[train_count:]]

    def get_ordered(self, digit: int, index: int) -> np.ndarray:
        """
        Get a random sample of the given digit.

        :returns: 2D numpy array of 28x28 pixels
        """
        return self.train_x[self.train_indices_by_number[digit][index]]

    def get_random(self, digit: int) -> np.ndarray:
        """
        Get a random sample of the given digit.

        :returns: 2D numpy array of 28x28 pixels
        """
        return self.train_x[np.random.choice(self.train_indices_by_number[digit])]

    @property
    def train(self):
        """
        Returns the random train split (60.000 samples by default) of the MNIST dataset.
        """
        return self.train_x, self.train_y

    @property
    def test(self):
        """
        Returns the random test split (10.000 samples by default) of the MNIST dataset.
        """
        return self.test_x, self.test_y


class MNIST(CharacterDataset):

    def __init__(self, data_home="datasets/", shuffle=True):
        self.data_home = data_home
        super().__init__(28, shuffle)

    def _load(self):
        print("Loading MNIST dataset")
        # Load data from https://www.openml.org/d/554
        os.makedirs(self.data_home, exist_ok=True)

        x, y = fetch_openml('mnist_784', version=1, return_X_y=True, data_home=self.data_home, cache=True)
        x = np.array(x, dtype=np.uint8).reshape((70000, 28, 28))
        y = np.array(y, dtype=int)

        indices = np.arange(70000)
        if self.shuffle:
            np.random.shuffle(indices)

        self.train_x = x[indices[:60000]]
        self.train_y = y[indices[:60000]]
        self.test_x = x[indices[60000:]]
        self.test_y = y[indices[60000:]]

        self.train_indices_by_number = {i: np.flatnonzero(self.train_y == i) for i in range(0, 10)}
        self.test_indices_by_number = {i: np.flatnonzero(self.test_y == i) for i in range(0, 10)}
        del x, y


class FilteredMNIST(MNIST):

    def __init__(self):
        super().__init__()
        filtered = self.train_y > 0
        self.train_x = self.train_x[filtered]
        self.train_y = self.train_y[filtered]
        filtered = self.test_y > 0
        self.test_x = self.test_x[filtered]
        self.test_y = self.test_y[filtered]
        self.train_indices_by_number = {i: np.flatnonzero(self.train_y == i) for i in range(1, 10)}
        self.test_indices_by_number = {i: np.flatnonzero(self.test_y == i) for i in range(1, 10)}

        # Reduce all labels by one
        self.train_y -= 1
        self.test_y -= 1

    def get_random(self, digit: int) -> np.ndarray:
        """
        Get a random sample of the given digit.

        :returns: 2D numpy array of 28x28 pixels
        """
        if digit == 0:
            raise ValueError("FilteredMNIST does not contain any 0 digits!")
        return self.train_x[np.random.choice(self.train_indices_by_number[digit - 1])]


class ClassSeparateMNIST(MNIST):
    _digit_offset = 10

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        zeros = self.train_y == 0
        self.train_y[zeros] = CLASS_OUT
        zeros = self.test_y == 0
        self.test_y[zeros] = CLASS_OUT

        filtered = self.train_y > 0
        self.train_y[filtered] += self._digit_offset
        filtered = self.test_y > 0
        self.test_y[filtered] += self._digit_offset

        self.train_indices_by_number = {i: np.flatnonzero(self.train_y == i) for i in set(range(11, 20)) | {CLASS_OUT}}
        self.test_indices_by_number = {i: np.flatnonzero(self.test_y == i) for i in set(range(11, 20)) | {CLASS_OUT}}


class CuratedCharactersDataset(CharacterDataset):
    _default_digit_archive_path = "datasets/curated.tar.gz"
    _default_digit_parent_path = "datasets/"
    _default_digit_path = "datasets/curated/"

    def __init__(
            self,
            digits_path=_default_digit_archive_path,
            resolution=64,
            load_chars=None,
            **kwargs
    ):
        # If no specific list of characters is given, load all by default
        if load_chars is None:
            self.load_chars = list(range(33, 92))
            # Skip char 92 ('\') as it is not in the dataset
            self.load_chars.extend(list(range(93, 127)))
        else:
            if isinstance(load_chars, str):
                self.load_chars = [ord(c) for c in load_chars]
            else:
                self.load_chars = load_chars
        if not os.path.exists(digits_path):
            raise FileNotFoundError(digits_path)

        # Extract the dataset if is compressed
        if digits_path.endswith((".zip", ".tar", ".tar.gz")):
            self.digit_path = Path(strip_file_ext(digits_path))
            parent = Path(digits_path).parent
            if not os.path.exists(self.digit_path):
                if digits_path.endswith(".zip"):
                    f_archive = zipfile.ZipFile(digits_path)
                else:
                    f_archive = tarfile.TarFile(digits_path)
                f_archive.extractall(parent)
                f_archive.close()
        else:
            self.digit_path: Path = Path(digits_path)

        # Construct a map of all character paths and their respective label
        self.file_map = {}
        for char in self.load_chars:
            files = os.listdir(self.digit_path / str(char))
            for file in files:
                label = self.get_label(char)
                self.file_map.update({str(self.digit_path / str(char) / file): label})

        super().__init__(resolution, **kwargs)

    def _load(self):
        """
        Load the Curated Handwritten Character dataset.
        """
        char_count = len(self.file_map)

        digits = np.zeros((char_count, self.resolution, self.resolution), dtype=np.uint8)
        labels = np.zeros(char_count, dtype=int)

        for i, (path, label) in tqdm(enumerate(self.file_map.items()), total=char_count, desc="Loading images"):
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if self.resolution != img.shape[0]:
                interpolation = cv2.INTER_LANCZOS4 if self.resolution < img.shape[0] else cv2.INTER_CUBIC
                digits[i] = cv2.resize(img, (self.resolution, self.resolution), interpolation=interpolation)
            else:
                digits[i] = img
            labels[i] = label

        self._split(digits, labels)

        self.train_indices_by_number = {i: np.flatnonzero(self.train_y == i) for i in
                                        set(range(1 + self.digit_offset, 10 + self.digit_offset)) | {CLASS_OUT}}
        self.test_indices_by_number = {i: np.flatnonzero(self.test_y == i) for i in
                                       set(range(1 + self.digit_offset, 10 + self.digit_offset)) | {CLASS_OUT}}


class ClassSeparateCuratedCharactersDataset(CuratedCharactersDataset):
    """
    A variant of the CuratedCharactersDataset which assigns the classes 11-19 to digits.
    """
    digit_offset = 10


class PrerenderedDigitDataset(CharacterDataset):
    default_digit_parent = "datasets/"
    default_digit_path = "datasets/digits/"

    def __init__(self, digits_path="datasets/digits.zip", resolution=128, digit_count=915):
        if not os.path.exists(digits_path):
            raise FileNotFoundError(digits_path)

        self.digit_count = digit_count
        if digits_path.endswith(".zip"):
            parent = Path(digits_path).parent
            self.digit_path = Path(digits_path.rstrip(".zip"))
            if not os.path.exists(self.digit_path):
                with zipfile.ZipFile(digits_path) as f_zip:
                    f_zip.extractall(parent)
        else:
            self.digit_path: Path = Path(digits_path)

        super().__init__(resolution)

    def _load(self):
        digit_count = 9 * self.digit_count
        digits = np.zeros((digit_count, self.resolution, self.resolution), dtype=np.uint8)
        labels = np.zeros(digit_count, dtype=int)

        for i in trange(digit_count, desc="Loading images"):
            digit_path = self.digit_path / f"{i}.png"
            img = cv2.imread(str(digit_path), cv2.IMREAD_GRAYSCALE)
            if self.resolution != img.shape[0]:
                interpolation = cv2.INTER_LANCZOS4 if self.resolution < img.shape[0] else cv2.INTER_CUBIC
                digits[i] = cv2.resize(img, (self.resolution, self.resolution), interpolation=interpolation)
            else:
                digits[i] = img
            labels[i] = np.floor(i / self.digit_count)

        self._split(digits, labels)


class PrerenderedCharactersDataset(CuratedCharactersDataset):
    _default_digit_parent_path = "datasets/"
    _default_digit_path = "datasets/characters/"

    def __init__(self, digits_path=_default_digit_path, resolution=64, load_chars=None, **kwargs):
        super().__init__(digits_path, resolution, load_chars, **kwargs)


class ConcatDataset(CharacterDataset):
    r"""Concatenate multiple datasets into a single one. Old datasets should be removed afterwards.
    """

    def __init__(self, datasets: List[CharacterDataset], delete=True):
        assert len(datasets) > 0, 'Datasets should not be an empty iterable'
        train_size, test_size = 0, 0
        res = None
        for d in datasets:
            if res is None:
                res = d.resolution
            else:
                d.resize(res)
            train_size += d.train_x.shape[0]
            test_size += d.test_x.shape[0]
        super(ConcatDataset, self).__init__(res)

        self.train_x: np.ndarray = np.zeros((train_size, res, res), dtype=np.uint8)
        self.train_y: np.ndarray = np.zeros(train_size, dtype=int)
        self.test_x: np.ndarray = np.zeros((test_size, res, res), dtype=np.uint8)
        self.test_y: np.ndarray = np.zeros(test_size, dtype=int)

        train_offset = 0
        test_offset = 0
        for d in datasets:
            train_size = d.train_x.shape[0]
            test_size = d.test_x.shape[0]
            self.train_x[train_offset:train_offset + train_size] = d.train_x
            self.train_y[train_offset:train_offset + train_size] = d.train_y
            self.test_x[test_offset:test_offset + test_size] = d.test_x
            self.test_y[test_offset:test_offset + test_size] = d.test_y
            train_offset += train_size
            test_offset += test_size
            if delete:
                del d


class EmptyDataset(CharacterDataset):
    def __init__(self, resolution, size=1000):
        self.size = size
        super().__init__(resolution)

    def _load(self):
        data = np.zeros((self.size, self.resolution, self.resolution), dtype=np.uint8)
        labels = np.full(self.size, CLASS_EMPTY, dtype=int)
        self._split(data, labels)

    def _split(self, data, labels):
        """
        Split the dataset into train and validation splits.

        :param data: An array of images
        :param labels: An array of labels
        """
        train_count = int(data.shape[0] * 0.9)
        self.train_x = data[:train_count]
        self.train_y = labels[:train_count]
        self.test_x = data[train_count:]
        self.test_y = labels[train_count:]
