import os
import zipfile
from pathlib import Path
from typing import List, Tuple

import keras
from sklearn.datasets import fetch_openml
from tqdm import trange, tqdm

from image.image_transforms import *

DEBUG = False


class MNIST:
    def __init__(self, shuffle=True):
        print("Loading MNIST dataset")
        # Load data from https://www.openml.org/d/554
        os.makedirs('datasets/', exist_ok=True)
        x, y = fetch_openml('mnist_784', version=1, return_X_y=True, data_home="datasets/", cache=True)
        x = np.array(x, dtype=np.uint8).reshape((70000, 28, 28))
        y = np.array(y, dtype=int)
        indices = np.arange(70000)
        if shuffle:
            np.random.shuffle(indices)
        self.train_x = x[indices[:60000]]
        self.train_y = y[indices[:60000]]
        self.test_x = x[indices[60000:]]
        self.test_y = y[indices[60000:]]
        self.train_indices_by_number = [np.flatnonzero(self.train_y == i) for i in range(0, 10)]
        self.test_indices_by_number = [np.flatnonzero(self.test_y == i) for i in range(0, 10)]
        del x, y

    def get_random(self, digit: int) -> np.ndarray:
        """
        Get a random sample of the given digit.

        :returns: 2D numpy array of 28x28 pixels
        """
        return self.train_x[np.random.choice(self.train_indices_by_number[digit])]

    def get_ordered(self, digit: int, index: int) -> np.ndarray:
        """
        Get a random sample of the given digit.

        :returns: 2D numpy array of 28x28 pixels
        """
        return self.train_x[self.train_indices_by_number[digit][index]]

    @property
    def train(self):
        """
        Returns the random train split (60.000 samples) of the MNIST dataset.
        """
        return (self.train_x, self.train_y)

    @property
    def test(self):
        """
        Returns the random test split (10.000 samples) of the MNIST dataset.
        """
        return (self.test_x, self.test_y)


class FilteredMNIST(MNIST):
    def __init__(self):
        super().__init__()
        filtered = self.train_y > 0
        self.train_x = self.train_x[filtered]
        self.train_y = self.train_y[filtered]
        filtered = self.test_y > 0
        self.test_x = self.test_x[filtered]
        self.test_y = self.test_y[filtered]
        self.train_indices_by_number = [np.flatnonzero(self.train_y == i) for i in range(1, 10)]
        self.test_indices_by_number = [np.flatnonzero(self.test_y == i) for i in range(1, 10)]

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


class DigitDataset:
    default_digit_parent = "datasets/"
    default_digit_path = "datasets/digits/"

    def __init__(self, digits_path="datasets/digits.zip", resolution=28, digit_count=915):
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

        self.transforms: List[List[ImageTransform]] = list()
        self.res = resolution
        self.digits = np.empty((9 * self.digit_count, self.res, self.res), dtype=np.uint8)
        self.labels = np.empty(9 * self.digit_count, dtype=int)
        self._load()

    def _load(self):
        for i in trange(9 * self.digit_count, desc="Loading images"):
            digit_path = self.digit_path / f"{i}.png"
            img = cv2.imread(str(digit_path), cv2.IMREAD_GRAYSCALE)
            self._add_digit_and_label(i, img)

    def _add_digit_and_label(self, i, img):
        if self.res != img.shape[0]:
            interpolation = cv2.INTER_LANCZOS4 if self.res < img.shape[0] else cv2.INTER_CUBIC
            self.digits[i] = cv2.resize(img, (self.res, self.res), interpolation=interpolation)
        else:
            self.digits[i] = img
        self.labels[i] = np.floor(i / self.digit_count)

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
        o_digits = self.digits
        n_digits = o_digits.shape[0]
        n_transforms = len(self.transforms)
        new_shape = o_digits.shape * np.array([int(keep) + n_transforms, 1, 1])
        self.digits = np.empty(new_shape, dtype=np.uint8)
        if keep:
            self.digits[:n_digits] = o_digits
        for i, transforms in enumerate(self.transforms, start=int(keep)):
            for j in range(n_digits):
                img = o_digits[j]
                for transform in transforms:
                    img = transform.apply(img)
                self.digits[n_digits * i + j] = img
        self.labels = np.tile(self.labels, int(keep) + n_transforms)

    def __len__(self):
        return self.digits.shape[0]

    def __getitem__(self, item):
        return self.digits[item]


def plot_img(img, figsize=(4, 3)):
    plt.figure(figsize=figsize)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()


class Chars74KI(DigitDataset):
    """
    Class for the handwritten numbers part of the Chars74KI "EnglishHnd" dataset.
    :source: http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/
    """

    default_digit_path = "datasets/digits_hnd/"

    def __init__(self, digits_path="datasets/digits_hnd.zip", resolution=28, digit_count=55):
        super().__init__(digits_path=digits_path, resolution=resolution, digit_count=digit_count)

    def _load(self):
        all_digits = []
        for i in range(1, 10):
            for file_name in os.listdir(str(self.digit_path / f"{i}")):
                all_digits.append(self.digit_path / (f"{i}/" + file_name))
        for i, digit_path in enumerate(tqdm(all_digits, desc="Loading images")):
            img = cv2.imread(str(digit_path), cv2.IMREAD_GRAYSCALE)
            img = self._crop(img)
            img = cv2.bitwise_not(img)
            self._add_digit_and_label(i, img)

    def _crop(self, img):
        """
        Crop off the image side which is empty. Crop to the center if both sides are empty or both sides contain part
        of the digit.
        """
        if np.any(img[:, :300] == 0) == np.any(img[:, 900:] == 0):
            if DEBUG:
                img[:, :150] = 128
                img[:, 1050:] = 128
                plot_img(img)
            return img[:, 150:1050]
        elif np.any(img[:, :300] == 0):
            if DEBUG:
                img[:, 900:] = 128
                plot_img(img)
            return img[:, :900]
        else:
            if DEBUG:
                img[:, :300] = 128
                plot_img(img)
            return img[:, 300:]

    def get_random(self, digit: int) -> np.ndarray:
        """
        Get a random sample of the given digit.

        :returns: 2D numpy array of 28x28 pixels
        """
        return self.digits[np.random.randint(self.digit_count * (digit - 1), self.digit_count * digit)]

    def get_ordered(self, digit: int, index: int) -> np.ndarray:
        """
        Get a random sample of the given digit.

        :returns: 2D numpy array of 28x28 pixels
        """
        return self.digits[self.digit_count * (digit - 1) + (index % self.digit_count)]


class Chars74KIRGBA(Chars74KI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load(self):
        self.digits = np.empty((9 * self.digit_count, self.res, self.res, 4), dtype=np.uint8)
        all_digits = []
        for i in range(1, 10):
            for file_name in os.listdir(str(self.digit_path / f"{i}")):
                all_digits.append(self.digit_path / (f"{i}/" + file_name))
        for i, digit_path in enumerate(tqdm(all_digits, desc="Loading images")):
            img = cv2.imread(str(digit_path), cv2.IMREAD_GRAYSCALE)
            img = self._crop(img)
            shape = img.shape
            alpha = cv2.bitwise_not(img)
            img = img.repeat(4).reshape(shape[0], shape[1], 4)
            img[:, :, 3] = alpha
            self._add_digit_and_label(i, img)


class DigitDataGenerator(keras.utils.Sequence):
    def __init__(self, dataset: DigitDataset, batch_size=32, shuffle=True, **kwargs):
        """Initialization"""
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.dataset) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        x, y = self.__data_generation(indices)
        x = x.astype(np.float32)

        return x[:, :, :, np.newaxis], y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_generation(self, indices):
        """Generates data containing batch_size samples"""
        x = self.dataset.digits[indices]
        y = self.dataset.labels[indices]

        return x, keras.utils.to_categorical(y, num_classes=9)


class BalancedMnistDigitDataGenerator(DigitDataGenerator):
    def __init__(self, dataset: DigitDataset, mnist_data: Tuple[np.ndarray, np.ndarray], batch_size=32, shuffle=True,
                 separate_mnist=True, flatten=False, **kwargs):
        self.mnist_x = mnist_data[0]
        self.mnist_y = mnist_data[1]
        self.separate_mnist = separate_mnist
        self.mnist_classes = len(np.unique(self.mnist_y))

        # Number of classes is 9 if separate_mnist is False,
        # 18 if separate_mnist is True and the MNIST dataset was zero-filtered
        # and 19 otherwise. In any case, the MNIST classes are LAST in the label order.
        self.num_classes = 9 + self.mnist_classes if self.separate_mnist else 9

        self.flatten = flatten

        super().__init__(dataset, batch_size, shuffle, **kwargs)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(len(self.dataset) * 2 / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data
        :param index: The batch number.
        :return: Returns a tuple of a 4-dimensional ndarray and the class-categorical label ndarray
        """
        # Generate indexes of the batch
        mini_batch_size = int(self.batch_size / 2)
        digit_indices = self.indices[index * mini_batch_size:(index + 1) * mini_batch_size]
        mnist_indices = self.mnist_indices[index * mini_batch_size:(index + 1) * mini_batch_size]

        # Generate data
        xd, yd = self.__data_generation(digit_indices)
        xm, ym = self.__mnist_data_generation(mnist_indices)

        # Stack data and convert images to float
        x = np.vstack((xd, xm)).astype(np.float32)
        y = np.vstack((yd, ym))

        # Scale x to 0..1
        x /= 255.

        if self.flatten:
            x = x.reshape(-1, 28 ** 2)
        else:
            x = x[:, :, :, np.newaxis]

        return x, y

    def on_epoch_end(self):
        super().on_epoch_end()
        self.mnist_indices = np.arange(len(self.mnist_y))
        if self.shuffle:
            np.random.shuffle(self.mnist_indices)

    def __data_generation(self, indices, ):
        """
        Generates data containing batch_size samples. Machine written digits have class <digit>.
        :param indices: The indices to select
        :return: A tuple of a digit array and a class categorical array
        """
        X = self.dataset.digits[indices]
        y = self.dataset.labels[indices]

        return X, keras.utils.to_categorical(y, num_classes=self.num_classes)

    def __mnist_data_generation(self, indices, ):
        """
        Generates data containing batch_size samples. MNIST digits have class <digit> + 9.
        :param indices: The indices to select
        :return: A tuple of a digit array and a class categorical array
        """
        X = self.mnist_x[indices]
        y = self.mnist_y[indices] + (self.mnist_classes if self.separate_mnist else 0)

        return X, keras.utils.to_categorical(y, num_classes=self.num_classes)


def generate_composition():
    transform = RandomPerspectiveTransform()
    images = [[] for _ in range(9)]
    for i in range(0, 9):
        img = cv2.imread(f"datasets/digits/{i * 917}.png", cv2.IMREAD_GRAYSCALE)
        img = cv2.rectangle(img, (16, 16), (112, 112), (255, 255, 255), 2)
        images[i].append(img)
        for _ in range(4):
            digit = transform.apply(img)
            images[i].append(digit)
    imgs = np.hstack([np.vstack(digits) for digits in images])
    imgs = cv2.bitwise_not(imgs)
    imgs.save("composition.png")
    plt.imshow(imgs, cmap="gray")
    plt.axis('off')
    plt.show()


def test_generator():
    dataset = DigitDataset(resolution=28)
    dataset.add_transforms(RandomPerspectiveTransform())
    # dataset.add_transforms(RandomPerspectiveTransformX())
    # dataset.add_transforms(RandomPerspectiveTransformY())
    dataset.apply_transforms(keep=False)
    mnist = FilteredMNIST()
    d = BalancedMnistDigitDataGenerator(dataset, (mnist.train_x, mnist.train_y),
                                        batch_size=8, shuffle=True, resolution=28)
    img_l = []
    for i in range(8):
        X, y = d[i]
        img_l.append(np.hstack([img for img in X]))
    plt.imshow(np.vstack(img_l), cmap="gray")
    plt.axis('off')
    plt.show()


def transform_sudoku():
    transform = RandomPerspectiveTransform()
    img = cv2.imread(f"../sudoku.jpeg", cv2.IMREAD_GRAYSCALE)
    transformed = transform.apply(img)
    plt.imshow(transformed, cmap="gray")
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # transform_sudoku()
    test_generator()
    # generate_composition()
