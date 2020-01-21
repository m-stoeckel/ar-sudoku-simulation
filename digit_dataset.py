import os
import zipfile
from typing import List, Union

import PIL
import keras
import matplotlib.pyplot as plt
import numpy as np
import typing
from PIL import Image, ImageDraw, ImageOps
from tqdm import trange

from image_transforms import ImageTransform, RandomPerspectiveTransform

DEBUG = False
DIGIT_COUNT = 915
DIGIT_RESOLUTION = 128


class DigitDataset:

    def __init__(self, digits_path="digits/", resolution=128):
        if not os.path.exists(digits_path):
            raise FileNotFoundError(digits_path)

        if digits_path.endswith(".zip"):
            self.digit_path = "digits/"
            if not os.path.exists("digits/") or len(os.listdir("digits")) < 9 * DIGIT_COUNT:
                with zipfile.ZipFile(digits_path) as f_zip:
                    f_zip.extractall()
        else:
            self.digit_path = digits_path

        self.digits = np.empty((9 * DIGIT_COUNT, resolution, resolution), dtype=np.uint8)
        for i in trange(9 * DIGIT_COUNT, desc="Loading images"):
            digit_path = f"digits/{i}.png"
            img = Image.open(digit_path)
            if resolution != DIGIT_RESOLUTION:
                self.digits[i] = np.array(img.resize((resolution, resolution)))
            else:
                self.digits[i] = np.array(img)

        if DEBUG:
            numbers = np.hstack(
                [np.vstack([self.digits[i] for i in range(o, DIGIT_COUNT * 9, 915)]) for o in range(9)])
            plt.imshow(numbers, cmap="gray")
            plt.axis('off')
            plt.show()

    def __len__(self):
        return self.digits.shape[0]

    def __getitem__(self, item):
        return self.digits[item]

    def get_label(self, item):
        if 0 > item >= DIGIT_COUNT * 9:
            raise IndexError
        return np.floor(item / DIGIT_COUNT)


class DigitDataGenerator(keras.utils.Sequence):
    def __init__(self, dataset: DigitDataset = None, batch_size=32, shuffle=True):
        """Initialization"""
        if dataset is not None:
            self.dataset = dataset
        else:
            self.dataset = DigitDataset()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        self.transforms: List[List[ImageTransform]] = list()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.dataset) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        X, y = self.__data_generation(indices)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_generation(self, indices):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, DIGIT_RESOLUTION, DIGIT_RESOLUTION), dtype=np.uint8)
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, index in enumerate(indices):
            # Store sample
            digit: np.ndarray = self.dataset[index]
            # TODO: apply transformations
            X[i] = digit

            # Store class
            y[i] = self.dataset.get_label(index)

        return X, keras.utils.to_categorical(y, num_classes=9)

    def add_transforms(self, transforms: Union[ImageTransform, List[ImageTransform]]):
        """
        Add a transform or a list of sequential transforms to this generator.

        :param transforms: Single ImageTransform or list of sequential ImageTransform
        :return: None
        """
        if type(transforms) is ImageTransform:
            transforms = [transforms]
        elif type(transforms) is not list:
            transforms = list(transforms)
        self.transforms.append(transforms)


def generate_composition():
    transform = RandomPerspectiveTransform()
    images = [[] for _ in range(9)]
    for i in range(0, 9):
        img = Image.open(f"digits/{i * 917}.png")
        images[i].append(np.array(img))
        d = PIL.ImageDraw.Draw(img)
        d.rectangle([(16, 16), (112, 112)], outline="white")
        del d
        for _ in range(4):
            digit: Image.Image = transform.apply(img)
            images[i].append(np.array(digit))
    imgs = np.hstack([np.vstack(digits) for digits in images])
    imgs = ImageOps.invert(Image.fromarray(imgs))
    imgs.save("composition.png")
    plt.imshow(imgs, cmap="gray")
    plt.axis('off')
    plt.show()


def test_generator():
    pass
    # d = DigitDataGenerator(batch_size=1, shuffle=False)
    # X, y = d.__getitem__(0)
    # plt.imshow(np.hstack([img for img in X]), cmap="gray")
    # plt.axis('off')
    # print(y)
    # plt.show()


def transform_sudoku():
    transform = RandomPerspectiveTransform()
    img: Image.Image = Image.open(f"sudoku.jpeg")
    transformed = transform.apply(img, return_mode="RGBA")
    background = Image.new(transformed.mode, transformed.size, "white")
    Image.alpha_composite(background, transformed).save("transformed.png")
    plt.imshow(transformed, cmap="gray")
    plt.axis('off')
    plt.show()


# test_generator()
generate_composition()
transform_sudoku()
