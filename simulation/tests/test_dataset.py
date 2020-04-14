from unittest import TestCase

import numpy as np
from matplotlib import pyplot as plt

from simulation.data import BalancedDataGenerator
from simulation.data.dataset import PrerenderedDigitDataset, CuratedCharactersDataset, \
    ClassSeparateMNIST, ConcatDataset, PrerenderedCharactersDataset, EmptyDataset


class Test(TestCase):
    def test_generator(self):
        digit_characters = "123456789"
        non_digit_characters = "0abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.!?"

        # Digit dataset
        digit_dataset = PrerenderedDigitDataset(digits_path="../../datasets/digits/")
        digit_dataset.resize(28)

        # Prerendered character dataset - digits
        prerendered_digit_dataset = PrerenderedCharactersDataset(
            digits_path="../../datasets/characters/",
            load_chars=digit_characters
        )
        prerendered_digit_dataset.resize(28)

        # Prerendered character dataset - non-digits
        prerendered_nondigit_dataset = PrerenderedCharactersDataset(
            digits_path="../../datasets/characters/",
            load_chars=non_digit_characters
        )
        prerendered_nondigit_dataset.resize(28)

        # Handwritten non-digits
        curated_out = CuratedCharactersDataset(
            digits_path="../../datasets/curated/",
            load_chars=non_digit_characters
        )
        curated_out.resize(28)

        # Handwritten digits
        curated_digits = CuratedCharactersDataset(digits_path="../../datasets/curated/", load_chars=digit_characters)
        curated_digits.resize(28)

        # Mnist digits
        mnist = ClassSeparateMNIST(data_home="../../datasets/")

        # Empty dataset
        empty_dataset = EmptyDataset(28, 1000)

        # Concatenate datasets
        concat_machine = ConcatDataset(digit_dataset, prerendered_digit_dataset)
        concat_hand = ConcatDataset(mnist, curated_digits)
        concat_out = ConcatDataset(curated_out, prerendered_nondigit_dataset, empty_dataset)

        batch_size = 12
        d = BalancedDataGenerator(
            concat_machine.train, concat_hand.train, concat_out.train,
            batch_size=batch_size,
            shuffle=True
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
    Test().test_generator()
