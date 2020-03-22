from unittest import TestCase

import numpy as np
from matplotlib import pyplot as plt

from simulation.digit import BalancedDataGenerator
from simulation.digit.dataset import PrerenderedDigitDataset, RandomPerspectiveTransform, CuratedCharactersDataset, \
    ClassSeparateMNIST, ConcatDataset


class Test(TestCase):
    def test_generator(self):
        prerendered_dataset = PrerenderedDigitDataset(digits_path="../../datasets/digits/")
        prerendered_dataset.add_transforms(RandomPerspectiveTransform())
        # dataset.add_transforms(RandomPerspectiveTransformX())
        # dataset.add_transforms(RandomPerspectiveTransformY())
        prerendered_dataset.apply_transforms(keep=False)
        prerendered_dataset.resize(28)

        # handwritten non-digits
        curated_out = CuratedCharactersDataset(
            digits_path="../../datasets/curated/",
            load_chars="0abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.!?"
        )
        curated_out.add_transforms(RandomPerspectiveTransform())
        curated_out.apply_transforms(keep=False)
        curated_out.resize(28)

        # handwritten digits
        curated_digits = CuratedCharactersDataset(digits_path="../../datasets/curated/", load_chars="123456789")
        curated_digits.add_transforms(RandomPerspectiveTransform())
        curated_digits.apply_transforms(keep=False)
        curated_digits.resize(28)

        # mnist digits
        mnist = ClassSeparateMNIST(data_home="../../datasets/")
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
