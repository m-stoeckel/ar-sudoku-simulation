from unittest import TestCase

import cv2
import numpy as np
from matplotlib import pyplot as plt

from image.image_transforms import GaussianNoise, RandomPerspectiveTransform, GaussianBlur


class Test(TestCase):
    def test(self):
        img = cv2.imread("../sudoku.jpeg", cv2.IMREAD_GRAYSCALE)
        img = GaussianNoise().apply(img)
        img = RandomPerspectiveTransform(0.2).apply(img)
        img = GaussianBlur().apply(img)

        plt.imshow(img)
        plt.imshow(img, cmap="gray")
        plt.axis('off')
        plt.show()

    def transform_sudoku(self):
        transform = RandomPerspectiveTransform()
        img = cv2.imread(f"../sudoku.jpeg", cv2.IMREAD_GRAYSCALE)
        transformed = transform.apply(img)
        plt.imshow(transformed, cmap="gray")
        plt.axis('off')
        plt.show()

    def generate_composition(self):
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
