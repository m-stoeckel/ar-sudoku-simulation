from unittest import TestCase

import cv2
import numpy as np
from matplotlib import pyplot as plt

from simulation.image import GaussianNoise, RandomPerspectiveTransform, GaussianBlur
from simulation.image.image_transforms import EmbedInGrid


class ImageTransformTests(TestCase):
    def test_with_sudoku(self):
        img = cv2.imread("../../sudoku.jpeg", cv2.IMREAD_GRAYSCALE)
        img = GaussianNoise().apply(img)
        img = RandomPerspectiveTransform(0.2).apply(img)
        img = GaussianBlur().apply(img)

        plt.imshow(img)
        plt.imshow(img, cmap="gray")
        plt.axis('off')
        plt.show()

    def test_embed(self):
        digit = cv2.imread("../../datasets/digits/1.png")
        cv2.bitwise_not(digit, digit)
        transform = EmbedInGrid()
        tdigit = transform.apply(digit)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2, 1))
        ax1.axis('off')
        ax2.axis('off')
        ax1.imshow(digit)
        ax2.imshow(tdigit)
        plt.show()

    def transform_sudoku(self):
        transform = RandomPerspectiveTransform()
        img = cv2.imread(f"../../sudoku.jpeg", cv2.IMREAD_GRAYSCALE)
        transformed = transform.apply(img)
        plt.imshow(transformed, cmap="gray")
        plt.axis('off')
        plt.show()

    def generate_composition(self):
        transform = RandomPerspectiveTransform()
        images = [[] for _ in range(9)]
        for i in range(0, 9):
            img = cv2.imread(f"../datasets/digits/{i * 917}.png", cv2.IMREAD_GRAYSCALE)
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
