from unittest import TestCase

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from sudoku import Color
from sudoku.render.paper_renderer import DigitalCompositionLayer


class TestPrintingLayer(TestCase):
    def test_add_element(self):
        printing_layer = DigitalCompositionLayer()
        el_1 = np.array([[[1, 2, 3, 255], [1, 2, 3, 255]], [[4, 5, 6, 255], [1, 2, 3, 255]]], dtype=np.uint8)
        printing_layer.add_element(el_1)
        self.assertTrue(np.array_equal(printing_layer.elements[0], el_1))

        el_2 = np.array([[[128], [128]], [[255], [255]]], dtype=np.uint8)
        el_2_rgba = np.array(
            [[[128, 128, 128, 255], [128, 128, 128, 255]], [[255, 255, 255, 255], [255, 255, 255, 255]]],
            dtype=np.uint8)
        printing_layer.add_element(el_2)
        self.assertTrue(np.array_equal(printing_layer.elements[1], el_2_rgba),
                        f"{printing_layer.elements[1]} != {el_2_rgba}")

        el_3 = np.array([[1, 2, 3, 255], [1, 2, 3, 255], [1, 2, 3, 255], [1, 2, 3, 255]], dtype=np.uint8)
        self.assertRaises(ValueError, printing_layer.add_element, el_3)

    def test_DigitalCompositionLayer(self):
        plt.subplots(4, 4, figsize=(12, 12))
        for i, ralpha in enumerate([63, 127, 191, 255]):
            for j, balpha in enumerate([63, 127, 191, 255]):
                red = np.zeros((100, 100, 4), dtype=np.uint8)
                red[:, :, 0] = 255
                red[:, :, 3] = ralpha

                blue = np.zeros((100, 100, 4), dtype=np.uint8)
                blue[:, :, 2] = 255
                blue[:, :, 3] = balpha

                printing_layer = DigitalCompositionLayer((100, 100), bg_color=Color.WHITE)
                printing_layer.add_element(red)
                printing_layer.add_element(blue)

                ax: Axes = plt.subplot2grid((4, 4), (i, j))
                ax.axis('off')
                ax.imshow(printing_layer.compose())
                ax.set_title(f"$\\alpha_r={ralpha}, \\alpha_b={balpha}$")
        plt.show()

    def test_DigitalCompositionLayer_rect(self):
        red = np.zeros((256, 256, 4), dtype=np.uint8)
        red[:, :, 0] = 255
        blue = np.zeros((256, 256, 4), dtype=np.uint8)
        blue[:, :, 2] = 255
        for i in range(256):
            for j in range(256):
                red[i, j, 3] = i
                blue[i, j, 3] = j

        printing_layer = DigitalCompositionLayer((256, 256), bg_color=Color.WHITE)
        printing_layer.add_element(red)
        printing_layer.add_element(blue)

        plt.figure(figsize=(3, 3))
        plt.axis('off')
        plt.imshow(printing_layer.compose())
        plt.show()

    def test_composites(self):
        printing_layer = DigitalCompositionLayer((100, 100))

        for alpha in [64, 128, 192, 255]:
            red = np.zeros((100, 100, 4), dtype=np.uint8)
            red[:, :, 0] = 255
            red[:, :, 3] = alpha

            blue = np.zeros((100, 100, 4), dtype=np.uint8)
            blue[:, :, 2] = 255
            blue[:, :, 3] = alpha

            img_1 = red.copy()
            img_1[:, 50:] = blue[:, 50:]
            img_2 = printing_layer.composite_alpha(red, blue)
            img_3 = printing_layer.composite_alpha_precise(red, blue)
            img_4 = cv2.addWeighted(red, 0.5, blue, 0.5, 0)

            self.plot(alpha, img_1, img_2, img_3, img_4)

            img_1 = red.copy()
            img_1[:, 50:] = blue[:, 50:]
            img_3 = printing_layer.composite_add(red, blue)
            img_4 = printing_layer.composite_average(red, blue)

            self.plot(alpha, img_1, img_2, img_3, img_4)

    def plot(self, alpha, img_1, img_2, img_3, img_4):
        plt.figure(figsize=(6, 2))
        plt.axis('off')
        plt.tight_layout()
        plt.imshow(np.hstack((img_1, img_2, img_3, img_4)))
        plt.title(f"$\\alpha={alpha}$")
        plt.show()

    @staticmethod
    def scale_color_by_alpha(color):
        alpha_scale = color[:, :, 3].astype(np.float)[:, :, np.newaxis] / 255
        scaled = color[:, :, :3].astype(np.float) * alpha_scale
        return scaled.astype(np.uint8), alpha_scale
