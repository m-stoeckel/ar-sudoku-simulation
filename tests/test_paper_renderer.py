from unittest import TestCase

import numpy as np
from matplotlib import pyplot as plt

from digit.digit_dataset import Chars74KIRGBA
from sudoku.render.layers.base_layers import DigitalCompositionLayer, SubstrateLayer
from sudoku.render.renderer import LayeredPaperRenderer
from sudoku.render.util.digital_composition import DigitalCompositionMethod


class Test(TestCase):
    def test_add_element(self):
        printing_layer = DigitalCompositionLayer((2, 2))
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

    def test_digital_composition(self):
        alpha = np.zeros((256, 256, 4), dtype=np.uint8)
        for i in range(256):
            for j in range(256):
                alpha[i, j, 3] = i
        red = alpha.transpose((1, 0, 2)).copy().transpose((1, 0, 2))
        red[:, :, 0] = 255
        green = alpha.transpose((1, 0, 2)).copy()
        green[:, :, 1] = 255
        blue = alpha.transpose((1, 0, 2)).copy()
        blue[:, :, 2] = 255
        yellow = alpha.transpose((1, 0, 2)).copy()
        yellow[:, :, 0] = 255
        yellow[:, :, 1] = 255

        for Method in DigitalCompositionMethod.__subclasses__():
            method = Method()
            fig, axes = plt.subplots(1, 3, figsize=(9, 3.25))
            for ax, color in zip(axes, [green, blue, yellow]):
                layer = DigitalCompositionLayer((256, 256), composite=method)
                layer.add_element(red.copy())
                layer.add_element(color.copy())
                ax.axis('off')
                ax.imshow(layer.compose())
            fig.suptitle(f'{Method.__name__}')
            plt.show()

    def test_printing(self):
        res = 900
        substrate = SubstrateLayer(shape=(res, res), background_color=(255, 240, 240, int(0.8 * 255)))

        renderer = LayeredPaperRenderer(substrate)

        digits = Chars74KIRGBA(digits_path="../datasets/digits_hnd.zip", resolution=res)

        digit = digits.get_ordered(1, 0)
        renderer.drawing_layer.add_element(digit)

        digit = digits.get_ordered(9, 0)
        renderer.backside_drawing_layer.add_element(digit)

        img = renderer.render()
        self.plot(img)

        # def test_DigitalCompositionLayer(self):

    def plot(self, img):
        plt.figure(figsize=(9, 9))
        plt.axis('off')
        plt.tight_layout()
        plt.imshow(img)
        plt.show()

    #     plt.subplots(4, 4, figsize=(12, 12))
    #     for i, ralpha in enumerate([63, 127, 191, 255]):
    #         for j, balpha in enumerate([63, 127, 191, 255]):
    #             red = np.zeros((100, 100, 4), dtype=np.uint8)
    #             red[:, :, 0] = 255
    #             red[:, :, 3] = ralpha
    #
    #             blue = np.zeros((100, 100, 4), dtype=np.uint8)
    #             blue[:, :, 2] = 255
    #             blue[:, :, 3] = balpha
    #
    #             printing_layer = DigitalCompositionLayer((100, 100), bg_color=Color.WHITE)
    #             printing_layer.add_element(red)
    #             printing_layer.add_element(blue)
    #
    #             ax: Axes = plt.subplot2grid((4, 4), (i, j))
    #             ax.axis('off')
    #             ax.imshow(printing_layer.compose())
    #             ax.set_title(f"$\\alpha_r={ralpha}, \\alpha_b={balpha}$")
    #     plt.show()

    # def test_composites(self):
    #     alpha_composition = AlphaComposition()
    #
    #     for alpha in [64, 128, 192, 255]:
    #         red = np.zeros((100, 100, 4), dtype=np.uint8)
    #         red[:, :, 0] = 255
    #         red[:, :, 3] = alpha
    #
    #         blue = np.zeros((100, 100, 4), dtype=np.uint8)
    #         blue[:, :, 2] = 255
    #         blue[:, :, 3] = alpha
    #
    #         img_1 = red.copy()
    #         img_1[:, 50:] = blue[:, 50:]
    #         img_2 = alpha_composition(red, blue)
    #         img_3 = alpha_composition(red, blue)
    #         img_4 = cv2.addWeighted(red, 0.5, blue, 0.5, 0)
    #
    #         self.plot(alpha, img_1, img_2, img_3, img_4)
    #
    #         img_1 = red.copy()
    #         img_1[:, 50:] = blue[:, 50:]
    #         img_3 = alpha_composition(red, blue)
    #         img_4 = alpha_composition(red, blue)
    #
    #         self.plot(alpha, img_1, img_2, img_3, img_4)
    #
    # def plot(self, alpha, img_1, img_2, img_3, img_4):
    #     plt.figure(figsize=(6, 2))
    #     plt.axis('off')
    #     plt.tight_layout()
    #     plt.imshow(np.hstack((img_1, img_2, img_3, img_4)))
    #     plt.title(f"$\\alpha={alpha}$")
    #     plt.show()

    @staticmethod
    def scale_color_by_alpha(color):
        alpha_scale = color[:, :, 3].astype(np.float)[:, :, np.newaxis] / 255
        scaled = color[:, :, :3].astype(np.float) * alpha_scale
        return scaled.astype(np.uint8), alpha_scale
