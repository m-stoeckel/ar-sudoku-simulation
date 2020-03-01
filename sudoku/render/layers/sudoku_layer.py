from typing import Union

import cv2
import numpy as np

from digit.digit_dataset import MNIST, plt, DigitDataset
from sudoku.render.layers.base_layers import DigitalCompositionLayer
from sudoku.render.util.colors import Color

DEBUG = True


@DeprecationWarning
class SudokuLayer(DigitalCompositionLayer):
    def __init__(self, shape: tuple, **kwargs):
        super().__init__(shape, **kwargs)

    @staticmethod
    def get_sudoku_grid(cell_size=28, major_line_width=2, minor_line_width=1):
        grid_size = cell_size * 9 + 4 * major_line_width + 6 * minor_line_width

        data = np.ndarray((grid_size, grid_size, 4), dtype=np.uint8)
        data.fill(255)
        coords = np.zeros(10, dtype=np.int)
        idx = 0
        for i in range(10):
            if i % 3 == 0:  # draw major line
                data[idx:idx + major_line_width, :, 0:3] = 0
                data[:, idx:idx + major_line_width, 0:3] = 0
                coords[i] = idx + major_line_width
                idx += major_line_width + cell_size
            else:  # draw minor line
                data[idx:idx + minor_line_width, :, 0:3] = 0
                data[:, idx:idx + minor_line_width, 0:3] = 0
                coords[i] = idx + minor_line_width
                idx += minor_line_width + cell_size

        return data, coords

    @staticmethod
    def draw_sudoku_pil(sudoku, masking_rate=0.7):
        mask = np.random.choice([True, False], size=sudoku.shape, p=[masking_rate, 1 - masking_rate])

        grid_image, coords = SudokuLayer.get_sudoku_grid()

        x_offset = 5
        y_offset = 4
        for i in range(9):
            for j in range(9):
                digit = sudoku[i][j]
                x_coord, y_coord = coords[i], coords[j]
                if digit == 0:
                    continue
                elif not mask[i][j]:
                    (text_width, text_height) = cv2.getTextSize(str(digit), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)[0]
                    grid_image = cv2.putText(grid_image, str(digit),
                                             (x_coord + x_offset, y_coord + text_height + y_offset),
                                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, Color.BLACK.value, 1, cv2.LINE_AA)
        return grid_image

    @staticmethod
    def draw_sudoku_pil_mnist(sudoku, masking_rate=0.7, digits: Union[MNIST, DigitDataset] = None, hw_rate=0.2,
                              cell_size=28):
        # Mask setup
        printed_digit_mask = np.random.choice([False, True], size=sudoku.shape, p=[masking_rate, 1 - masking_rate])
        mnist_digit_mask = np.random.choice([True, False], size=sudoku.shape, p=[hw_rate, 1 - hw_rate])
        mnist_digit_mask[printed_digit_mask] = False

        # Image data setup
        grid_image, coords = SudokuLayer.get_sudoku_grid(cell_size=cell_size)
        mnist_image = np.zeros(grid_image.shape, dtype=np.uint8)

        # Drawing
        x_offset = 5
        y_offset = 4
        for i in range(9):
            for j in range(9):
                digit = sudoku[i][j]
                x_coord, y_coord = coords[i], coords[j]
                if digit == 0:
                    continue
                if printed_digit_mask[i][j]:
                    (text_width, text_height) = cv2.getTextSize(str(digit), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)[0]
                    grid_image = cv2.putText(grid_image, str(digit),
                                             (x_coord + x_offset, y_coord + text_height + y_offset),
                                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, Color.BLACK.value, 1, cv2.LINE_AA)
                elif mnist_digit_mask[i][j]:
                    digit_index = i * 10 + j
                    mnist_digit = np.array(digits.get_ordered(digit, digit_index), dtype=np.uint8)
                    mnist_digit = mnist_digit.repeat(4).reshape((cell_size, cell_size, 4))
                    mnist_digit = cv2.bitwise_not(mnist_digit)

                    mask = SudokuLayer.get_bool_mask_from_color(mnist_digit)
                    mnist_digit[:, :, 3] = 0
                    mnist_digit[mask, 3] = 255
                    mnist_image[y_coord:y_coord + cell_size, x_coord:x_coord + cell_size] = mnist_digit

        # Image composition
        img = SudokuLayer.composite_threshold(grid_image, mnist_image)
        if DEBUG:
            plt.figure(figsize=(3, 3))
            plt.imshow(img)
            plt.axis('off')
            plt.show()
        return img

    @staticmethod
    def composite_threshold(background, foreground, reset_alpha=True):
        mask = SudokuLayer.get_mask_from_alpha(foreground)
        img = cv2.bitwise_and(background, background, mask=mask)
        img = cv2.add(img, foreground)
        if reset_alpha:
            img[:, :, 3] = 0
            img[SudokuLayer.get_bool_mask_from_alpha(img), 3] = 255
        return img

    @staticmethod
    def get_mask_from_alpha(image):
        _, mask = cv2.threshold(image[:, :, 3], 1, 255, cv2.THRESH_BINARY)
        return cv2.bitwise_not(mask)

    @staticmethod
    def get_bool_mask_from_alpha(image):
        return SudokuLayer.get_mask_from_alpha(image).astype(np.bool)

    @staticmethod
    def get_mask_from_color(image):
        mask = SudokuLayer.get_bool_mask_from_color(image)
        ret = np.zeros_like(mask, 0, dtype=np.uint8)
        ret[mask] = 255
        return ret

    @staticmethod
    def get_bool_mask_from_color(image):
        return np.any(image[:, :, :3] != 255, axis=2)
