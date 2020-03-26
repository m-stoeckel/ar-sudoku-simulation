from typing import Union

import cv2
import numpy as np

from simulation.transforms.base import ImageTransform


class BoxBlur(ImageTransform):
    def __init__(self, ksize=2):
        super().__init__()
        self.ksize = ksize if isinstance(ksize, tuple) else (ksize, ksize)

    def apply(self, img: np.ndarray) -> np.ndarray:
        return cv2.blur(img, self.ksize)


class GaussianBlur(ImageTransform):
    def __init__(self, ksize: Union[int, tuple] = 3, sigma=0):
        super().__init__()
        self.ksize = ksize if isinstance(ksize, tuple) else (ksize, ksize)
        self.sigma = sigma

    def apply(self, img: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(img, self.ksize, self.sigma)


class Dilate(ImageTransform):
    def __init__(self, kernel: Union[np.ndarray, int] = 3):
        super().__init__()
        if isinstance(kernel, int):
            self.kernel = np.ones((kernel, kernel))
        else:
            self.kernel = kernel

    def apply(self, img: np.ndarray) -> np.ndarray:
        return cv2.dilate(img, self.kernel, iterations=1)


class Convolve(ImageTransform):
    def __init__(self, passes):
        self.passes = passes
        self.kernel = np.array([[0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0]])

    def apply(self, img: np.ndarray) -> np.ndarray:
        for _ in range(self.passes):
            img = cv2.filter2D(img.astype(np.float), -1, self.kernel)
        return img.astype(np.uint8)


class SharpenFilter(Convolve):
    def __init__(self, passes=1):
        super().__init__(passes)
        self.kernel = np.array([[0, 0, 0, 0, 0],
                                [0, 0, -1, 0, 0],
                                [0, -1, 5, -1, 0],
                                [0, 0, -1, 0, 0],
                                [0, 0, 0, 0, 0]])


class ReliefFilter(Convolve):
    def __init__(self, passes=1):
        super().__init__(passes)
        self.kernel = np.array([[0, 0, 0, 0, 0],
                                [0, -2, -1, 0, 0],
                                [0, -1, 1, 1, 0],
                                [0, 0, 1, 2, 0],
                                [0, 0, 0, 0, 0]])


class EdgeFilter(Convolve):
    def __init__(self, passes=1):
        super().__init__(passes)
        self.kernel = np.array([[0, 0, 0, 0, 0],
                                [0, 1 / 4., 2 / 4., 1 / 4., 0],
                                [0, 2 / 4., -12 / 4., 2 / 4., 0],
                                [0, 1 / 4., 2 / 4., 1 / 4., 0],
                                [0, 0, 0, 0, 0]])


class UnsharpMaskingFilter(Convolve):
    def __init__(self, passes=1):
        super().__init__(passes)
        self.kernel = np.array([[-1 / 256., -4 / 256., -6 / 256., -4 / 256., -1 / 256.],
                                [-4 / 256., -16 / 256., -24 / 256., -16 / 256., -4 / 256.],
                                [-6 / 256., -24 / 256., 476 / 256., -24 / 256., -6 / 256.],
                                [-4 / 256., -16 / 256., -24 / 256., -16 / 256., -4 / 256.],
                                [-1 / 256., -4 / 256., -6 / 256., -4 / 256., -1 / 256.]])
