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
        cv2.dilate(img, self.kernel, img)
        return img