from typing import Tuple, List

import cv2
import numpy as np

from simulation.transforms.base import ImageTransform


class Rescale(ImageTransform):
    """
    Reduce the detail of images by resizing it to a given resolution
    and consecutively scaling it back to its original size again.
    """

    def __init__(self, size: Tuple[int, int], inter_initial=cv2.INTER_AREA, inter_consecutive=cv2.INTER_LINEAR):
        """
        Reduce the detail of images by resizing it to a given resolution
        and consecutively scaling it back to its original size again.

        :param size: The intermediate size of the transforms.
        :type size: Tuple[int, int]
        :param inter_initial: The initial OpenCV interpolation algorithm.
        :type inter_initial: int
        :param inter_consecutive: The consecutive rescaling OpenCV interpolation algorithm.
        :type inter_consecutive: int
        """
        self.size = size
        self.inter_initial = inter_initial
        self.inter_consecutive = inter_consecutive

    def apply(self, img: np.ndarray) -> np.ndarray:
        orig_size = tuple(img.shape[:2])
        img = cv2.resize(img, self.size, interpolation=self.inter_initial)
        img = cv2.resize(img, orig_size, interpolation=self.inter_consecutive)
        return img


class RescaleIntermediateTransforms(Rescale):
    """
    Reduce the detail of images by resizing it to a given resolution, applying a series of intermediate transforms and
    consecutively scaling it back to its original size again.
    """

    def __init__(self, size: Tuple[int, int], intermediate_transforms: List[ImageTransform],
                 inter_initial=cv2.INTER_AREA, inter_consecutive=cv2.INTER_LINEAR):
        """
        Reduce the detail of images by resizing it to a given resolution, applying a series of intermediate transforms
        and consecutively scaling it back to its original size again.

        :param size: The intermediate size of the transforms.
        :type size: Tuple[int, int]
        :param intermediate_transforms: A list of transforms to be applied after the initial rescale
        :type intermediate_transforms: List[ImageTransform]
        :param inter_initial: The initial OpenCV interpolation algorithm.
        :type inter_initial: int
        :param inter_consecutive: The consecutive rescaling OpenCV interpolation algorithm.
        :type inter_consecutive: int
        """
        super().__init__(size, inter_initial, inter_consecutive)
        self.intermediate_transforms = intermediate_transforms

    def add_transforms(self, *transforms: ImageTransform):
        """
        Add a sequence of intermediate transforms.

        :param transforms: Sequence of transforms to be added.
        :type transforms: Iterable[ImageTransform]
        :return: None
        """
        self.intermediate_transforms.extend(transforms)

    def apply(self, img: np.ndarray) -> np.ndarray:
        orig_size = tuple(img.shape[:2])
        img = cv2.resize(img, self.size, interpolation=self.inter_initial)

        # Apply intermediate transforms
        for transform in self.intermediate_transforms:
            img = transform.apply(img)

        img = cv2.resize(img, orig_size, interpolation=self.inter_consecutive)
        return img
