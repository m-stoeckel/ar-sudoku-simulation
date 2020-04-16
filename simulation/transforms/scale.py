from typing import Tuple, List

import cv2
import numpy as np

from simulation.transforms import ImageTransform


class Rescale(ImageTransform):
    """
    Reduce the detail of images by resizing it to a given resolution
    and consecutively scaling it back to its original size again.

    """

    def __init__(self, size: Tuple[int, int], inter_initial=cv2.INTER_AREA, inter_consecutive=cv2.INTER_LINEAR):
        """
        Reduce the detail of images by resizing it to a given resolution
        and consecutively scaling it back to its original size again.

        Args:
            size(tuple[int, int]): The intermediate size of the transforms.
            inter_initial(inter_initial: int, optional): The initial OpenCV interpolation algorithm.
                (Default value = cv2.INTER_AREA)
            inter_consecutive(inter_consecutive: int, optional): The consecutive rescaling OpenCV interpolation
                algorithm. (Default value = cv2.INTER_LINEAR)

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

    def __init__(
            self,
            size: Tuple[int, int],
            intermediate_transforms: List[ImageTransform],
            inter_initial=cv2.INTER_AREA,
            inter_consecutive=cv2.INTER_LINEAR
    ):
        """
        Reduce the detail of images by resizing it to a given resolution, applying a series of intermediate transforms
        and consecutively scaling it back to its original size again.

        Args:
            size(tuple[int, int]): The intermediate size of the transforms.
            intermediate_transforms(list[ImageTransform]): A list of transforms to be applied after the initial rescale.
            inter_initial(inter_initial: int, optional): The initial OpenCV interpolation algorithm.
                (Default value = cv2.INTER_AREA)
            inter_consecutive(inter_consecutive: int, optional): The consecutive rescaling OpenCV interpolation
                algorithm. (Default value = cv2.INTER_LINEAR)

        """
        super().__init__(size, inter_initial, inter_consecutive)
        self.intermediate_transforms = intermediate_transforms

    def add_transforms(self, *transforms: ImageTransform):
        """
        Add a sequence of intermediate transforms.

        Args:
            transforms(Iterable[ImageTransform]): Sequence of transforms to be added.

        Returns:
            None

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
