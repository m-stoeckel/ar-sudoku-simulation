from abc import abstractmethod, ABCMeta

import numpy as np


class ImageTransform(metaclass=ABCMeta):
    """Base class for all image transforms"""

    @abstractmethod
    def apply(self, img: np.ndarray) -> np.ndarray:
        """
        Apply the transformation to the input image.

        Args:
            img(:py:class:`numpy.ndarray`): The input image, as a numpy array.

        Returns:
            :py:class:`numpy.ndarray`: A new :py:class:`numpy.ndarray` containing the transformed image.

        """
        pass
