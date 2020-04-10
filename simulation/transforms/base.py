from abc import abstractmethod, ABCMeta

import numpy as np


class ImageTransform(metaclass=ABCMeta):
    """
    Base class for all image transforms

    """

    @abstractmethod
    def apply(self, img: np.ndarray) -> np.ndarray:
        """
        Apply the transformation to the input image.

        :param img: The input image, as a numpy array.
        :type img: numpy.ndarray
        :return: A new ndarray containing the transformed image.
        :rtype: numpy.ndarray
        """
        pass
