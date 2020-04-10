from typing import *

import cv2
import numpy as np

from simulation.transforms.base import ImageTransform


class RandomPerspectiveTransform(ImageTransform):
    """
    Applies a homographic perspective transform to images.
    The transforms is mapped from its original space to a narrowed space.
    """
    flags = cv2.INTER_LINEAR

    def __init__(self, max_shift=0.25, background_color: int = 0):
        """
        :param max_shift: The maximum amount of shift on either axis.
        :type max_shift: float
        :param background_color: The background color to fill in. If None, cv2.BORDER_REPLICATE will be used.
        :type background_color: int
        """
        super().__init__()
        self.max_shift = max_shift
        self.bg = background_color
        self.bg_mode = cv2.BORDER_REPLICATE if self.bg is None else cv2.BORDER_CONSTANT

    def apply(self, img: np.ndarray) -> np.ndarray:
        mat = self.get_transform_matrix(img.shape)
        img = cv2.warpPerspective(img, mat, img.shape[:2], flags=self.flags,
                                  borderMode=self.bg_mode, borderValue=self.bg)
        return img

    def get_transform_matrix(self, shape):
        """
        Compute the homographic matrix H.

        :param shape: The shape of the image to transform.
        :type shape: Tuple[int, int]
        :return: The 3x3 perspective transform matrix.
        :rtype:
        """
        x_dim, y_dim = shape[:2]
        x_dim, y_dim = x_dim - 1, y_dim - 1
        x_pos = self.get_x_displacement(x_dim)
        y_pos = self.get_y_displacement(y_dim)
        pa = np.array([[0, 0], [x_dim, 0], [x_dim, y_dim], [0, y_dim]], dtype=np.float32)
        pb = np.array([[x_pos[0], y_pos[0]],
                       [x_dim - x_pos[1], y_pos[1]],
                       [x_dim - x_pos[2], y_dim - y_pos[2]],
                       [x_pos[3], y_dim - y_pos[3]]], dtype=np.float32)
        return cv2.getPerspectiveTransform(pa, pb)

    def get_x_displacement(self, x_dim):
        """
        Get the displacement along the x-axis.

        :param x_dim: The shape of input image along the x-axis.
        :type x_dim: int
        :return: A 4-dimensional array.
        :rtype: numpy.ndarray
        """
        return np.random.randint(0, np.floor(x_dim * self.max_shift) + 1, 4)

    def get_y_displacement(self, y_dim):
        """
        Get the displacement along the y-axis.

        :param y_dim: The shape of input image along the y-axis.
        :type y_dim: int
        :return: A 4-dimensional array.
        :rtype: numpy.ndarray
        """
        return np.random.randint(0, np.floor(y_dim * self.max_shift) + 1, 4)


class RandomPerspectiveTransformBackwards(RandomPerspectiveTransform):
    """
    Applies a backwards homographic perspective transform to images.
    In contrast to RandomPerspectiveTransform, the homography is computed the other way around, mapping from a given
    space to the original transforms space.
    """
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP


class RandomPerspectiveTransformX(RandomPerspectiveTransform):
    """
    Applies a homographic perspective transform to images, only without modification along the Y axis of the transforms
    (only left or right tilt).
    """

    def get_y_displacement(self, _):
        """
        Returns a 4-dimensional 0-array.

        :return: A 4-dimensional 0-array.
        :rtype: numpy.ndarray
        """
        return np.zeros(4)


class RandomPerspectiveTransformY(RandomPerspectiveTransform):
    """
    Applies a homographic perspective transform to images, only without modification along the X axis of the transforms
    (only forward or backward tilt).
    """

    def get_x_displacement(self, _):
        """
        Returns a 4-dimensional 0-array.

        :return: A 4-dimensional 0-array.
        :rtype: numpy.ndarray
        """
        return np.zeros(4)


class LensDistortion(ImageTransform):
    """
    Applies camera lens distortion to input images.
    """

    def __init__(self, focal_lengths: Tuple[float, float] = None, dist_coeffs: Iterable[float] = None,
                 principal_point: Tuple[int, int] = None):
        """
        :param focal_lengths: The focal lengths of the simulated lens. Default: [500, 500].
        :type focal_lengths: Tuple[float, float]
        :param dist_coeffs: The distance coefficients of the simulated lens. Default: [0,0,0,0].
        :type dist_coeffs: Iterable[float]
        :param principal_point: The principal point of the camera. If None, the center of each image will be used.
        :type principal_point: Tuple[int, int]
        """
        super().__init__()
        self.dist_coeffs = np.array([0, 0, 0, 0]) if dist_coeffs is None else dist_coeffs

        self.f = [500, 500] if focal_lengths is None else focal_lengths
        self.c = principal_point

    def apply(self, img: np.ndarray):
        if type(img) is not np.ndarray:
            img: np.ndarray = np.array(img)
        c = self.c if self.c is not None else np.floor(np.array(img.shape) / 2)
        camera_matrix = np.array([[self.f[0], 0, c[0]], [0, self.f[1], c[1]], [0, 0, 1]])
        img = cv2.undistort(img, camera_matrix, self.dist_coeffs)
        return img
