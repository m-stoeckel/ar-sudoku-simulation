from typing import Union, Tuple, List

import cv2
import numpy as np

from simulation.render import Color


class ImageTransform:
    def apply(self, img: np.ndarray) -> np.ndarray:
        """
        Apply the transformation to the input image.

        :param img: The input image, as a numpy array.
        :return: The transformed image.
        """
        pass


class BoxBlur(ImageTransform):
    def __init__(self, ksize=2):
        self.ksize = ksize if isinstance(ksize, tuple) else (ksize, ksize)
        pass

    def apply(self, img: np.ndarray) -> np.ndarray:
        return cv2.blur(img, self.ksize)


class GaussianNoise(ImageTransform):
    def __init__(self, mu=0, sigma=8):
        self.mu = mu
        self.sigma = sigma
        pass

    def noise(self, size):
        return np.random.normal(self.mu, self.sigma, size)

    def apply(self, img: np.ndarray) -> np.ndarray:
        noise = self.noise(img.shape)
        img = img.astype(np.float) + noise
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)


class GaussianBlur(ImageTransform):
    def __init__(self, ksize: Union[int, tuple] = 3, sigma=0):
        self.ksize = ksize if isinstance(ksize, tuple) else (ksize, ksize)
        self.sigma = sigma
        pass

    def apply(self, img: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(img, self.ksize, self.sigma)


class RandomPerspectiveTransform(ImageTransform):
    """
    Applies a homographic perspective transform to images.
    The image is mapped from its original space to a narrowed space.
    """
    flags = cv2.INTER_LINEAR

    def __init__(self, max_shift=0.25, background_color=None):
        self.max_shift = max_shift
        self.bg = background_color
        self.bg_mode = cv2.BORDER_REPLICATE if self.bg is None else cv2.BORDER_CONSTANT

    def apply(self, img: np.ndarray) -> np.ndarray:
        mat = self.get_transform_matrix(img)
        img = cv2.warpPerspective(img, mat, img.shape[:2], flags=self.flags,
                                  borderMode=self.bg_mode, borderValue=self.bg)
        return img

    def get_transform_matrix(self, img):
        """
        Compute the homographic matrix H.
        """
        x_dim, y_dim = img.shape[:2]
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
        return np.random.randint(0, np.floor(x_dim * self.max_shift) + 1, 4)

    def get_y_displacement(self, x_dim):
        return np.random.randint(0, np.floor(x_dim * self.max_shift) + 1, 4)


class RandomPerspectiveTransformBackwards(RandomPerspectiveTransform):
    """
    Applies a backwards homographic perspective transform to images.
    In contrast to RandomPerspectiveTransform, the homography is computed the other way around, mapping from a given
    space to the original image space.
    """
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP


class RandomPerspectiveTransformX(RandomPerspectiveTransform):
    """
    Applies a homographic perspective transform to images, only without modification along the Y axis of the image
    (only left or right tilt).
    """

    def get_y_displacement(self, x_dim):
        return np.zeros(4)


class RandomPerspectiveTransformY(RandomPerspectiveTransform):
    """
    Applies a homographic perspective transform to images, only without modification along the X axis of the image
    (only forward or backward tilt).
    """

    def get_x_displacement(self, x_dim):
        return np.zeros(4)


class LensDistortion(ImageTransform):
    def __init__(self, focal_lengths=None, dist_coeffs=None, principal_point=None):
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


class EmbedInGrid(ImageTransform):
    def __init__(self, inset=0.2):
        self.inset = inset
        self.offset = inset / 2

    def apply(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3:
            grid_image_shape = (
                int(img.shape[0] + self.inset * img.shape[0]),
                int(img.shape[1] + self.inset * img.shape[1]),
                img.shape[2]
            )
        else:
            grid_image_shape = (
                int(img.shape[0] + self.inset * img.shape[0]),
                int(img.shape[1] + self.inset * img.shape[1]),
            )
        grid_image = np.full(grid_image_shape, 255, dtype=np.uint8)
        offset_x, offset_y = int(self.offset * img.shape[0]), int(self.offset * img.shape[1])
        grid_image[offset_x:offset_x + img.shape[0], offset_y:offset_y + img.shape[1]] = img
        cv2.rectangle(grid_image, (offset_x, offset_y),
                      (grid_image.shape[0] - offset_x, grid_image.shape[1] - offset_y),
                      Color.BLACK.value, thickness=int(img.shape[0] * 0.05))
        return self.random_crop(grid_image, img.shape)

    @staticmethod
    def random_crop(grid_img, shape) -> np.ndarray:
        offset = np.array(grid_img.shape) - np.array(shape)
        offset_x = np.random.randint(0, offset[0])
        offset_y = np.random.randint(0, offset[1])
        return grid_img[offset_x:offset_x + shape[0], offset_y:offset_y + shape[1]]


class Rescale(ImageTransform):
    """
    Reduce the detail of images by resizing it to a given resolution
    and consecutively scaling it back to its original size again.
    """

    def __init__(self, size: Tuple[int, int], inter_initial=cv2.INTER_AREA, inter_consecutive=cv2.INTER_LINEAR):
        """
        Reduce the detail of images by resizing it to a given resolution
        and consecutively scaling it back to its original size again.

        :param size: The intermediate size of the image.
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
        cv2.resize(img, self.size, dst=img, interpolation=self.inter_initial)
        cv2.resize(img, orig_size, dst=img, interpolation=self.inter_consecutive)
        return img


class RescaleIntermediateTransforms(Rescale):
    """
    Reduce the detail of images by resizing it to a given resolution, applying a series of intermediate transforms and
    consecutively scaling it back to its original size again.
    """

    def __init__(self, size: Tuple[int, int], intermediate_transforms: List[ImageTransform],
                 inter_initial=cv2.INTER_AREA,
                 inter_consecutive=cv2.INTER_LINEAR):
        """
        Reduce the detail of images by resizing it to a given resolution, applying a series of intermediate transforms
        and consecutively scaling it back to its original size again.

        :param size: The intermediate size of the image.
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
        self.intermediate_transforms.extend(transforms)

    def apply(self, img: np.ndarray) -> np.ndarray:
        orig_size = tuple(img.shape[:2])
        cv2.resize(img, self.size, dst=img, interpolation=self.inter_initial)

        # Apply intermediate transforms
        for transform in self.intermediate_transforms:
            img = transform.apply(img)

        cv2.resize(img, orig_size, dst=img, interpolation=self.inter_consecutive)
        return img
