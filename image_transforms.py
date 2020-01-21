from typing import Union

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


class ImageTransform:
    def apply(self, img: Union[np.ndarray, Image.Image]) -> Image.Image:
        pass


class RandomPerspectiveTransform(ImageTransform):
    def __init__(self, max_shift=0.25):
        self.max_shift = max_shift

    def apply(self, img: Union[np.ndarray, Image.Image], return_mode="L") -> Image.Image:
        if type(img) is np.ndarray:
            img: Image.Image = Image.fromarray(img, mode="RGBA")
        else:
            img = img.convert("RGBA")
        x_dim, y_dim = img.size
        x_pos = np.random.randint(0, np.floor(x_dim * self.max_shift) + 1, 4)
        y_pos = np.random.randint(0, np.floor(y_dim * self.max_shift) + 1, 4)
        transform = [(x_pos[0], y_pos[0]),
                     (x_dim - x_pos[1], y_pos[1]),
                     (x_dim - x_pos[2], y_dim - y_pos[2]),
                     (x_pos[3], y_dim - y_pos[3])]
        img = self.perspective_transform(img, transform)
        return img.convert(return_mode)

    def perspective_transform(self, img: Image.Image, transform) -> Image.Image:
        x_dim, y_dim = img.size
        coeffs = self.find_coeffs(transform, x_dim, y_dim)
        img = img.transform(
            (x_dim, y_dim),
            Image.PERSPECTIVE, coeffs,
            Image.BICUBIC
        )
        return img

    def find_coeffs(self, pa, x_dim, y_dim):
        """
        Maps the image with its corner points at pa back to their origin and thus making a perspective transform.
        Point order: TL, TR, BR, BL
        Changing the origin
        https://stackoverflow.com/a/14178717
        """
        pb = [(0, 0), (x_dim, 0), (x_dim, y_dim), (0, y_dim)]
        matrix = []
        for p1, p2 in zip(pa, pb):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

        A = np.matrix(matrix, dtype=np.float)
        B = np.array(pb).reshape(8)

        res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
        return np.array(res).reshape(8)


class RandomPerspectiveTransformBackwards(RandomPerspectiveTransform):
    def __init__(self, max_shift=0.25, mode='l'):
        super().__init__(max_shift, mode)

    def find_coeffs(self, pb, x_dim=128, y_dim=128):
        """
        Maps the image with its corner points at pa back to their origin and thus making a perspective transform.
        Point order: TL, TR, BR, BL
        Changing the origin
        https://stackoverflow.com/a/14178717
        """
        pa = [(0, 0), (x_dim, 0), (x_dim, y_dim), (0, y_dim)]
        matrix = []
        for p1, p2 in zip(pa, pb):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

        A = np.matrix(matrix, dtype=np.float)
        B = np.array(pb).reshape(8)

        res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
        return np.array(res).reshape(8)


class LensDistortion(ImageTransform):
    def __init__(self, focal_lengths=None, dist_coeffs=None, principal_point=None):
        self.dist_coeffs = np.array([0, 0, 0, 0]) if dist_coeffs is None else dist_coeffs

        self.f = [500, 500] if focal_lengths is None else focal_lengths
        self.c = principal_point

    def apply(self, img: Union[np.ndarray, Image.Image]):
        if type(img) is not np.ndarray:
            img: np.ndarray = np.array(img)
        c = self.c if self.c is not None else np.floor(np.array(img.shape) / 2)
        camera_matrix = np.array([[self.f[0], 0, c[0]], [0, self.f[1], c[1]], [0, 0, 1]])
        img = cv2.undistort(img, camera_matrix, self.dist_coeffs)
        return Image.fromarray(img)


def test_camera():
    img = Image.open("sudoku.jpeg").convert('L')
    transform = LensDistortion(dist_coeffs=np.array([0, 0.5, 0, 0, 0]))
    img = transform.apply(img)

    plt.imshow(img)
    plt.imshow(img, cmap="gray")
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    test_camera()
