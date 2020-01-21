from typing import Union

import numpy as np
from PIL import Image


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


class CameraTransformTwoD(ImageTransform):
    def __init__(self, f=None, c=None, angles=None, translations=None):
        self.f = [500, 500] if f is None else f
        self.c = [400, 300] if c is None else c

        self.angles = [np.radians(0), np.radians(0), np.radians(0)] if angles is None else angles
        self.translations = [0, 0, 0] if translations is None else translations

    def compute_Rt(self):
        X_rot = np.array([
            [1, 0, 0],
            [0, np.cos(self.angles[0]), -np.sin(self.angles[0])],
            [0, np.sin(self.angles[0]), np.cos(self.angles[0])]
        ])
        Y_rot = np.array([
            [np.cos(self.angles[1]), 0, np.sin(self.angles[1])],
            [0, 1, 0],
            [-np.sin(self.angles[1]), 0, np.cos(self.angles[1])]
        ])
        ret = np.zeros((3, 3))
        ret = np.eye(3)
        ret = np.dot(X_rot, ret)
        ret = np.dot(Y_rot, ret)
        ret[:, 2] = np.array(self.translations).reshape(3, 1)

        return ret

    def apply(self, img: Union[np.ndarray, Image.Image]):
        if type(img) is Image.Image:
            img: np.ndarray = np.array(img)
        arr = img.reshape(-1, 2)
        arr = np.hstack(arr, np.ones(arr.shape[0]))
        Rt = self.compute_Rt()
        M = np.array([[self.f[0], 0, self.c[0]], [0, self.f[1], self.c[1]], [0, 0, 1]])
        arr = np.dot(M, np.dot(Rt, arr))
        return (arr / arr[-1])[:, :1].reshape(img.shape)
