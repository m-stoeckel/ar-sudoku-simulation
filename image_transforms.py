import cv2
import numpy as np
from matplotlib import pyplot as plt


class ImageTransform:
    def apply(self, img: np.ndarray) -> np.ndarray:
        pass


class RandomPerspectiveTransform(ImageTransform):
    def __init__(self, max_shift=0.25, background_color=None):
        self.max_shift = max_shift
        self.bg = background_color
        self.bg_mode = cv2.BORDER_REPLICATE if self.bg is None else cv2.BORDER_CONSTANT
        self.flags = [cv2.INTER_LINEAR]

    def apply(self, img: np.ndarray) -> np.ndarray:
        mat = self.get_transform_matrix(img)
        img = cv2.warpPerspective(img, mat, img.size, flags=self.flags, borderMode=self.bg_mode, borderValue=self.bg)
        return img

    def get_transform_matrix(self, img):
        """
        TODO
        """
        x_dim, y_dim = img.shape[:2]
        x_pos = np.random.randint(0, np.floor(x_dim * self.max_shift) + 1, 4)
        y_pos = np.random.randint(0, np.floor(y_dim * self.max_shift) + 1, 4)
        pa = [(x_pos[0], y_pos[0]),
              (x_dim - x_pos[1], y_pos[1]),
              (x_dim - x_pos[2], y_dim - y_pos[2]),
              (x_pos[3], y_dim - y_pos[3])]
        pb = [(0, 0), (x_dim, 0), (x_dim, y_dim), (0, y_dim)]
        return cv2.getPerspectiveTransform(pa, pb)


class RandomPerspectiveTransformBackwards(RandomPerspectiveTransform):
    def __init__(self, max_shift=0.25, background_color=None):
        super().__init__(max_shift, background_color)
        self.flags = [cv2.INTER_LINEAR, cv2.WARP_INVERSE_MAP]


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


def test_camera():
    img = cv2.imread("sudoku.jpeg", cv2.IMREAD_GRAYSCALE)
    transform = LensDistortion(dist_coeffs=np.array([0, 0.5, 0, 0, 0]))
    img = transform.apply(img)

    plt.imshow(img)
    plt.imshow(img, cmap="gray")
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    test_camera()
