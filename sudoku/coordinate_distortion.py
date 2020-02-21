import cv2
import numpy as np
from matplotlib import pyplot as plt

from sudoku.sudoku_generator import SudokuGenerator

if __name__ == '__main__':
    img, coords = SudokuGenerator.get_sudoku_grid(cell_size=10)
    l = []
    for i in range(len(coords)):
        for j in range(len(coords)):
            l.append([coords[i] - 1, coords[j] - 1])
    coords = np.array(l).T

    mask = np.full(img.shape, False, dtype=np.bool)
    mask[coords[0], coords[1]] = True
    dots = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
    dots[:, :, [0, 3]] = 254
    dots = dots * mask

    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap="gray")
    plt.imshow(dots)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    x_dim, y_dim = img.shape[:2]

    x_pos = np.array([16, 16, 0, 0], dtype=np.float32)
    y_pos = np.zeros(4, dtype=np.float32)

    src = np.array([[0, 0], [x_dim, 0], [x_dim, y_dim], [0, y_dim]], dtype=np.float32)
    dst = np.array([[x_pos[0], y_pos[0]],
                    [x_dim - x_pos[1], y_pos[1]],
                    [x_dim - x_pos[2], y_dim - y_pos[2]],
                    [x_pos[3], y_dim - y_pos[3]]], dtype=np.float32)
    mat = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(img, mat, dsize=img.shape[:2])
    coords = np.vstack((coords, np.ones(coords.shape[1])))
    coords = mat @ coords
    coords /= coords[2]
    coords = np.round(coords[:2])
    coords = coords.astype(np.int)
    coords = np.flipud(coords)

    mask = np.full(img.shape, False, dtype=np.bool)
    mask[coords[0], coords[1]] = True
    dots = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
    dots[:, :, [0, 3]] = 254
    dots = dots * mask

    plt.figure(figsize=(5, 5))
    plt.imshow(warped, cmap="gray")
    plt.imshow(dots)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    f = [500, 500]
    c = np.floor(np.array(warped.shape[:2]) / 2)
    camera_matrix = np.array([[f[0], 0, c[0]], [0, f[1], c[1]], [0, 0, 1]])
    dist_coeffs = np.array([10, 0, 0, 0, 0])

    distored = cv2.undistort(warped, camera_matrix, dist_coeffs)
    coords = np.expand_dims(coords.T, 1)
    coords = cv2.undistortPoints(coords.astype(np.float32), camera_matrix, dist_coeffs, None, camera_matrix)
    coords = np.squeeze(coords).T
    coords = np.round(coords).astype(np.int)
    print(coords)

    mask = np.full(img.shape, False, dtype=np.bool)
    mask[coords[0], coords[1]] = True
    dots = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
    dots[:, :, [0, 3]] = 254
    dots = dots * mask

    plt.figure(figsize=(5, 5))
    plt.imshow(distored, cmap="gray")
    plt.imshow(dots)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
