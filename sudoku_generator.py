import multiprocessing as mp
import os
from pathlib import Path
from typing import Union, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.datasets import fetch_openml

BLACK = (0, 0, 0, 255)
WHITE = (255, 255, 255, 255)
WHITE_NO_ALPHA = (255, 255, 255, 0)


class MNIST:
    def __init__(self):
        print("Loading MNIST dataset")
        # Load data from https://www.openml.org/d/554
        os.makedirs('datasets/', exist_ok=True)
        X, Y = fetch_openml('mnist_784', version=1, return_X_y=True, data_home="datasets/", cache=True)
        self.x = np.array(X).reshape((70000, 28, 28))
        self.y = np.array(Y)
        self.indices_by_number = [np.flatnonzero(self.y == str(i)) for i in range(0, 10)]
        del X, Y

    def get_random(self, digit: int) -> np.ndarray:
        """
        Get a random sample of the given digit.

        :returns: 2D numpy array of 28x28 pixels
        """
        return self.x[np.random.choice(self.indices_by_number[digit])]


class Sudoku:
    # TODO: cite sources
    def __init__(self, data=None):
        if data is not None:
            self.data = data
        else:
            while True:
                n = 9
                self.data = np.zeros((n, n), dtype=np.int)
                rg = np.arange(1, n + 1)
                self.data[0, :] = np.random.choice(rg, n, replace=False)
                try:
                    for r in range(1, n):
                        for c in range(n):
                            col_rest = np.setdiff1d(rg, self.data[:r, c])
                            row_rest = np.setdiff1d(rg, self.data[r, :c])
                            avb1 = np.intersect1d(col_rest, row_rest)
                            sub_r, sub_c = r // 3, c // 3
                            avb2 = np.setdiff1d(np.arange(0, n + 1),
                                                self.data[sub_r * 3:(sub_r + 1) * 3, sub_c * 3:(sub_c + 1) * 3].ravel())
                            avb = np.intersect1d(avb1, avb2)
                            self.data[r, c] = np.random.choice(avb, size=1)
                    break
                except ValueError:
                    pass

    @staticmethod
    def from_array(data: np.ndarray):
        return Sudoku(data=data)

    @property
    def shape(self):
        return self.data.shape

    def __getitem__(self, item):
        return self.data[item]

    @staticmethod
    def solve(sudoku):
        if isinstance(sudoku, list):
            sudoku = np.array(sudoku)
        elif isinstance(sudoku, str):
            sudoku = np.loadtxt(sudoku, dtype=np.int, delimiter=",")
        rg = np.arange(sudoku.shape[0] + 1)
        while True:
            mt = sudoku.copy()
            while True:
                d = []
                d_len = []
                for i in range(sudoku.shape[0]):
                    for j in range(sudoku.shape[1]):
                        if mt[i, j] == 0:
                            possibles = np.setdiff1d(rg, np.union1d(np.union1d(mt[i, :], mt[:, j]),
                                                                    mt[3 * (i // 3):3 * (i // 3 + 1),
                                                                    3 * (j // 3):3 * (j // 3 + 1)]))
                            d.append([i, j, possibles])
                            d_len.append(len(possibles))
                if len(d) == 0:
                    break
                idx = np.argmin(d_len)
                i, j, p = d[idx]
                if len(p) > 0:
                    num = np.random.choice(p)
                else:
                    break
                mt[i, j] = num
                if len(d) == 0:
                    break
            if np.all(mt != 0):
                break

        print("\nTrail:\n", mt)
        return mt

    @property
    def is_valid(self):
        return Sudoku.check_solution(self.data)

    @staticmethod
    def check_solution(arr):
        """
        Returns True if Sudoku is solved and valid by asserting each number only occurs once per row, column or
        field.

        :return: True if valid, else False
        """
        for i in range(9):
            set_row = set(arr[i, :]).difference({0})
            set_column = set(arr[:, i]).difference({0})
            if len(set_row) is not 9 or len(
                    set_column) is not 9:
                return False
        for i in range(0, 7, 3):
            for j in range(0, 7, 3):
                if len(set(np.reshape(arr[i:i + 3, j:j + 3], 9)).difference({0})) is not 9:
                    return False
        return True


class SudokuGenerator:
    def __init__(self, n: int, font_path: Union[Path, str] = 'fonts/FreeMono.ttf',
                 workers=1):
        self.font = ImageFont.truetype(font_path, 32)
        self.generated: List[Sudoku] = []
        self.n = n
        self.workers = min(workers, n)
        self._generate()

    def _generate(self):
        """
        Main sudoku generation method. Spawns self.workers processes each of which creates the same number of sudokus.
        If the number of sudokus to be generated is not a multiple of the number of desired workers, an additional
        process is spawned to produce the remaining number of sudokus.
        """
        nums = np.arange(0, self.n, dtype=np.int)
        out_q = mp.Queue()
        procs = []
        chunksize = self.n // self.workers
        for i in range(self.workers):
            p = mp.Process(
                target=SudokuGenerator.get_sudokus,
                args=(nums[chunksize * i:chunksize * (i + 1)],
                      out_q))
            procs.append(p)
            p.start()

        if self.n % self.workers is not 0:
            p = mp.Process(
                target=SudokuGenerator.get_sudokus,
                args=(nums[chunksize * self.workers:],
                      out_q))
            procs.append(p)
            p.start()

        print(f"Started {len(procs)} workers")
        for _ in range(len(procs)):
            larr = out_q.get()
            self.generated.extend(larr)

        for p in procs:
            p.join()
        print("Finished generation")

    @staticmethod
    def get_sudokus(indices, out_q):
        larr: List[Sudoku] = []
        for i, seed in enumerate(indices):
            np.random.seed(seed)
            while True:
                sudoku = Sudoku()
                if sudoku.is_valid:  # Ensure generated sudoku is valid, otherwise run again
                    larr.append(sudoku)
                    break
        out_q.put(larr)

    @staticmethod
    def get_sudoku_grid(cell_size=28, major_line_width=2, minor_line_width=1):
        grid_size = cell_size * 9 + 4 * major_line_width + 6 * minor_line_width

        data = np.ndarray((grid_size, grid_size, 4), dtype=np.uint8)
        data.fill(255)
        coords = np.zeros(10, dtype=np.int)
        idx = 0
        for i in range(10):
            if i % 3 == 0:  # draw major line
                data[idx:idx + major_line_width, :, 0:3] = 0
                data[:, idx:idx + major_line_width, 0:3] = 0
                coords[i] = idx + major_line_width
                idx += major_line_width + cell_size
            else:  # draw minor line
                data[idx:idx + minor_line_width, :, 0:3] = 0
                data[:, idx:idx + minor_line_width, 0:3] = 0
                coords[i] = idx + minor_line_width
                idx += minor_line_width + cell_size

        return data, coords

    def draw_sudoku_pil(self, idx: int, masking_rate=0.7):
        sudoku = self.generated[idx].data

        mask = np.random.choice([True, False], size=sudoku.shape, p=[masking_rate, 1 - masking_rate])

        grid_data, coords = self.get_sudoku_grid()

        grid_image = Image.fromarray(grid_data, 'RGBA')
        txt_image = Image.new('RGBA', grid_image.size, WHITE_NO_ALPHA)
        txt_draw = ImageDraw.Draw(txt_image)

        x_offset = 4
        y_offset = -2
        for i in range(9):
            for j in range(9):
                digit = sudoku[i][j]
                x_coord, y_coord = coords[i], coords[j]
                if digit == 0:
                    continue
                elif not mask[i][j]:
                    txt_draw.text((x_offset + x_coord, y_offset + y_coord), str(digit), font=self.font, fill=BLACK)
        out = Image.alpha_composite(grid_image, txt_image)
        return out

    def draw_sudoku_pil_mnist(self, idx: int, masking_rate=0.7, mnist: MNIST = None, mnist_rate=0.2):
        np.random.seed(idx)  # FIXME
        sudoku = self.generated[idx]

        # Mask setup
        mask = np.random.choice([False, True], size=sudoku.shape, p=[masking_rate, 1 - masking_rate])
        mnist_mask = np.random.choice([True, False], size=sudoku.shape, p=[mnist_rate, 1 - mnist_rate])
        mnist_mask = np.logical_xor(mnist_mask, mask)

        # Image data setup
        grid_data, coords = self.get_sudoku_grid()
        mnist_data = np.zeros(grid_data.shape, dtype=np.uint8)

        # Image setup
        grid_image = Image.fromarray(grid_data, 'RGBA')
        background_image = Image.new('RGBA', grid_image.size, WHITE)
        mnist_image = Image.fromarray(mnist_data, 'RGBA')
        txt_image = Image.new('RGBA', grid_image.size, WHITE_NO_ALPHA)
        txt_draw = ImageDraw.Draw(txt_image)

        # Drawing
        x_offset = -2
        y_offset = 4
        for i in range(9):
            for j in range(9):
                digit = sudoku[i][j]
                x_coord, y_coord = coords[i], coords[j]
                if digit == 0:
                    continue
                if mask[i][j]:
                    txt_draw.text((y_offset + y_coord, x_offset + x_coord), str(digit), font=self.font, fill=BLACK)
                elif mnist_mask[i][j]:
                    mnist_digit = np.array(mnist.get_random(digit), dtype=np.int).repeat(4).reshape((28, 28, 4))
                    mnist_digit = mnist_digit * -1 + 255
                    mnist_digit[:, :, 3] = 255
                    mnist_data[x_coord:x_coord + 28, y_coord:y_coord + 28] = mnist_digit

        # Image composition
        out = Image.alpha_composite(background_image, grid_image)
        out = Image.alpha_composite(out, txt_image)
        out = Image.alpha_composite(out, mnist_image)
        return out
