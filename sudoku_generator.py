import multiprocessing as mp
import os
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.datasets import fetch_openml


class MNIST:
    def __init__(self):
        print("Loading MNIST dataset")
        # Load data from https://www.openml.org/d/554
        os.makedirs('datasets/', exist_ok=True)
        X, Y = fetch_openml('mnist_784', version=1, return_X_y=True, data_home="datasets/", cache=True)
        self.x = np.array(X).reshape((70000, 28, 28))
        self.y = np.array(Y)
        self.indices_by_number = [np.flatnonzero(self.y == str(i)) for i in range(0, 10)]
        print(self.indices_by_number)
        del X, Y

    def get_random(self, digit: int) -> np.array:
        return self.x[np.random.choice(self.indices_by_number[digit])]


class Sudoku:
    def __init__(self):
        while True:
            n = 9
            self.solution = np.zeros((n, n), dtype=np.int)
            rg = np.arange(1, n + 1)
            self.solution[0, :] = np.random.choice(rg, n, replace=False)
            try:
                for r in range(1, n):
                    for c in range(n):
                        col_rest = np.setdiff1d(rg, self.solution[:r, c])
                        row_rest = np.setdiff1d(rg, self.solution[r, :c])
                        avb1 = np.intersect1d(col_rest, row_rest)
                        sub_r, sub_c = r // 3, c // 3
                        avb2 = np.setdiff1d(np.arange(0, n + 1),
                                            self.solution[sub_r * 3:(sub_r + 1) * 3, sub_c * 3:(sub_c + 1) * 3].ravel())
                        avb = np.intersect1d(avb1, avb2)
                        self.solution[r, c] = np.random.choice(avb, size=1)
                break
            except ValueError:
                pass

    @property
    def shape(self):
        return self.solution.shape

    def __getitem__(self, item):
        return self.solution[item]

    def solve(self, m):
        if isinstance(m, list):
            m = np.array(m)
        elif isinstance(m, str):
            m = np.loadtxt(m, dtype=np.int, delimiter=",")
        rg = np.arange(m.shape[0] + 1)
        while True:
            mt = m.copy()
            while True:
                d = []
                d_len = []
                for i in range(m.shape[0]):
                    for j in range(m.shape[1]):
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
        return Sudoku.check_solution(self.solution)

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
        self.font_path = font_path
        self.font_path_bold = 'fonts/FreeMonoBold.ttf'
        self.generated = np.zeros((n, 9, 9), dtype=np.int)
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
            indices, larr = out_q.get()
            self.generated[indices] = larr

        for p in procs:
            p.join()
        print("Finished generation")

    @staticmethod
    def get_sudokus(indices, out_q):
        larr = np.zeros((indices.shape[0], 9, 9), dtype=np.int)
        for i, seed in enumerate(indices):
            np.random.seed(seed)
            while True:
                sudoku = Sudoku()
                if sudoku.is_valid:  # Ensure generated sudoku is valid, otherwise run again
                    larr[i] = sudoku.solution
                    break
        out_q.put((indices, larr))

    def get_sudoku_grid(self, cell_size=28, major_line_width=2, minor_line_width=1):
        grid_size = cell_size * 9 + 4 * major_line_width + 6 * minor_line_width
        major_sep = major_line_width + 3 * cell_size + 2 * minor_line_width

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
        sudoku = self.generated[idx]
        data, coords = self.get_sudoku_grid()
        mask = np.random.choice([True, False], size=sudoku.shape, p=[masking_rate, 1 - masking_rate])

        image = Image.fromarray(data, 'RGBA')

        fnt = ImageFont.truetype(self.font_path, 32)
        txt = Image.new('RGBA', image.size, (255, 255, 255, 0))
        d = ImageDraw.Draw(txt)
        x_offset = 4
        y_offset = -2
        for i in range(9):
            for j in range(9):
                digit = sudoku[i][j]
                x_coord, y_coord = coords[i], coords[j]
                if digit == 0:
                    continue
                elif not mask[i][j]:
                    d.text((x_offset + x_coord, y_offset + y_coord), str(digit), font=fnt, fill=(0, 0, 0, 255))
        out = Image.alpha_composite(image, txt)
        return out

    def draw_sudoku_pil_mnist(self, idx: int, masking_rate=0.7, mnist: MNIST = None, mnist_rate=0.2):
        sudoku = self.generated[idx]
        data, coords = self.get_sudoku_grid()

        np.random.seed(idx)  # FIXME
        mask = np.random.choice([False, True], size=sudoku.shape, p=[masking_rate, 1 - masking_rate])
        mnist_mask = np.random.choice([True, False], size=sudoku.shape, p=[mnist_rate, 1 - mnist_rate])
        mnist_mask = np.logical_xor(mnist_mask, mask)

        image = Image.fromarray(data, 'RGBA')
        fnt = ImageFont.truetype(self.font_path, 32)
        txt = Image.new('RGBA', image.size, (255, 255, 255, 0))
        d = ImageDraw.Draw(txt)

        x_offset = -2
        y_offset = 4
        for i in range(9):
            for j in range(9):
                digit = sudoku[i][j]
                x_coord, y_coord = coords[i], coords[j]
                if digit == 0:
                    continue
                if mask[i][j]:
                    d.text((y_offset + y_coord, x_offset + x_coord), str(digit), font=fnt, fill=(0, 0, 0, 255))
                elif mnist_mask[i][j]:
                    # FIXME
                    mnist_digit = np.array(mnist.get_random(digit), dtype=np.int).repeat(4).reshape((28, 28, 4))
                    mnist_digit = mnist_digit * -1 + 255
                    mnist_digit[:, :, 3] = 255
                    data[x_coord:x_coord + 28, y_coord:y_coord + 28] = mnist_digit
        out = Image.alpha_composite(image, txt)
        return out


def save_img(img: Image, name: str):
    with open(name, 'wb') as fout:
        img.save(fout)


if __name__ == '__main__':
    # mnist = None
    mnist = MNIST()
    sgen = SudokuGenerator(1, workers=8)
    for i in range(1):
        print(sgen.generated[i])
        # valid = Sudoku.check_solution(sgen.generated[i])
        # nt = 'not ' if not valid else ''
        # print(f"Sudoku is {nt}valid")
    # sgen.draw_sudoku_pil(0)
    for p in [0., .2, .4, .6, .8, 1.0]:
        save_img(sgen.draw_sudoku_pil_mnist(0, mnist=mnist, mnist_rate=p), f"sudoku_0.7_{p}.png")
