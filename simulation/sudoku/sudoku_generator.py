import multiprocessing as mp
from pathlib import Path
from typing import Union, List

import numpy as np

DEBUG = False


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
    def __init__(self, n: int, workers=1):
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
