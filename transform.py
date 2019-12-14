from typing import List

import numpy as np

from sudoku_generator import Sudoku, SudokuGenerator


class Transform:
    def apply(self, sudoku: Sudoku) -> List[Sudoku]:
        pass


class Rotation(Transform):
    def apply(self, sudoku: Sudoku) -> List[Sudoku]:
        ret: List[Sudoku] = [sudoku]
        for i in range(3):
            sudoku_from_array = Sudoku.from_array(np.rot90(ret[i].data))
            if sudoku_from_array.is_valid:
                ret.append(sudoku_from_array)
        return ret


class Flip(Transform):
    def apply(self, sudoku: Sudoku) -> List[Sudoku]:
        ret: List[Sudoku] = [sudoku]

        for ax in [0, 1, None]:
            sudoku_from_array = Sudoku.from_array(np.flip(sudoku.data, ax))
            if sudoku_from_array.is_valid:
                ret.append(sudoku_from_array)
        return ret


class TransformSudokuGenerator(SudokuGenerator):
    def __init__(self, n: int, **kwargs):
        super().__init__(n, **kwargs)
        self.transforms: List[Transform] = []

    def add_transform(self, transform: Transform):
        self.transforms.append(transform)

    def build(self):
        for transform in self.transforms:
            newl: List[Sudoku] = []
            for s in self.generated:
                newl.extend(transform.apply(s))
            self.generated = newl
