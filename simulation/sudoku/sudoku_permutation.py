from abc import ABCMeta, abstractmethod
from typing import List

import numpy as np

from simulation.sudoku import Sudoku, SudokuGenerator


class SudokuPermutation(metaclass=ABCMeta):
    """
    Abstract base class for all Sudoku permutations.
    """

    @abstractmethod
    def apply(self, sudoku: Sudoku) -> List[Sudoku]:
        pass


class Rotation(SudokuPermutation):
    """
    Rotates the sudoku by 90, 180 and 270 degrees.
    """

    def apply(self, sudoku: Sudoku) -> List[Sudoku]:
        """

        Args:
            sudoku: The sudoku to be rotated.

        Returns:
            The sudoku in all three possible rotations.
        """
        ret: List[Sudoku] = [np.rot90(sudoku.data)]
        for i in range(2):
            sudoku_from_array = Sudoku.from_array(np.rot90(ret[i].data))
            if not sudoku_from_array.is_valid:
                print(sudoku_from_array.data)
                raise ValueError
            ret.append(sudoku_from_array)
        return ret


class Flip(SudokuPermutation):
    """
    Flips the Sudoku along the x and y axis.
    """

    def apply(self, sudoku: Sudoku) -> List[Sudoku]:
        """

        Args:
            sudoku: The Sudoku to be flipped.

        Returns:


        """
        ret: List[Sudoku] = [sudoku]

        for ax in [0, 1, None]:
            sudoku_from_array = Sudoku.from_array(np.flip(sudoku.data, ax))
            if not sudoku_from_array.is_valid:
                print(sudoku_from_array.data)
                raise ValueError
            ret.append(sudoku_from_array)
            return ret


class MajorSwitch(SudokuPermutation):
    """
    Performs a major row and/or column switch in the Sudoku.
    """

    def __init__(self, column_switch=(0, 1, 2), row_switch=(0, 1, 2), column_first=False):
        """

        Args:
            column_switch(Tuple[int, int, int]): Column order after switching. (Default value = (0, 1, 2))
            row_switch(Tuple[int, int, int]): Row order after switching. (Default value = (0, 1, 2))
            column_first(bool): If True apply the column switch first. Otherwise rows are switched first.
                (Default value = False)
        """
        lookup = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        if not set(column_switch) == set(row_switch) == {0, 1, 2}:
            raise ValueError()

        self.column_order = [n for cid in column_switch for n in lookup[cid]]
        self.row_order = [n for cid in row_switch for n in lookup[cid]]
        self.column_first = column_first

    def apply(self, sudoku: Sudoku) -> List[Sudoku]:
        sudoku_from_array = Sudoku.from_array(sudoku.data.copy())
        if self.column_first:
            sudoku_from_array.data[:, self.column_order] = sudoku.data[:, [0, 1, 2, 3, 4, 5, 6, 7, 8]]
            sudoku_from_array.data[self.row_order, :] = sudoku_from_array.data[[0, 1, 2, 3, 4, 5, 6, 7, 8], :]
        else:
            sudoku_from_array.data[self.row_order, :] = sudoku.data[[0, 1, 2, 3, 4, 5, 6, 7, 8], :]
            sudoku_from_array.data[:, self.column_order] = sudoku_from_array.data[:, [0, 1, 2, 3, 4, 5, 6, 7, 8]]

        if not sudoku_from_array.is_valid:
            print(sudoku_from_array.data)
            raise ValueError
        return [sudoku_from_array]


class PermutationSudokuGenerator(SudokuGenerator):
    """
    Superclass of :py:class:`SudokuGenerator <simulation.sudoku.sudoku_generator.SudokuGenerator>` that applies
    permuations to all generated Sudokus.
    """

    def __init__(self, n: int, **kwargs):
        super().__init__(n, **kwargs)
        self.permutations: List[SudokuPermutation] = []

    def add_permutation(self, permutation: SudokuPermutation):
        """
        Add a permutation.

        Args:
            permutation(SudokuPermutation): The permutation to add.

        Returns:
            None

        """
        self.permutations.append(permutation)

    def apply(self):
        """
        Apply all permutations for each generated Sudoku.

        Returns:
            None
        """
        generated = self.generated.copy()
        for permutation in self.permutations:
            newl: List[Sudoku] = []
            for s in generated:
                newl.extend(permutation.apply(s))
            self.generated.extend(newl)
