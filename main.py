import cv2
import numpy as np
from PIL import Image

from digit.digit_dataset import Chars74KI
from sudoku.render.sudoku_renderer import SudokuRenderer
from sudoku.sudoku_permutation import PermutationSudokuGenerator, MajorSwitch


def save_img(img: Image, name: str):
    cv2.imwrite(name, img)


def temp_row_transform():
    sgen = PermutationSudokuGenerator(1, workers=8)
    sgen.permutations.append(MajorSwitch(row_switch=(1, 0, 2)))
    sgen.permutations.append(MajorSwitch(column_switch=(1, 0, 2)))
    sgen.permutations.append(MajorSwitch(row_switch=(1, 0, 2), column_switch=(1, 0, 2)))
    sgen.build()
    print(sgen.generated[0].data)
    save_img(SudokuRenderer.draw_sudoku_pil(0, masking_rate=0.0), f"sudoku_0.0_switchM([0,1,2],[0,1,2]).png")

    print(sgen.generated[1].data)
    print(f"Is valid: {sgen.generated[1].is_valid:b}")
    save_img(SudokuRenderer.draw_sudoku_pil(1, masking_rate=0.0), f"sudoku_0.0_switchM([0,1,2],[1,0,2]).png")

    print(sgen.generated[2].data)
    print(f"Is valid: {sgen.generated[2].is_valid:b}")
    save_img(SudokuRenderer.draw_sudoku_pil(2, masking_rate=0.0), f"sudoku_0.0_switchM([1,0,2],[0,1,2]).png")

    print(sgen.generated[3].data)
    print(f"Is valid: {sgen.generated[3].is_valid:b}")
    save_img(SudokuRenderer.draw_sudoku_pil(3, masking_rate=0.0), f"sudoku_0.0_switchM([1,0,2],[1,0,2]).png")


if __name__ == '__main__':
    # temp_row_transform()

    mnist = Chars74KI()
    sgen = PermutationSudokuGenerator(1, workers=8)
    sgen.permutations.append(MajorSwitch(row_switch=(1, 0, 2)))
    sgen.permutations.append(MajorSwitch(column_switch=(1, 0, 2)))
    sgen.permutations.append(MajorSwitch(row_switch=(1, 0, 2), column_switch=(1, 0, 2)))
    sgen.build()
    print(sgen.generated[2].data)
    for p in [0., .2, .4, .6, .8, 1.0]:
        np.random.seed(2)
        save_img(SudokuRenderer.draw_sudoku_pil_mnist(sgen.generated[2], mnist=mnist, mnist_rate=p),
                 f"sudoku_0.7_{p}.png")
