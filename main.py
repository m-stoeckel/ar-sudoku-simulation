from PIL import Image

from sudoku_generator import SudokuGenerator, Sudoku
from transform import TransformSudokuGenerator, Rotation


def save_img(img: Image, name: str):
    with open(name, 'wb') as fout:
        img.save(fout)


def temp_row_transform():
    sgen = SudokuGenerator(1, workers=8)
    print(sgen.generated[0].data.copy())
    save_img(sgen.draw_sudoku_pil(0, masking_rate=0.0), f"sudoku_0.0_switchM([0,1,2],[0,1,2]).png")
    sudoku = Sudoku.from_array(sgen.generated[0].data.copy())
    sudoku.data[:, [3, 4, 5, 0, 1, 2, 6, 7, 8]] = sgen.generated[0].data[:, [0, 1, 2, 3, 4, 5, 6, 7, 8]]
    sgen.generated.append(sudoku)
    save_img(sgen.draw_sudoku_pil(1, masking_rate=0.0), f"sudoku_0.0_switchM([0,1,2],[1,0,2]).png")
    print(sudoku.data)
    print(f"Is valid: {sudoku.is_valid:b}")
    sudoku = Sudoku.from_array(sgen.generated[0].data.copy())
    sudoku.data[[3, 4, 5, 0, 1, 2, 6, 7, 8]] = sgen.generated[1].data[[0, 1, 2, 3, 4, 5, 6, 7, 8]]
    sgen.generated.append(sudoku)
    save_img(sgen.draw_sudoku_pil(2, masking_rate=0.0), f"sudoku_0.0_switchM([1,0,2],[0,1,2]).png")
    print(sudoku.data)
    print(f"Is valid: {sudoku.is_valid:b}")


if __name__ == '__main__':
    # mnist = MNIST()
    sgen = TransformSudokuGenerator(1, workers=8)
    sgen.add_transform(Rotation())
    sgen.build()
    print(sgen.generated[2].data)

    # temp_row_transform()
    # sgen.draw_sudoku_pil(0)
    # for p in [0., .2, .4, .6, .8, 1.0]:
    #     save_img(sgen.draw_sudoku_pil_mnist(0, mnist=mnist, mnist_rate=p), f"sudoku_0.7_{p}.png")
