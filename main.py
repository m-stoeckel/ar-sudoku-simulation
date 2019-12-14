from PIL import Image

from transform import TransformSudokuGenerator, MajorSwitch


def save_img(img: Image, name: str):
    with open(name, 'wb') as fout:
        img.save(fout)


def temp_row_transform():
    sgen = TransformSudokuGenerator(1, workers=8)
    sgen.transforms.append(MajorSwitch(row_switch=(1, 0, 2)))
    sgen.transforms.append(MajorSwitch(column_switch=(1, 0, 2)))
    sgen.transforms.append(MajorSwitch(row_switch=(1, 0, 2), column_switch=(1, 0, 2)))
    sgen.build()
    print(sgen.generated[0].data)
    save_img(sgen.draw_sudoku_pil(0, masking_rate=0.0), f"sudoku_0.0_switchM([0,1,2],[0,1,2]).png")

    print(sgen.generated[1].data)
    print(f"Is valid: {sgen.generated[1].is_valid:b}")
    save_img(sgen.draw_sudoku_pil(1, masking_rate=0.0), f"sudoku_0.0_switchM([0,1,2],[1,0,2]).png")

    print(sgen.generated[2].data)
    print(f"Is valid: {sgen.generated[2].is_valid:b}")
    save_img(sgen.draw_sudoku_pil(2, masking_rate=0.0), f"sudoku_0.0_switchM([1,0,2],[0,1,2]).png")

    print(sgen.generated[3].data)
    print(f"Is valid: {sgen.generated[3].is_valid:b}")
    save_img(sgen.draw_sudoku_pil(3, masking_rate=0.0), f"sudoku_0.0_switchM([1,0,2],[1,0,2]).png")


if __name__ == '__main__':
    # mnist = MNIST()
    # sgen = TransformSudokuGenerator(1, workers=8)
    # sgen.add_transform(Rotation())
    # sgen.build()
    # print(sgen.generated[2].data)

    temp_row_transform()
    # sgen.draw_sudoku_pil(0)
    # for p in [0., .2, .4, .6, .8, 1.0]:
    #     save_img(sgen.draw_sudoku_pil_mnist(0, mnist=mnist, mnist_rate=p), f"sudoku_0.7_{p}.png")
