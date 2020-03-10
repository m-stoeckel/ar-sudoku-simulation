from unittest import TestCase

import matplotlib.pyplot as plt
from tqdm import tqdm

from digit.character_provider import SingleFontCharacterRenderer
from digit.fonts import Font


class TestCharacterProvider(TestCase):

    def test_all_fonts(self):
        t = tqdm(Font, position=0)
        for font in t:
            t.set_description_str(f"Rendering {font.name}")
            test_grid(font)


def test_grid(font):
    cp = SingleFontCharacterRenderer(font=font)
    x_base = 10
    y_base = 7
    plt.subplots(x_base, y_base, figsize=(y_base + 0.5, x_base + 1.5))
    plt.tight_layout()
    for i in range(x_base * y_base):
        ax = plt.subplot2grid((x_base, y_base), (i % x_base, i // x_base))
        ax.axis('off')
        ax.imshow(cp[i], cmap='gray', interpolation='lanczos')
    plt.suptitle(font.name, y=0.99)
    plt.show()

    # def test(self):
    #     cp = CharacterProvider()
    #     for i in [0, 4, "A", "R"]:
    #         Image.fromarray(cp[i]).show()
