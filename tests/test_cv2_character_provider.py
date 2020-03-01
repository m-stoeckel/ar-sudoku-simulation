from unittest import TestCase

import matplotlib.pyplot as plt
from tqdm import trange

from digit.character_provider import CharacterProvider


class TestCharacterProvider(TestCase):

    def test_grid(self):
        cp = CharacterProvider()
        base = 10
        plt.subplots(base, base, figsize=(base + 0.5, base + 0.5))
        plt.tight_layout()
        for i in trange(base * base):
            ax = plt.subplot2grid((base, base), (i % base, i // base))
            ax.axis('off')
            ax.imshow(cp[i], cmap='gray', interpolation='lanczos')
        plt.show()

    # def test(self):
    #     cp = CharacterProvider()
    #     for i in [0, 4, "A", "R"]:
    #         Image.fromarray(cp[i]).show()
