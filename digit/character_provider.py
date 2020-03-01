from enum import Enum
from typing import Union, Tuple

import numpy as np
from PIL import ImageFont, ImageDraw, Image
from tqdm import trange


class Font(Enum):
    FREE_MONO = "../fonts/FreeMono.ttf"
    FREE_MONO_BOLD = "../fonts/FreeMonoBold.ttf"
    FREE_MONO_BOLD_OBLIQUE = "../fonts/FreeMonoBoldOblique.ttf"
    FREE_MONO_OBLIQUE = "../fonts/FreeMonoOblique.ttf"


class CharacterProvider:

    def __init__(self, base_resolution: Union[int, Tuple[int, int]] = 128, font=Font.FREE_MONO):
        if isinstance(base_resolution, int):
            self.base_resolution = (base_resolution, base_resolution)
        else:
            assert base_resolution[0] == base_resolution[1]
            self.base_resolution = base_resolution
        self.characters = {}
        self.lookup = []
        font = ImageFont.truetype(font.value, 88)
        for rng in ((48, 58), (65, 91), (97, 123)):
            for i in trange(rng[0], rng[1]):
                char = str(chr(i))
                char_img = np.full(self.base_resolution, 255, dtype=np.uint8)
                char_img = Image.fromarray(char_img)
                draw = ImageDraw.Draw(char_img)
                draw.text((int(0.25 * self.base_resolution[0]), int(0.05 * self.base_resolution[0])), char, font=font)
                char_img = np.array(char_img)

                self.lookup.append(char)
                self.characters[char] = char_img
        for char in ["ä", "ü", "ö", "Ä", "Ü", "Ö", "ß"]:
            char_img = np.full(self.base_resolution, 255, dtype=np.uint8)
            char_img = Image.fromarray(char_img)
            draw = ImageDraw.Draw(char_img)
            draw.text((int(0.25 * self.base_resolution[0]), int(0.05 * self.base_resolution[0])), char, font=font)
            char_img = np.array(char_img)

            self.lookup.append(char)
            self.characters[char] = char_img

    def __getitem__(self, item):
        if type(item) is int:
            if item >= len(self.lookup):
                return np.array(Image.new("RGBA", self.base_resolution))
            return self.characters[self.lookup[item]]
        else:
            return self.characters[item]

    def get(self, item, resolution=100):
        return ...
