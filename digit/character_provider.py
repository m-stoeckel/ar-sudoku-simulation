from pathlib import Path
from typing import Union, Tuple

import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from p_tqdm import p_map

from digit.fonts import Font
from sudoku import Color


class CharacterRenderer:
    char_list = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZÄÜÖabcdefghijklmnopqrstuvwxyzäüöß,.-!(){}[]")

    def __init__(self, render_resolution: Union[int, Tuple[int, int]] = 128):
        self.render_resolution = get_resolution(render_resolution)

    def render_character(self, char, font):
        char_img = Image.new('RGBA', self.render_resolution)
        draw = ImageDraw.Draw(char_img)
        w, h = draw.textsize(char, font=font)
        draw.text((int((self.render_resolution[0] - w) / 2), int(self.render_resolution[1] - h) / 2), char,
                  font=font, fill=Color.BLACK.value, stroke_fill=Color.BLACK.value)
        char_img = np.array(char_img)
        return char_img

    def prerender_all(self):
        character_dir = Path("../datasets/characters/")
        character_dir.mkdir(exist_ok=True)

        def _prerender_font(font):
            output_dir = character_dir / font.name
            output_dir.mkdir(exist_ok=True)

            # Choose fontsize from resolution
            font_size_pt = int(self.render_resolution[1] / 1.3)
            font = ImageFont.truetype(font.value, font_size_pt)
            for char in self.char_list:
                char_img = self.render_character(char, font)
                cv2.imwrite(str(output_dir / f"{ord(char)}.png"), char_img)

        p_map(_prerender_font, list(Font), desc="Rendering fonts")


class SingleFontCharacterRenderer(CharacterRenderer):

    def __init__(self, render_resolution: Union[int, Tuple[int, int]] = 128, font=Font.FREE_MONO):
        super().__init__(render_resolution)
        self.characters = {}
        self.lookup = []

        # choose font size as 90% of the entire image area
        font_size_pt = int(self.render_resolution[1] / 1.3)
        font = ImageFont.truetype(font.value, font_size_pt)
        for char in self.char_list:
            char_img = self.render_character(char, font)
            self.lookup.append(char)
            self.characters[char] = char_img

    def __getitem__(self, item):
        if type(item) is int:
            if item >= len(self.lookup):
                return np.array(Image.new("RGBA", self.render_resolution))
            return self.characters[self.lookup[item]]
        else:
            return self.characters[item]

    def get(self, item, resolution=128):
        resolution = get_resolution(resolution)
        if resolution == self.render_resolution:
            return self[item]
        else:
            digit = self[item]
            if resolution[0] < self.render_resolution[0] or resolution[1] < self.render_resolution[1]:
                return cv2.resize(digit, (resolution[0], resolution[1], 4), interpolation=cv2.INTER_LANCZOS4)
            else:
                return cv2.resize(digit, (resolution[0], resolution[1], 4), interpolation=cv2.INTER_CUBIC)


def get_resolution(render_resolution):
    if isinstance(render_resolution, int):
        return render_resolution, render_resolution
    else:
        assert render_resolution[0] == render_resolution[1]
        return render_resolution


if __name__ == '__main__':
    CharacterRenderer().prerender_all()
