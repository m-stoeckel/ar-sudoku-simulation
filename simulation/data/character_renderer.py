"""
A module for the rendering of characters using TrueType fonts.

.. codeauthor:: Manuel Stoeckel <manuel.stoeckel@stud.uni-frankfurt.de>
"""

import os
from pathlib import Path
from typing import Union, Tuple

import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from p_tqdm import p_map

from simulation import Color
from simulation.data.fonts import Font


class CharacterRenderer:
    """
    A simple class to render TrueType fonts.
    """
    #: The list of characters to render by default. Includes all numbers, common english letters and some punctuation
    #: marks and brackets.
    char_list = list(u"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,.-!?(){}[]")

    def __init__(self, render_resolution: Union[int, Tuple[int, int]] = 128):
        """
        :param render_resolution: The width and height of the output images. Default: 128.
        :type render_resolution: Union[int, Tuple[int, int]]
        """
        self.render_resolution = get_resolution(render_resolution)

    def render_character(self, char, font, mode='RGBA') -> np.ndarray:
        """
        Render a single character with a given font.

        :param char: The character to render.
        :type char: str
        :param font: The TrueType font path.
        :type font: str
        :param mode: The Pillow image mode to use for the output image. Default: 'RGBA'.
        :type mode: str
        :return: The rendered image as a numpy array.
        :rtype: numpy.ndarray
        """
        if mode == 'RGBA':
            char_img = Image.new(mode, self.render_resolution)
            draw_color = Color.BLACK.value
        else:
            char_img = Image.new(mode, self.render_resolution, color=0)
            draw_color = 255
        draw = ImageDraw.Draw(char_img)
        w, h = draw.textsize(char, font=font)
        draw.text((int((self.render_resolution[0] - w) / 2), int(self.render_resolution[1] - h) / 2), char,
                  font=font, fill=draw_color, stroke_fill=draw_color)
        char_img = np.array(char_img)
        return char_img

    def prerender_all(self, base_dir=Path("."), mode='RGBA'):
        """
        Render all characters of the *char_list* and save the images with the given mode.

        :param base_dir: The base path of the project. The images will be saved in the directory
            '{base_path}/datasets/characters/' in a separate folder for each character.
        :type base_dir: Path
        :param mode: The Pillow image mode to use for the output image. Default: 'RGBA'.
        :type mode: str
        :return: None
        """
        character_dir = base_dir / "datasets/characters/"
        character_dir.mkdir(exist_ok=True)
        font_list = list(Font)

        def _prerender_font(font: Font):
            """
            Helper function for pre-rendering fonts in parallel.

            :param font: The font to pre-render.
            :type font: Font
            :return: None
            """
            # Choose fontsize from resolution
            # Note: The divisor should be 1.3 for unit correctness,
            # but 1.2 works better for the given task
            font_size_pt = int(self.render_resolution[1] / 1.2)

            ttfont = ImageFont.truetype(str(base_dir / font.value), font_size_pt)
            for char in self.char_list:
                char_img = self.render_character(char, ttfont, mode)
                output_dir = character_dir / str(ord(char))
                output_dir.mkdir(exist_ok=True)
                cv2.imwrite(str(output_dir / f"{font_list.index(font)}.png"), char_img)

        p_map(_prerender_font, font_list, desc="Rendering fonts", num_cpus=os.cpu_count())


class SingleFontCharacterRenderer(CharacterRenderer):
    """
    A CharacterRenderer that renders characters in a single font and saves them in a list. By default all characters in
    :py:attr:`char_list` are rendered upon construction. Other characters are rendered in a just-in-time manor, but saved for
    later.
    """

    def __init__(self, render_resolution: Union[int, Tuple[int, int]] = 128, font=Font.FREE_MONO):
        """
        :param render_resolution: The width and height of the output images. Default: 128.
        :type render_resolution: Union[int, Tuple[int, int]]
        :param font: The font
        :type font: :class:`<simulation.data.fonts.Font>`
        """
        super().__init__(render_resolution)
        self.characters = {}
        self.lookup = []

        # choose font size as 90% of the entire transforms area
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


def get_resolution(render_resolution: Union[int, Tuple[int, int]]):
    """
    Helper function which returns a tuple of two identical values.
    If the input is an integer, returns a new tuple.
    If the input is a tuple, the equality of both first elements is asserted and the tuple returned unchanged.

    :param render_resolution: The resolution.
    :type render_resolution: Union[int, Tuple[int, int]]
    :return: A tuple of two ints.
    :rtype: Tuple[int, int]
    """
    if isinstance(render_resolution, int):
        return render_resolution, render_resolution
    else:
        assert render_resolution[0] == render_resolution[1]
        return render_resolution


if __name__ == '__main__':
    CharacterRenderer().prerender_all(Path("../../"), mode='L')
