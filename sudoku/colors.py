from enum import Enum


class Color(Enum):
    BLACK = (0, 0, 0, 255)
    BLACK_NO_ALPHA = (0, 0, 0, 0)
    WHITE = (255, 255, 255, 255)
    WHITE_NO_ALPHA = (255, 255, 255, 0)
