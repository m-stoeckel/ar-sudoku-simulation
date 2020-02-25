from enum import Enum

import numpy as np


class Color(Enum):
    NONE = (0, 0, 0, 0)
    BLACK = (0, 0, 0, 255)
    BLACK_NO_ALPHA = (0, 0, 0, 0)
    WHITE = (255, 255, 255, 255)
    WHITE_NO_ALPHA = (255, 255, 255, 0)


def uint8_from_number(number):
    if type(number) is float:
        return np.clip(number * 255, 0, 255).astype(np.uint8)
    else:
        return np.clip(number, 0, 255).astype(np.uint8)