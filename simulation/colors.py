from enum import Enum
from typing import Union

import numpy as np


class Color(Enum):
    NONE = (0, 0, 0, 0)
    BLACK = (0, 0, 0, 255)
    BLACK_NO_ALPHA = (0, 0, 0, 0)
    WHITE = (255, 255, 255, 255)
    WHITE_NO_ALPHA = (255, 255, 255, 0)


def uint8_from_number(number: Union[int, float]):
    """
    Converts a given number to a uint8. If the given number is float, it will be interpreted as a normalized float
    [0.1, 1.0] and as such scaled up and clipped. Otherwise, it will only be clipped to [0, 255].

    Args:
        number(Union[int, float]): The number to convert.

    Returns:
        int: The number as uint8.

    """
    if type(number) is float:
        return np.clip(number * 255, 0, 255).astype(np.uint8)
    else:
        return np.clip(number, 0, 255).astype(np.uint8)
