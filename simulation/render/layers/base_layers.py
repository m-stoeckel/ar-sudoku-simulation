from typing import List, Tuple, Union

import cv2
import numpy as np

from simulation.colors import uint8_from_number
from simulation.render.digital_composition import AlphaComposition, GammaCorrectedAlphaComposition


class Layer:
    def __init__(self, shape: tuple, backside=False):
        self.backside = backside
        self.shape = shape

    def compose(self):
        pass


class DigitalCompositionLayer(Layer):
    def __init__(self, shape: tuple, composite=AlphaComposition(), **kwargs):
        super().__init__(shape, **kwargs)
        self.composite = composite
        self.elements: List[np.ndarray] = []

    def add_element(self, element: np.ndarray) -> None:
        """
        Add an element to the printing layer, with same resolution as all other layers.

        :param element: A numpy.ndarray of the same shape as existing elements. If element is grayscale or RGB, it will
        be converted to RGBA.
        :raises: ValueError if the shapes of the elements do not match.
        :return: None
        """
        if len(self.elements) > 0 and self.elements[0].shape[:2] != element.shape[:2]:
            raise ValueError(f"The shape {element.shape[:2]} "
                             f"does not match the layer shape {self.elements[0].shape[:2]}")
        if element.shape[2] < 4:
            if element.shape[2] == 1:
                element = cv2.cvtColor(element, cv2.COLOR_GRAY2RGBA)
            elif element.shape[2] == 3:
                element = cv2.cvtColor(element, cv2.COLOR_RGB2RGBA)
            element[:, :, 3] = 255
        self.elements.append(element)

    def compose(self):
        if len(self.elements) == 0:
            raise ValueError(f"This {self.__class__.__name__} does not have elements!")

        if len(self.elements) > 1:
            result = self.composite(self.elements[0], self.elements[1])
            for element in self.elements[:2]:
                result = self.composite(result, element)
        elif len(self.elements) == 1:
            result = self.elements[0]
        else:
            shape = (self.shape[0], self.shape[1], 4)
            result = np.zeros(shape, dtype=np.uint8)

        if self.backside:
            result = np.fliplr(result)
        return result

    def not_empty(self):
        return len(self.elements)


class DrawingLayer(DigitalCompositionLayer):
    def __init__(self, *args, composite=GammaCorrectedAlphaComposition(), **kwargs):
        super().__init__(*args, composite=composite, **kwargs)
        self.elements: List[np.ndarray] = []


class SubstrateLayer(Layer):
    def __init__(self, shape: Tuple[int, int] = None, background_color: Union[tuple, np.ndarray] = None,
                 background_texture: Union[np.ndarray, str] = None,
                 override_opacity: Union[int, float] = None, **kwargs):
        if background_texture is not None:
            if isinstance(background_texture, np.ndarray):
                self.background = background_texture
            else:
                self.background = cv2.imread(background_texture, -1)
                if self.background.shape[2] == 1:
                    self.background = cv2.cvtColor(self.background, cv2.COLOR_GRAY2RGBA)
                elif self.background.shape[2] == 3:
                    self.background = cv2.cvtColor(self.background, cv2.COLOR_RGB2RGBA)

            super().__init__(self.background.shape, **kwargs)
        elif shape is not None and background_color is not None:
            super().__init__(shape, **kwargs)
            self.background = np.zeros((self.shape[0], self.shape[1], 4), dtype=np.uint8)
            self.background[:, :, :] = background_color
        else:
            raise RuntimeError("Either background_texture OR shape and background_color must be set.")

        if override_opacity is not None:
            self.background[:, :, 3] = uint8_from_number(override_opacity)

    def compose(self):
        return self.background
