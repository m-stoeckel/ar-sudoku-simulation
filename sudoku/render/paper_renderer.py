from pathlib import Path
from typing import Union, List, Tuple

import cv2
import numpy as np

from sudoku import Color
from sudoku.render.colors import uint8_from_number
from sudoku.render.digital_composition import AlphaComposition


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

        :param element: A numpy.ndarray of the same shape as existing elements.
        If element is grayscale or RGB, it will be converted to RGBA.
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
            result = np.empty(shape, dtype=np.uint8)

        if self.backside:
            result = np.fliplr(result)
        return result

    def not_empty(self):
        return len(self.elements)


class DrawingLayer(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.elements: List[np.ndarray] = []

    def not_empty(self):
        return len(self.elements)


class SubstrateLayer(Layer):
    def __init__(self, shape: Tuple[int, int] = None, background_color: Color = None,
                 background_texture: Union[np.ndarray, str] = None,
                 override_opacity: Union[int, float] = None, print_area=0.98,
                 **kwargs):
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
            self.background = np.full((self.shape[0], self.shape[1], 4), background_color.value, dtype=np.uint8)
        else:
            raise RuntimeError("Either background_texture OR shape and background_color must be set.")

        if override_opacity is not None:
            self.background[:, :, 3] = uint8_from_number(override_opacity)

    def compose(self):
        return self.background


class LayeredPaperRenderer:
    def __init__(self, substrate_layer: SubstrateLayer, shape=(1000, 1000), print_area=0.98):
        self.shape = shape
        self.print_area = np.clip(print_area, 0.0, 1.0)

        self.backside_drawing_layer = DrawingLayer(shape, backside=True)
        self.backside_print_layer = DigitalCompositionLayer(shape, backside=True)
        self.background_layer = substrate_layer
        self.print_layer = DigitalCompositionLayer(shape)
        self.drawing_layer = DrawingLayer(shape)

    def render(self):
        draw_composition = AlphaComposition()
        print_composition = AlphaComposition(self.print_area)

        result = self.background_layer.compose()

        if self.backside_drawing_layer.not_empty():
            result = print_composition(self.backside_drawing_layer.compose(), result)
        if self.backside_print_layer.not_empty():
            result = draw_composition(self.backside_print_layer.compose(), result)

        if self.print_layer.not_empty():
            result = print_composition(result, self.print_layer.compose())
        if self.drawing_layer.not_empty():
            result = draw_composition(result, self.drawing_layer.compose())

        return result


class MeshedLayeredPaperRenderer(LayeredPaperRenderer):
    def __init__(self, mesh=None, mesh_file: Union[str, Path] = None, **kwargs):
        super().__init__(**kwargs)
        if mesh is not None:
            self.mesh = mesh
        elif mesh_file is not None:
            self.mesh = self.load_mesh(Path(mesh_file))
        else:
            raise ValueError("Either a mesh or a path to a valid mash file must be passed!")

    def load_mesh(self, mesh_file: Path):
        # TODO: Implement mesh loading
        raise NotImplementedError
