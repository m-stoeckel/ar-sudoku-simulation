from pathlib import Path
from typing import Union, List

import cv2
import numpy as np

from sudoku import Color
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


class DrawingLayer(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SubstrateLayer(Layer):
    def __init__(self, shape=None, background_color: Color = None,
                 background_texture: Union[np.ndarray, str] = None,
                 opacity=None, **kwargs):
        if background_texture is not None:
            if isinstance(background_texture, np.ndarray):
                self.background = background_texture
            else:
                self.background = cv2.imread(background_texture, -1)
                if self.background.shape[2] == 1:
                    self.background = cv2.cvtColor(self.background, cv2.COLOR_GRAY2RGBA)
                elif self.background.shape[2] == 3:
                    self.background = cv2.cvtColor(self.background, cv2.COLOR_RGB2RGBA)

                if opacity is not None:
                    self.background[:, :, 3] = self.as_int(opacity)

            super().__init__(self.background.shape, **kwargs)
        elif shape is not None and background_color is not None:
            super().__init__(shape, **kwargs)
            self.background = np.full(self.shape, background_color.value)
        else:
            raise RuntimeError

        self.prints: List[DigitalCompositionLayer] = []
        self.drawings: List[DrawingLayer] = []

    def add_print(self, layer: DigitalCompositionLayer):
        self.prints.append(layer)

    def add_drawing(self, layer: DrawingLayer):
        self.drawings.append(layer)

    def compose(self):
        for print_layer in self.prints:
            if not print_layer.backside:
                ...
            else:
                ...

        for drawing in self.drawings:
            if not drawing.backside:
                ...
            else:
                ...

        return ...

    @staticmethod
    def as_int(opacity):
        if type(opacity) is float:
            return np.clip(opacity * 255, 0, 255).astype(np.uint8)
        else:
            return np.clip(opacity, 0, 255).astype(np.uint8)


class LayeredPaperRenderer:
    def __init__(self, shape=(1000, 1000)):
        self.shape = shape
        self.backside_layer = DigitalCompositionLayer(shape, backside=True)
        self.background_layer = SubstrateLayer(shape)
        self.printing_layer = DigitalCompositionLayer(shape)
        self.drawing_layer = DrawingLayer(shape)


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
