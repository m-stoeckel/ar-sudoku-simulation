from abc import ABC
from pathlib import Path
from typing import Union, List

import cv2
import numpy as np

from sudoku import Color


class Layer:
    def __init__(self, shape: tuple, backside=False, bg_color=Color.NONE):
        self.backside = backside
        self.shape = shape
        self.elements: List[np.ndarray] = []
        self.background_color = bg_color.value

        # composition callback, see child classes
        self.composite = self.composite_noop

    def compose(self):
        shape = (self.shape[0], self.shape[1], 4)
        result = np.empty(shape, dtype=np.uint8)
        result[:, :, :] = self.background_color

        for element in self.elements:
            result = self.composite(result, element)

        return result

    def composite_noop(self, color1, color2):
        raise NotImplementedError("Do not use the base Layer class!")


class DigitalCompositionLayer(Layer, ABC):
    COMPOSITE_OVERLAY = 0
    COMPOSITE_OVERLAY_PRECISE = 1
    COMPOSITE_ADD = 2
    COMPOSITE_SUBTRACT = 3
    COMPOSITE_AVERAGE = 4
    COMPOSITE_MULTIPLY = 5
    COMPOSITE_THRESHOLD = 6

    def __init__(self, shape: tuple, composite=COMPOSITE_OVERLAY, composite_kwargs=None, **kwargs):
        super().__init__(shape, **kwargs)
        self.composite = [
            self.composite_alpha,
            self.composite_alpha_precise,
            self.composite_add,
            self.composite_subtract,
            self.composite_multiply,
            self.composite_threshold
        ][composite]

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
        return super(DigitalCompositionLayer, self).compose()

    @staticmethod
    def composite_add(color1, color2):
        return cv2.add(color1, color2)

    @staticmethod
    def composite_subtract(color1, color2):
        return cv2.subtract(color1, color2)

    @staticmethod
    def composite_average(color1, color2):
        return cv2.addWeighted(color1, 0.5, color2, 0.5, 0)

    @staticmethod
    def composite_multiply(color1, color2):
        return cv2.multiply(color1, color2)

    @staticmethod
    def composite_threshold(background, foreground, reset_alpha=True):
        mask = DigitalCompositionLayer.get_mask_from_alpha(foreground)
        img = cv2.bitwise_and(background, background, mask=mask)
        img = cv2.add(img, foreground)
        if reset_alpha:
            img[:, :, 3] = 0
            mask = DigitalCompositionLayer.get_bool_mask_from_alpha(img)
            img[mask, 3] = 255
        return img

    @staticmethod
    def get_mask_from_alpha(image):
        _, mask = cv2.threshold(image[:, :, 3], 1, 255, cv2.THRESH_BINARY)
        return cv2.bitwise_not(mask)

    @staticmethod
    def get_bool_mask_from_alpha(image):
        return DigitalCompositionLayer.get_mask_from_alpha(image).astype(np.bool)

    @staticmethod
    def get_mask_from_color(image):
        mask = DigitalCompositionLayer.get_bool_mask_from_color(image)
        ret = np.zeros_like(mask, 0, dtype=np.uint8)
        ret[mask] = 255
        return ret

    @staticmethod
    def get_bool_mask_from_color(image):
        return np.any(image[:, :, :3] != 255, axis=2)

    @staticmethod
    def composite_alpha(background_rgba, overlay_rgba, alpha=1., gamma=1.):
        if background_rgba.shape[2] == 3:
            background_rgba = cv2.cvtColor(background_rgba, cv2.COLOR_RGB2RGBA)
        return DigitalCompositionLayer._composite_alpha(background_rgba, overlay_rgba, alpha, gamma)

    @staticmethod
    def composite_alpha_precise(background_rgba, overlay_rgba, alpha=1., gamma=2.2):
        if background_rgba.shape[2] == 3:
            background_rgba = cv2.cvtColor(background_rgba, cv2.COLOR_RGB2RGBA)
        return DigitalCompositionLayer._composite_alpha(background_rgba, overlay_rgba, alpha, gamma)

    @staticmethod
    def _composite_alpha(background_rgba, overlay_rgba, alpha=1., gamma=1.):
        """
        TODO: comment

        :sources:
            - https://en.wikipedia.org/wiki/Alpha_compositing#Alpha_blending
            - https://gist.github.com/pthom/5155d319a7957a38aeb2ac9e54cc0999
        """
        # get normalized alpha channels and scale overlay with alpha parameter
        overlay_alpha = DigitalCompositionLayer.get_alpha_from_rgba(overlay_rgba)
        overlay_alpha *= alpha

        background_alpha = DigitalCompositionLayer.get_alpha_from_rgba(background_rgba)

        out_alpha = overlay_alpha + background_alpha * (1 - overlay_alpha)

        # convert RGBs values to float and scale with gamma
        overlay_rgb_f = overlay_rgba[:, :, : 3].astype(np.float)
        if gamma != 1.:
            overlay_rgb_f = np.float_power(overlay_rgb_f, gamma)

        background_rgb_f = background_rgba[:, :, : 3].astype(np.float)
        if gamma != 1.:
            background_rgb_f = np.float_power(background_rgb_f, gamma)

        # compute output rgb and compensate for gamma if needed
        out_rgb = (overlay_rgb_f * overlay_alpha +
                   background_rgb_f * background_alpha * (1. - overlay_alpha)) / out_alpha
        if gamma != 1.:
            out_rgb = np.float_power(out_rgb, 1. / gamma)

        # convert back to np.uint8 and rescale out_alpha to 0..255
        out_rgb = out_rgb.astype(np.uint8)
        out_alpha = (out_alpha * 255.).astype(np.uint8)
        return np.dstack((out_rgb, out_alpha))

    @staticmethod
    def get_alpha_from_rgba(overlay_rgba):
        overlay_alpha = overlay_rgba[:, :, 3].astype(np.float) / 255.
        overlay_alpha = np.expand_dims(overlay_alpha, 2)
        return overlay_alpha


class DrawingLayer(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class BackgroundLayer(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class BacksideLayer(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class LayeredPaperRenderer:
    def __init__(self, shape=(1000, 1000)):
        self.shape = shape
        self.backside_layer = DigitalCompositionLayer(shape, backside=True)
        self.background_layer = BackgroundLayer(shape)
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
