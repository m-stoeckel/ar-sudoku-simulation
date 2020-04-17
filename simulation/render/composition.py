from abc import ABCMeta, abstractmethod

import cv2
import numpy as np


class DigitalCompositionMethod(metaclass=ABCMeta):
    """
    Abstract base class for all composition methods.
    """

    @abstractmethod
    def apply(self, background_rgba: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
        pass

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)


class Add(DigitalCompositionMethod):
    """
    Adds two images. Resulting values are computed as full integer and clipped to uint8 afterwards.
    """

    def apply(self, background_rgba: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
        return np.clip(background_rgba.astype(np.int) + overlay_rgba.astype(np.int), 0, 255).astype(np.uint8)


class Subtract(DigitalCompositionMethod):
    """
    Subtracts two images. The overlay image is converted to a RGB float array by multiplying each color channel with the
    normalized alpha channel. The alpha of the background image remains unchanged. Resulting values are clipped to uint8
    afterwards.
    """

    def apply(self, background_rgba: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
        bg = background_rgba.astype(np.float)
        fg = overlay_rgba.astype(np.float)
        norm_alpha = np.expand_dims(fg[:, :, 3] / 255., axis=2).repeat(3, axis=2)
        bg[:, :, :3] -= fg[:, :, :3] * norm_alpha
        return np.clip(bg, 0, 255).astype(np.uint8)


class Average(DigitalCompositionMethod):
    """
    Computes the average of two images using :py:meth:`cv2.addWeighted()` with equal weights of 0.5.
    """

    def apply(self, background_rgba: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
        return cv2.addWeighted(background_rgba, 0.5, overlay_rgba, 0.5, 0)


class Multiply(DigitalCompositionMethod):
    """
    Multiplies two images. Resulting values are computed as full integer and clipped to uint8 afterwards.
    """

    def apply(self, background_rgba: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
        return np.clip(background_rgba.astype(np.int) * overlay_rgba.astype(np.int), 0, 255).astype(np.uint8)


class AlphaClip(DigitalCompositionMethod):
    """
    Composes two images by clipping away all pixels of the overlay image whose alpha value is 0.
    """

    def __init__(self, reset_alpha=True):
        """

        Args:
            reset_alpha(bool): If True, reset the alpha channel after composing the two images.

        """
        super().__init__()
        self.reset_alpha = reset_alpha

    def apply(self, background_rgba: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
        mask = self.get_mask_from_alpha(overlay_rgba)
        img = cv2.bitwise_and(background_rgba, background_rgba, mask=mask)
        img = cv2.add(img, overlay_rgba)
        if self.reset_alpha:
            mask = self.get_bool_mask_from_color(img)
            img[:, :, 3] = 0
            img[mask, 3] = 255
        return img

    @staticmethod
    def get_mask_from_alpha(image):
        _, mask = cv2.threshold(image[:, :, 3], 1, 255, cv2.THRESH_BINARY)
        return cv2.bitwise_not(mask)

    @staticmethod
    def get_bool_mask_from_alpha(image):
        return AlphaClip.get_mask_from_alpha(image).astype(np.bool)

    @staticmethod
    def get_mask_from_color(image):
        mask = AlphaClip.get_bool_mask_from_color(image)
        ret = np.zeros_like(mask, 0, dtype=np.uint8)
        ret[mask] = 255
        return ret

    @staticmethod
    def get_bool_mask_from_color(image):
        return np.any(image[:, :, :3] != 255, axis=2)


class AlphaComposition(DigitalCompositionMethod):
    """
    Combines the input images using alpha composition.
    """

    def __init__(self, alpha=1.0, gamma=1.0):
        """

        Args:
            alpha(float): Alpha scaling factor. (Default value = 1.0)
            gamma(float): Gamma value for the alpha composition algorithm. (Default value: 1.0)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def apply(self, background_rgba, overlay_rgba) -> np.ndarray:
        if background_rgba.shape[2] == 3:
            background_rgba = cv2.cvtColor(background_rgba, cv2.COLOR_RGB2RGBA)
        return self._composite_alpha(background_rgba, overlay_rgba)

    def _composite_alpha(self, background_rgba: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
        """
        Applies alpha composition to the two given images.

        :sources:
            - https://en.wikipedia.org/wiki/Alpha_compositing#Alpha_blending
            - https://gist.github.com/pthom/5155d319a7957a38aeb2ac9e54cc0999

        Args:
            background_rgba(:py:class:`numpy.ndarray`): Background image.
            overlay_rgba(:py:class:`numpy.ndarray`): Overlay image.

        Returns:
            :py:class:`numpy.ndarray`: The composed image.

        """
        # get normalized alpha channels and scale overlay with alpha parameter
        overlay_alpha = self.get_alpha_from_rgba(overlay_rgba)
        overlay_alpha *= self.alpha

        background_alpha = self.get_alpha_from_rgba(background_rgba)

        out_alpha = overlay_alpha + background_alpha * (1 - overlay_alpha)

        # convert RGBs values to float and scale with gamma
        overlay_rgb_f = overlay_rgba[:, :, : 3].astype(np.float)
        if self.gamma != 1.:
            overlay_rgb_f = np.float_power(overlay_rgb_f, self.gamma)

        background_rgb_f = background_rgba[:, :, : 3].astype(np.float)
        if self.gamma != 1.:
            background_rgb_f = np.float_power(background_rgb_f, self.gamma)

        # compute output rgb and compensate for gamma if needed
        out_rgb = (overlay_rgb_f * overlay_alpha +
                   background_rgb_f * background_alpha * (1. - overlay_alpha)) / (out_alpha + 1e-5)
        if self.gamma != 1.:
            out_rgb = np.float_power(out_rgb, 1. / self.gamma)

        # convert back to np.uint8 and rescale out_alpha to 0..255
        out_rgb = out_rgb.astype(np.uint8)
        out_alpha = (out_alpha * 255.).astype(np.uint8)
        return np.dstack((out_rgb, out_alpha))

    @staticmethod
    def get_alpha_from_rgba(overlay_rgba):
        overlay_alpha = overlay_rgba[:, :, 3].astype(np.float) / 255.
        overlay_alpha = np.expand_dims(overlay_alpha, 2)
        return overlay_alpha


class GammaCorrectedAlphaComposition(AlphaComposition, DigitalCompositionMethod):
    """
    A variant of :py:class:`AlphaComposition` that uses 2.2 as default gamma.
    """

    def __init__(self, alpha=1.0, gamma=2.2):
        """

        Args:
            alpha(float): Alpha scaling factor. (Default value = 1.0)
            gamma(float): Gamma value for the alpha composition algorithm. (Default value: 2.2)
        """
        super().__init__(alpha, gamma)
