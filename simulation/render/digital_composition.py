import cv2
import numpy as np


class DigitalCompositionMethod:
    def apply(self, background_rgba: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)


class Add(DigitalCompositionMethod):
    def apply(self, background_rgba: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
        return np.clip(background_rgba.astype(np.int) + overlay_rgba.astype(np.int), 0, 255).astype(np.uint8)


class Subtract(DigitalCompositionMethod):
    def apply(self, background_rgba: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
        return np.clip(background_rgba.astype(np.int) - overlay_rgba.astype(np.int), 0, 255).astype(np.uint8)


class Average(DigitalCompositionMethod):
    def apply(self, background_rgba: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
        return cv2.addWeighted(background_rgba, 0.5, overlay_rgba, 0.5, 0)


class Multiply(DigitalCompositionMethod):
    def apply(self, background_rgba: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
        return np.clip(background_rgba.astype(np.int) * overlay_rgba.astype(np.int), 0, 255).astype(np.uint8)


class Threshold(DigitalCompositionMethod):
    def __init__(self, reset_alpha=True):
        super().__init__()
        self.reset_alpha = reset_alpha

    def apply(self, background_rgba: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
        mask = self.get_mask_from_alpha(overlay_rgba)
        img = cv2.bitwise_and(background_rgba, background_rgba, mask=mask)
        img = cv2.add(img, overlay_rgba)
        if self.reset_alpha:
            img[:, :, 3] = 0
            mask = self.get_bool_mask_from_alpha(img)
            img[mask, 3] = 255
        return img

    @staticmethod
    def get_mask_from_alpha(image):
        _, mask = cv2.threshold(image[:, :, 3], 1, 255, cv2.THRESH_BINARY)
        return cv2.bitwise_not(mask)

    @staticmethod
    def get_bool_mask_from_alpha(image):
        return Threshold.get_mask_from_alpha(image).astype(np.bool)

    @staticmethod
    def get_mask_from_color(image):
        mask = Threshold.get_bool_mask_from_color(image)
        ret = np.zeros_like(mask, 0, dtype=np.uint8)
        ret[mask] = 255
        return ret

    @staticmethod
    def get_bool_mask_from_color(image):
        return np.any(image[:, :, :3] != 255, axis=2)


class AlphaComposition(DigitalCompositionMethod):
    def __init__(self, alpha=1.0, gamma=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def apply(self, background_rgba, overlay_rgba) -> np.ndarray:
        if background_rgba.shape[2] == 3:
            background_rgba = cv2.cvtColor(background_rgba, cv2.COLOR_RGB2RGBA)
        return self._composite_alpha(background_rgba, overlay_rgba)

    def _composite_alpha(self, background_rgba, overlay_rgba):
        """
        TODO: comment

        :sources:
            - https://en.wikipedia.org/wiki/Alpha_compositing#Alpha_blending
            - https://gist.github.com/pthom/5155d319a7957a38aeb2ac9e54cc0999
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
    def __init__(self, alpha=1.0, gamma=2.2):
        super().__init__(alpha, gamma)
