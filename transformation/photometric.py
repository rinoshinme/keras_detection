"""
photometric transformations for object detection
"""

import numpy as np
import cv2


class RGB2HSV(object):
    def __init__(self):
        pass

    def __call__(self, image, label=None):
        pass


class HSV2RGB(object):
    def __init__(self):
        pass

    def __call__(self, image, label=None):
        pass


class ConvertTo3Channel(object):
    def __init__(self):
        pass

    def __call__(self, image, label=None):
        assert isinstance(image, np.ndarray)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3:
            if image.shape[2] == 1:
                image = np.concatenate([image] * 3, axis=-1)
            elif image.shape[2] == 4:
                image = image[:, :, :3]

        if label is None:
            return image
        else:
            return image, label


class Hue(object):
    def __init__(self, delta):
        self.delta = delta

    def __call__(self, image, label=None):
        pass
