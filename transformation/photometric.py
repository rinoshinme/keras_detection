"""
photometric transformations for object detection
"""

import numpy as np
import cv2


class RGB2HSV(object):
    def __init__(self):
        pass

    def __call__(self, image, labels=None):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        if labels is None:
            return image
        else:
            return image, labels


class HSV2RGB(object):
    def __init__(self):
        pass

    def __call__(self, image, labels=None):
        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        if labels is None:
            return image
        else:
            return image, labels


class ChannelSwap(object):
    def __init__(self, indices=(2, 1, 0)):
        self.indices = indices

    def __call__(self, image, labels=None):
        pass


class RandomChannelSwap(object):
    def __init__(self):
        pass

    def __call__(self, image, labels=None):
        pass


class ConvertTo3Channel(object):
    def __init__(self):
        pass

    def __call__(self, image, labels=None):
        assert isinstance(image, np.ndarray)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3:
            if image.shape[2] == 1:
                image = np.concatenate([image] * 3, axis=-1)
            elif image.shape[2] == 4:
                image = image[:, :, :3]

        if labels is None:
            return image
        else:
            return image, labels


class Hue(object):
    def __init__(self, delta):
        if not (-180 <= delta <= 180):
            raise ValueError("delta must be in range [-180, 180]")
        self.delta = delta

    def __call__(self, image, labels=None):
        image[:, :, 0] = (image[:, :, 0] + self.delta) % 180.0
        if labels is None:
            return image
        else:
            return image, labels


class RandomHue(object):
    def __init__(self, max_delta=18, prob=0.5):
        self.max_delta = max_delta
        self.prob = prob

    def __call__(self, image, labels=None):
        p = np.random.uniform(0, 1)
        if p >= 1.0 - self.prob:
            delta = np.random.uniform(-self.max_delta, self.max_delta)
            image[:, :, 0] = (image[:, :, 0] + delta) % 180.0
        if labels is None:
            return image
        else:
            return image, labels


class Saturation(object):
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, image, labels=None):
        image[:, :, 1] = np.clip(image[:, :, 1] * self.factor, 0, 255)
        if labels is None:
            return image
        else:
            return image, labels


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.upper = upper
        self.lower = lower

    def __call__(self, image, labels=None):
        factor = np.random.uniform(self.lower, self.upper)
        image[:, :, 1] = np.clip(image[:, :, 1] * factor, 0, 255)
        if labels is None:
            return image
        else:
            return image, labels


class Brightness(object):
    def __init__(self, delta):
        self.delta = delta

    def __call__(self, image, labels=None):
        image = np.clip(image + self.delta, 0, 255)
        if labels is not None:
            return image
        else:
            return image, labels


class RandomBrightness(object):
    def __init__(self, max_delta):
        self.max_delta = max_delta

    def __call__(self, image, labels=None):
        delta = np.random.uniform(-self.max_delta, self.max_delta)
        image = np.clip(image + delta, 0, 255)
        if labels is not None:
            return image
        else:
            return image, labels


class Contrast(object):
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, image, labels=None):
        image = np.clip(127.5 + (image - 127.5) * self.factor, 0, 255)
        if labels is None:
            return image
        else:
            return image, labels


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper

    def __call__(self, image, labels):
        factor = np.random.uniform(self.lower, self.upper)
        image = np.clip(127.5 + (image - 127.5) * factor, 0, 255)
        if labels is None:
            return image
        else:
            return image, labels


class Gamma(object):
    def __init__(self, gamma):
        self.gamma = gamma
        self.lut = []

    def __call__(self, image, labels):
        pass


class RandomGamma(object):
    def __init__(self, lower=0.25, upper=2):
        self.lower = lower
        self.upper = upper

    def generate_loopup_table(self, gamma_value):
        pass

    def __call__(self, image, labels=None):
        factor = np.random.uniform(self.lower, self.upper)
        print(factor)
