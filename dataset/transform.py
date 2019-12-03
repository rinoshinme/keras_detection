"""
define all kinds of image transformations
"""
import cv2
import numpy as np

IMAGENET_MEAN = np.array([127, 127, 127], dtype=np.float)


class ImageTransformer(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def apply(self, image):
        for t in self.transforms:
            image = t(image)
        return image


class Resize(object):
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, image):
        return cv2.resize(image, self.target_size)


class Normalizer(object):
    def __init__(self, mean=None, dev=None):
        if mean is None:
            self.mean = IMAGENET_MEAN
        else:
            self.mean = mean
        self.dev = dev

    def __call__(self, image):
        image = image - self.mean
        if self.dev is not None:
            image = image / self.dev
        return image


class ChannelShuffler(object):
    def __init__(self):
        pass

    def __call__(self, image):
        pass


if __name__ == '__main__':
    ts = [Resize((224, 224))]
    transformer = ImageTransformer(ts)
    image_path = r'D:\data\cat_and_dog\demo\cat\1.jpg'
    img = cv2.imread(image_path)
    res = transformer.apply(img)
    print(res.shape)
