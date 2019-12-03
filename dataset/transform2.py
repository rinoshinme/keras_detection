"""
image transformation that is related to image bounding boxes
"""


class ImageLabelTransformer(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def apply(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)


class Cropping(object):
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def __call__(self, image, label):
        return image, label
