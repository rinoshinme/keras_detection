import cv2
import os
import numpy as np
from models.ssd300 import SSD300
from utils.ssd_encoder import SSDEncoder
from dataset.data_generator import get_ssd_input_transform


# ----------------------------- define parameters --------------------------
# model params
image_height = 300
image_width = 300
image_channels = 3

class_names = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
               'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
num_classes = len(class_names)
trained_weights = None

# anchor params
scales = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05]
feature_map_sizes = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
aspect_ratios = [[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5]]
two_boxes_for_ar1 = True
steps = [8, 16, 32, 64, 100, 300]
offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
clip_boxes = True
variances = [0.1, 0.1, 0.2, 0.2]
normalize_coords = True

input_transform = get_ssd_input_transform()

# ----------------------------- end of parameters --------------------------


class SSD300Inference(object):
    def __init__(self):
        self.input_encoder = SSDEncoder(image_height, image_width, num_classes,
                                        feature_map_sizes, scales, aspect_ratios,
                                        two_boxes_for_ar1, steps, offsets, clip_boxes,
                                        variances, 0.5, 0.5)

        self._build_model()
        self.input_transform = get_ssd_input_transform()

    def _build_model(self):
        ssd300 = SSD300(num_classes, scales, aspect_ratios, two_boxes_for_ar1, steps, offsets,
                        clip_boxes, variances, phase='inference')
        self.model = ssd300.model
        if trained_weights is not None:
            self.model.load_weights(trained_weights)

    def inference(self, image_path):
        if not os.path.exists(image_path):
            print('invalid image path')
            return
        img = cv2.imread(image_path)
        img = self.input_transform.apply(img)
        img = np.expand_dims(img, axis=0)

        prediction = self.model.predict(img)
        print(prediction)
        print(prediction.shape)
        return prediction


if __name__ == '__main__':
    tester = SSD300Inference()
    img_path = r'D:/data/20191203101320.png'
    tester.inference(img_path)
