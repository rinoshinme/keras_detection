import numpy as np
from keras.engine.topology import InputSpec
from keras.engine.topology import Layer
import tensorflow as tf
from utils.bbox import center2minmax, minmax2center

#
example_params = {
    'image_width': 300,
    'image_height': 300,
    'scale': 1.0,
    'next_scale': 0.1,
    'aspect_ratios': [1, 2.0, 0.5, 3.0, 1.0/3.0],
    'two_boxes_for_ar1': True,
    'step': 16,
    'offset': 0.5,  # [0.5, 0.5]
    'clip_box': True,
    'variances': [0.1, 0.1, 0.2, 0.2],
    'normalize_coord': True,
}


class AnchorBoxes(Layer):
    def __init__(self, args):
        # self.args = args
        self.image_width = args['image_width']
        self.image_height = args['image_height']
        self.scale = args['scale']
        self.next_scale = args['next_scale']
        self.aspect_ratios = args['aspect_ratios']
        self.two_boxes_for_ar1 = args['two_boxes_for_ar1']
        self.step = args['step']
        self.offset = args['offset']
        self.clip_box = args['clip_box']
        self.variance = args['variances']
        # TODO
        #  self.normalize_coord = args['normalize_coord']

        if 1 in self.aspect_ratios and self.two_boxes_for_ar1:
            self.n_boxes = len(self.aspect_ratios) + 1
        else:
            self.n_boxes = len(self.aspect_ratios)
        super(AnchorBoxes, self).__init__()

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        super(AnchorBoxes, self).build(input_shape)

    def call(self, feature_map, mask=None):
        # add mask parameter to avoid warning from inheriting
        size = min(self.image_width, self.image_height)

        # calculate per-pixel boxes
        wh_list = []
        for ar in self.aspect_ratios:
            if ar != 1.0:
                box_height = self.scale * size / np.sqrt(ar)
                box_width = self.scale * size * np.sqrt(ar)
                wh_list.append((box_height, box_width))
            else:
                box_width = self.scale * size
                box_height = self.scale * size
                wh_list.append((box_height, box_width))
                if self.two_boxes_for_ar1:
                    box_height = np.sqrt(self.scale * self.next_scale) * size
                    box_width = box_height
                    wh_list.append((box_height, box_width))
        wh_list = np.array(wh_list)

        # move box over every position in feature map
        feature_map_shape = feature_map.get_shape().as_list()
        feat_height = feature_map_shape[1]
        feat_width = feature_map_shape[2]

        cy = np.array([(i + self.offset) * self.step for i in range(feat_height)])
        cx = np.array([(i + self.offset) * self.step for i in range(feat_width)])
        cx_grid, cy_grid = np.meshgrid(cx, cy)

        cx_grid = np.expand_dims(cx_grid, -1)
        cy_grid = np.expand_dims(cy_grid, -1)

        # cx, cy, w, h
        boxes_tensor = np.zeros((feat_height, feat_width, self.n_boxes, 4))
        boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, self.n_boxes))
        boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, self.n_boxes))
        boxes_tensor[:, :, :, 2] = wh_list[:, 0]
        boxes_tensor[:, :, :, 3] = wh_list[:, 1]

        # do normalization
        boxes_tensor[:, :, :, 0] /= self.image_width
        boxes_tensor[:, :, :, 1] /= self.image_height
        boxes_tensor[:, :, :, 2] /= self.image_width
        boxes_tensor[:, :, :, 3] /= self.image_height

        # if self.clip_box:
        #     boxes_tensor = center2minmax(boxes_tensor)
        #     # this need to be done in (xmin, ymin, xmax, ymax) format
        #     x_coords = boxes_tensor[:, :, :, [0, 2]]
        #     x_coords[x_coords >= 1.0] = 1.0
        #     x_coords[x_coords < 0] = 0
        #     boxes_tensor[:, :, :, [0, 2]] = x_coords
        #     y_coords = boxes_tensor[:, :, :, [1, 3]]
        #     y_coords[y_coords >= 1.0] = 1.0
        #     y_coords[y_coords < 0] = 0
        #     boxes_tensor[:, :, :, [1, 3]] = y_coords
        #     boxes_tensor = minmax2center(boxes_tensor)

        # add variances to the end of boxes_tensor
        variances_tensor = np.zeros_like(boxes_tensor)
        variances_tensor += self.variance
        boxes_tensor = np.concatenate([boxes_tensor, variances_tensor], axis=-1)

        # add batch size dimension
        boxes_tensor = np.expand_dims(boxes_tensor, axis=0)

        boxes_tensor = tf.tile(tf.constant(boxes_tensor, dtype='float32'),
                               (tf.shape(feature_map)[0], 1, 1, 1, 1))

        return boxes_tensor

    def compute_output_shape(self, input_shape):
        batch_size, feature_map_height, feature_map_width, feature_map_channels = input_shape
        return batch_size, feature_map_height, feature_map_width, self.n_boxes, 8

    def get_config(self):
        new_config = {
            'img_height': self.image_height,
            'img_width': self.image_width,
            'scale': self.scale,
            'next_scale': self.next_scale,
            'aspect_ratios': list(self.aspect_ratios),
            'two_boxes_for_ar1': self.two_boxes_for_ar1,
            'clip_boxes': self.clip_box,
            'variances': list(self.variance),
            # 'coords': self.coords,
            # 'normalize_coords': self.normalize_coords
        }
        base_config = super(AnchorBoxes, self).get_config()

        config = {}
        for k, v in base_config.items():
            config[k] = v
        for k, v in new_config.items():
            config[k] = v
        return config


def test():
    feat = tf.placeholder(tf.float32, shape=[None, 100, 100, 3])
    ab = AnchorBoxes(example_params)
    boxes = ab(feat)
    print(boxes.shape)


if __name__ == '__main__':
    test()
