import tensorflow as tf
from models.ssd300 import SSD300
from utils.ssd_encoder import SSDEncoder


anchor_params_ssd300 = {
    'scales': [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05],
    'feature_map_sizes': [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
    'aspect_ratios': [[1, 2.0, 0.5],
                      [1, 2.0, 0.5, 3.0, 1.0/3.0],
                      [1, 2.0, 0.5, 3.0, 1.0/3.0],
                      [1, 2.0, 0.5, 3.0, 1.0/3.0],
                      [1, 2.0, 0.5],
                      [1, 2.0, 0.5]],
    'two_boxes_for_ar1': True,
    'num_boxes': [4, 6, 6, 6, 4, 4],
    'steps': [8, 16, 32, 64, 100, 300],
    'offset': 0.5,
    'clip_boxes': False,
    'variances': [0.1, 0.1, 0.2, 0.2],
}

config = {
    # model parameters
    'model_name': 'ssd300',
    'image_height': 300,
    'image_width': 300,
    'class_names': ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                    'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                    'sofa', 'train', 'tvmonitor'),
    'num_classes': 20,
    'trained_weights': None,
    'anchor_params': anchor_params_ssd300,
    'conf_thresh': 0.5,
    'iou_thresh_high': 0.5,
    'iou_thresh_low': 0.3,

    # dataset parameters
    'dataset_dirs': ['D:/data/VOCdevkit/VOC2007', 'D:/data/VOCdevkit/VOC2012'],
    'dataset_pickle': './data/voc0712.pkl',
    'input_transform': None,
    'batch_size': 1,
}


class SSDTester(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_name = self.cfg['model_name']
        self.image_height = self.cfg['image_height']
        self.image_width = self.cfg['image_width']
        self.num_classes = self.cfg['num_classes']
        self.trainde_weights = self.cfg['trained_weights']

        self.anchor_params = self.cfg['anchor_params']
        self.conf_thresh = self.cfg['conf_thresh']
        self.iou_thresh_high = self.cfg['iou_thresh_high']
        self.iou_thresh_low = self.cfg['iou_thresh_low']
        self.input_encoder = SSDEncoder(self.image_height, self.image_width, self.num_classes,
                                        self.anchor_params['feature_map_sizes'],
                                        self.anchor_params['scales'],
                                        self.anchor_params['aspect_ratios'],
                                        self.anchor_params['two_boxes_for_ar1'],
                                        self.anchor_params['steps'],
                                        self.anchor_params['offset'],
                                        self.anchor_params['clip_boxes'],
                                        self.anchor_params['variances'],
                                        self.iou_thresh_high,
                                        self.iou_thresh_low)
        self.output_decoder = None

        self._build_model()
        self._build_graph()

    def _build_model(self):
        self.model = SSD300(self.num_classes, self.anchor_params, phase='train')
        self.input = self.model.input
        self.output = self.model.output

    def _build_graph(self):
        self.saver = tf.train.Saver()
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def inference(self, image_path):
        pass


if __name__ == '__main__':
    pass
