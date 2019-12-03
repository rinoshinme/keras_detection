import tensorflow as tf
from keras.models import Model
from datetime import datetime
from dataset.dataset_pascalvoc import DatasetPascalvoc
from dataset.data_generator import get_ssd_input_transform, DataGenerator
from models.ssd300 import SSD300
from utils.ssd_encoder import SSDEncoder
from layers.ssd_loss import SSDLoss
from utils.train import get_optimizer
from utils.system import get_os_name

os_name = get_os_name()


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

anchor_params_ssd512 = {
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
    'pretrained_weights': None,
    'anchor_params': anchor_params_ssd300,
    'conf_thresh': 0.5,
    'iou_thresh_high': 0.5,
    'iou_thresh_low': 0.3,

    # dataset parameters
    'dataset_dirs': ['D:/data/VOCdevkit/VOC2007', 'D:/data/VOCdevkit/VOC2012'],
    'dataset_pickle': './data/voc0712.pkl',
    'input_transform': None,
    'batch_size': 1,

    # training parameters
    'optimizer': 'Adam',
    'learn_rate': 1.0e-4,
    'train_epochs': 100,
    'train_steps': 20000,
    'display_step': 10,
    'checkpoint_step': 2000,
}


class SSDTrainer(object):
    def __init__(self, cfg):
        self.cfg = cfg
        # model params
        self.model_name = self.cfg['model_name']
        self.image_height = self.cfg['image_height']
        self.image_width = self.cfg['image_width']
        self.num_classes = self.cfg['num_classes']
        self.pretrained_weights = self.cfg['pretrained_weights']
        if self.pretrained_weights is None:
            if os_name == 'Windows':
                self.pretrained_weights = r'D:\code\tools\keras_models\vgg16_weights_tf_dim_ordering_tf_kernels.h5',
            elif os_name == 'Linux':
                self.pretrained_weights = '/violence/code/pretrained_models/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
            else:
                raise ValueError('operating system currently not supported')

        # data params
        self.dataset_dirs = self.cfg['dataset_dirs']
        self.dataset_pickle = self.cfg['dataset_pickle']
        self.input_transform = self.cfg['input_transform']
        if self.input_transform is None:
            self.input_transform = get_ssd_input_transform()
        self.batch_size = self.cfg['batch_size']

        # train params
        optimizer_name = self.cfg['optimizer']
        learn_rate = self.cfg['learn_rate']
        self.optimizer = get_optimizer(optimizer_name, learn_rate)
        self.train_steps = self.cfg['train_steps']

        # other
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
        # self.output_decoder = None

        self._build_data()
        self._build_model()
        self._build_trainer()

    def _build_data(self):
        dataset = DatasetPascalvoc()
        if self.dataset_pickle is not None:
            print('loading from pickled data...')
            dataset.unpickle(self.dataset_pickle)
            print('done loading data')
        else:
            dataset.build(self.dataset_dirs, phase='train')
            print('saving data into pickle...')
            dataset.pickle('./data/voc0712.pkl')
            print('done saving data')
        dataset.shuffle()

        data_generator = DataGenerator(dataset, self.input_transform, self.batch_size, shuffle=True)
        self.data_generator = data_generator.generate()

    def _build_model(self):
        self.model = SSD300(self.num_classes, self.anchor_params, phase='train')

        self.keras_model = Model(inputs=self.model.input, outputs=self.model.output)
        # load pretrained weights
        print('loading pretrained weights...')
        self.keras_model.load_weights(self.pretrained_weights, by_name=True)
        print('done loading weights')

        self.input = self.model.input
        output_shape = self.model.output.get_shape().as_list()  # [None, 8732, 33]
        self.target = tf.placeholder(shape=output_shape, dtype=tf.float32, name='y_target')
        ssdloss = SSDLoss()
        self.loss = ssdloss.compute_loss(self.target, self.model.output)

    def _build_trainer(self):
        self.train_op = self.optimizer.minimize(self.loss)
        self.session = tf.Session()
        # additional metrics

    def train(self):
        dt = datetime.now()
        dt_str = dt.strftime('%Y%m%d_%H%M%S')
        output_dir = './logs/{}'.format(dt_str)
        print('saving logs and trained weights into {}'.format(output_dir))

        steps = 0
        self.session.run(tf.global_variables_initializer())
        for image, labels in self.data_generator:
            gt_labels = self.input_encoder.encode(labels)
            feed_dict = {
                self.input: image,
                self.target: gt_labels
            }
            _, loss_value = self.session.run([self.train_op, self.loss], feed_dict=feed_dict)

            print('train loss = {}'.format(loss_value))

            steps += 1
            if steps > self.train_steps:
                break


if __name__ == '__main__':
    task = 'train'  # ['train', 'data']

    if task == 'train':
        trainer = SSDTrainer(config)
        trainer.train()
    else:
        # generate data
        class_names = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
                       'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
                       'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
        if os_name == 'Windows':
            voc2007_dir = r'D:\data\VOCdevkit\VOC2007'
            voc2012_dir = r'D:\data\VOCdevkit\VOC2012'
            trainset = DatasetPascalvoc(class_names)
            trainset.build([voc2007_dir, voc2012_dir], 'train')
            trainset.pickle('./data/voc0712.pkl')
            print(trainset.size)
        elif os_name == 'Linux':
            voc2007_dir = '/violence/data/VOCdevkit/VOC2007'
            voc2012_dir = '/violence/data/VOCdevkit/VOC2012'
            trainset = DatasetPascalvoc()
            trainset.build([voc2007_dir, voc2012_dir], 'train')
            trainset.pickle('./data/voc0712.pkl')
            print(trainset.size)
        else:
            raise ValueError('operating system currently not supported')
