import argparse
import os
import tensorflow as tf
import keras.backend as K
from datetime import datetime
from dataset.dataset_pascalvoc import DatasetPascalvoc
from dataset.data_generator import get_ssd_input_transform, DataGenerator
from models.ssd300 import SSD300
from utils.ssd_encoder import SSDEncoder
from layers.ssd_loss import SSDLoss
from utils.system import get_os_name

os_name = get_os_name()


# ----------------------------- define parameters --------------------------
# model params
image_height = 300
image_width = 300
image_channels = 3

class_names = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
               'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
num_classes = len(class_names)
if os_name == 'Windows':
    pretrained_weights_path = r'D:\code\tools\keras_models\vgg16_weights_tf_dim_ordering_tf_kernels.h5'
elif os_name == 'Linux':
    pretrained_weights_path = '/violence/code/pretrained_models/vgg16_weights_tf_dim_ordering_tf_kernels.h5'

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

# dataset parameters
dataset_pickle = './data/voc0712.pkl'
dataset_dirs = ['D:/data/VOCdevkit/VOC2007', 'D:/data/VOCdevkit/VOC2012']
if os_name == 'Windows':
    batch_size = 1
else:
    batch_size = 128
input_transform = get_ssd_input_transform()

# training parameters
optimizer = 'Adam'
adam_config = {'lr': 0.001, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8, 'decay': 0.0}
sgd_config = {'lr': 0.001, 'moment': 0.9, 'decay': 0.0}
max_steps = 100000
max_epochs = 150
display_steps = 10
checkpoint_steps = 2000
output_root = './logs'
# ----------------------------- end of parameters --------------------------


def lr_schedule(epoch):
    if epoch < 80:
        return 0.001
    elif epoch < 100:
        return 0.0001
    else:
        return 0.00001


class SSD300Trainer(object):
    def __init__(self):
        self._load_model()
        self._load_data()
        self._load_trainer()

    def _load_model(self):
        K.clear_session()
        ssd300 = SSD300(num_classes, scales, aspect_ratios, two_boxes_for_ar1, steps, offsets,
                        clip_boxes, variances, phase='train')
        self.model = ssd300.model  # an instance of Keras Model
        if pretrained_weights_path is not None:
            self.model.load_weights(pretrained_weights_path, by_name=True)

        self.input_x = self.model.input
        self.output = self.model.output
        output_shape = self.output.get_shape().as_list()  # [None, 8732, 33]
        self.input_y = tf.placeholder(shape=output_shape, dtype=tf.float32, name='y_target')
        self.learn_rate = tf.placeholder(tf.float32, name='learning_rate')

        # ground truth encoder
        self.input_encoder = SSDEncoder(image_height, image_width, num_classes,
                                        feature_map_sizes, scales, aspect_ratios,
                                        two_boxes_for_ar1, steps, offsets, clip_boxes,
                                        variances, 0.5, 0.5)

    def _load_data(self):
        dataset = DatasetPascalvoc(class_names)
        if dataset_pickle is not None:
            print('loading from pickled data...')
            dataset.unpickle(dataset_pickle)
            print('done loading data')
        else:
            dataset.build(dataset_dirs, phase='train')
            print('saving data into pickle...')
            dataset.pickle('./data/voc0712.pkl')
            print('done saving data')
        dataset.shuffle()

        self.data_size = dataset.size

        data_generator = DataGenerator(dataset, input_transform, batch_size, shuffle=True)
        self.data_generator = data_generator.generate()

    def _load_trainer(self):
        if optimizer == 'Adam':
            optim = tf.train.AdamOptimizer(learning_rate=self.learn_rate)
        else:
            optim = tf.train.GradientDescentOptimizer(learning_rate=self.learn_rate)
        ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
        self.loss = tf.reduce_mean(ssd_loss.compute_loss(self.input_y, self.output))
        self.train_op = optim.minimize(self.loss)

    def _load_summary(self, session, log_path):
        loss_summary = tf.summary.scalar('loss', self.loss)
        summary_merged = tf.summary.merge([loss_summary])
        summary_writer = tf.summary.FileWriter(os.path.join(log_path, 'summaries'),
                                               graph=session.graph)
        return summary_merged, summary_writer

    def train(self):
        dt = datetime.now()
        dt_str = dt.strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(output_root, dt_str)

        checkpoint_dir = os.path.join(output_dir, 'ckpt')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
        saver = tf.train.Saver(tf.global_variables())

        current_step = 0
        max_train_steps = max_epochs * self.data_size // batch_size
        if max_train_steps < max_steps:
            max_train_steps = max_steps

        print('start training...')
        with tf.Session() as sess:
            # summary_merged, summary_writer = self._load_summary(sess, output_dir)

            sess.run(tf.global_variables_initializer())
            for image, labels in self.data_generator:
                gt_labels = self.input_encoder.encode(labels)

                epoch = current_step // self.data_size
                lr = lr_schedule(epoch)
                feed_dict = {
                    self.input_x: image,
                    self.input_y: gt_labels,
                    self.learn_rate: lr,
                }
                # _, loss_value, train_summary = sess.run([self.train_op, self.loss, summary_merged],
                #                                         feed_dict=feed_dict)
                _, loss_value = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
                # summary_writer.add_summary(train_summary, current_step)

                time_str = datetime.now().isoformat()
                print("{}: step: {}, loss: {:g}".format(time_str, current_step, loss_value))

                current_step += 1

                if current_step % checkpoint_steps == 0 and current_step > 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

                if current_step > max_train_steps:
                    break


def build_dataset():
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


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('-t', '--train', action='store_true', default=True)
    parse.add_argument('-d', '--data', action='store_true')
    args = parse.parse_args()

    if args.data:
        build_dataset()
    if args.train:
        trainer = SSD300Trainer()
        trainer.train()
