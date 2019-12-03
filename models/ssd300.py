"""
SSD300 model definition
"""
from keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, Reshape, Activation, Concatenate
from keras.regularizers import l2

from layers.l2_normalize import L2Normalization
from layers.anchor_boxes import AnchorBoxes
from layers.ssd_decode import SSDDecode


class SSD300(object):
    def __init__(self, num_classes, ssd_params=None, phase='train'):
        self.image_height = 300
        self.image_width = 300
        self.channels = 3
        self.l2_regularization = 0.0005
        self.num_classes = num_classes + 1

        assert phase in ['train', 'inference']
        self.phase = phase

        # preprocessing

        # anchor parameters
        self.scales = ssd_params['scales']
        self.aspect_ratios = ssd_params['aspect_ratios']
        self.two_boxes_for_ar1 = ssd_params['two_boxes_for_ar1']
        self.num_boxes = ssd_params['num_boxes']
        self.steps = ssd_params['steps']
        self.offset = ssd_params['offset']
        self.clip_boxes = ssd_params['clip_boxes']
        self.variances = ssd_params['variances']

        # build network
        self.end_points = self._build_backbone()
        self.mbox_conf, self.mbox_loc, self.mbox_priorbox = self._build_head()
        self._build_target()

    def _build_backbone(self):
        self.x = Input(shape=(self.image_height, self.image_width, self.channels), name='input_x')
        # do optional input preprocessing (mean subtraction, channel swapping)
        # TODO
        # x1 = tf.identity(self.x)

        conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(self.l2_regularization), name='conv1_1')(self.x)
        conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(self.l2_regularization), name='conv1_2')(conv1_1)
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1')(conv1_2)

        conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(self.l2_regularization), name='conv2_1')(pool1)
        conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(self.l2_regularization), name='conv2_2')(conv2_1)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool2')(conv2_2)

        conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(self.l2_regularization), name='conv3_1')(pool2)
        conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(self.l2_regularization), name='conv3_2')(conv3_1)
        conv3_3 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(self.l2_regularization), name='conv3_3')(conv3_2)
        pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool3')(conv3_3)

        conv4_1 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(self.l2_regularization), name='conv4_1')(pool3)
        conv4_2 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(self.l2_regularization), name='conv4_2')(conv4_1)
        conv4_3 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(self.l2_regularization), name='conv4_3')(conv4_2)
        pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool4')(conv4_3)

        conv5_1 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(self.l2_regularization), name='conv5_1')(pool4)
        conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(self.l2_regularization), name='conv5_2')(conv5_1)
        conv5_3 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(self.l2_regularization), name='conv5_3')(conv5_2)
        pool5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='pool5')(conv5_3)

        fc6 = Conv2D(1024, (3, 3), dilation_rate=(6, 6), activation='relu', padding='same',
                     kernel_initializer='he_normal', kernel_regularizer=l2(self.l2_regularization), name='fc6')(pool5)

        fc7 = Conv2D(1024, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(self.l2_regularization), name='fc7')(fc6)

        conv6_1 = Conv2D(256, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(self.l2_regularization), name='conv6_1')(fc7)
        conv6_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv6_padding')(conv6_1)
        conv6_2 = Conv2D(512, (3, 3), strides=(2, 2), activation='relu', padding='valid',
                         kernel_initializer='he_normal', kernel_regularizer=l2(self.l2_regularization),
                         name='conv6_2')(conv6_1)

        conv7_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(self.l2_regularization), name='conv7_1')(conv6_2)
        conv7_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv7_padding')(conv7_1)
        conv7_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='valid',
                         kernel_initializer='he_normal', kernel_regularizer=l2(self.l2_regularization),
                         name='conv7_2')(conv7_1)

        conv8_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(self.l2_regularization), name='conv8_1')(conv7_2)
        conv8_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid',
                         kernel_initializer='he_normal', kernel_regularizer=l2(self.l2_regularization),
                         name='conv8_2')(conv8_1)

        conv9_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(self.l2_regularization), name='conv9_1')(conv8_2)
        conv9_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid',
                         kernel_initializer='he_normal', kernel_regularizer=l2(self.l2_regularization),
                         name='conv9_2')(conv9_1)

        conv4_3_norm = L2Normalization(gamma_init=20, name='conv4_3_norm')(conv4_3)
        end_points = [conv4_3_norm, fc7, conv6_2, conv7_2, conv8_2, conv9_2]
        return end_points

    def _single_head(self, index):
        feature = self.end_points[index]
        nbox = self.num_boxes[index]
        mbox_conf = Conv2D(nbox * self.num_classes, (3, 3), padding='same',
                           kernel_initializer='he_normal', kernel_regularizer=l2(self.l2_regularization),
                           name='mbox_conf_{}'.format(index))(feature)
        mbox_loc = Conv2D(nbox * 4, (3, 3), padding='same',
                          kernel_initializer='he_normal', kernel_regularizer=l2(self.l2_regularization),
                          name='mbox_loc_{}'.format(index))(feature)
        anchor_params = {
            'image_width': self.image_width,
            'image_height': self.image_height,
            'scale': self.scales[index],
            'next_scale': self.scales[index + 1],
            'aspect_ratios': self.aspect_ratios[index],
            'two_boxes_for_ar1': self.two_boxes_for_ar1,
            'step': self.steps[index],
            'offset': self.offset,
            'clip_box': self.clip_boxes,
            'variances': self.variances,
        }

        prior_box = AnchorBoxes(anchor_params)(feature)

        feature_map_shape = feature.get_shape().as_list()
        feat_size = feature_map_shape[1] * feature_map_shape[2]

        mbox_conf_reshape = Reshape((feat_size * nbox, self.num_classes),
                                    name='mbox_conf_reshape_{}'.format(index))(mbox_conf)
        mbox_loc_reshape = Reshape((feat_size * nbox, 4), name='mbox_loc_reshape_{}'.format(index))(mbox_loc)
        prior_box_reshape = Reshape((feat_size * nbox, 8), name='prior_box_reshape_{}'.format(index))(prior_box)

        return mbox_conf_reshape, mbox_loc_reshape, prior_box_reshape

    def _build_head(self):
        # feat1, feat2, feat3, feat4, feat5, feat6 = self.end_points
        mbox_conf1, mbox_loc1, prior_box1 = self._single_head(0)
        mbox_conf2, mbox_loc2, prior_box2 = self._single_head(1)
        mbox_conf3, mbox_loc3, prior_box3 = self._single_head(2)
        mbox_conf4, mbox_loc4, prior_box4 = self._single_head(3)
        mbox_conf5, mbox_loc5, prior_box5 = self._single_head(4)
        mbox_conf6, mbox_loc6, prior_box6 = self._single_head(5)

        mbox_conf = Concatenate(axis=1, name='mbox_conf')([mbox_conf1, mbox_conf2, mbox_conf3,
                                                           mbox_conf4, mbox_conf5, mbox_conf6])
        mbox_loc = Concatenate(axis=1, name='mbox_loc')([mbox_loc1, mbox_loc2, mbox_loc3,
                                                         mbox_loc4, mbox_loc5, mbox_loc6])
        mbox_priorbox = Concatenate(axis=1, name='prior_box')([prior_box1, prior_box2, prior_box3,
                                                               prior_box4, prior_box5, prior_box6])
        return mbox_conf, mbox_loc, mbox_priorbox

    def _build_target(self):
        mbox_conf_softmax = Activation('softmax', name='mbox_conf_softmax')(self.mbox_conf)
        predictions = Concatenate(axis=2, name='predictions')([mbox_conf_softmax, self.mbox_loc, self.mbox_priorbox])

        if self.phase == 'train':
            self.input = self.x
            self.output = predictions
            print(predictions.get_shape().as_list())
            # self.model = Model(inputs=self.x, outputs=predictions)
        else:
            decoded = SSDDecode(self.image_height, self.image_width, self.num_classes)(predictions)
            self.input = self.x
            self.output = decoded
