"""
SSD300 model definition
"""
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, Reshape, Activation, Concatenate
from keras.regularizers import l2

from layers.l2_normalize import L2Normalization
from layers.anchor_boxes import AnchorBoxes
from layers.ssd_decode import SSDDecodeLayer


class SSD300(object):
    def __init__(self, num_classes, scales, aspect_ratios, two_boxes_for_ar1,
                 steps, offsets, clip_boxes, variances,
                 conf_thresh=0.01, iou_thresh=0.45, top_k=200, nms_max_output_size=200,
                 normalize_coords=True, phase='train'):
        self.image_height = 300
        self.image_width = 300
        self.channels = 3
        self.l2_regularization = 0.0005
        self.num_classes = num_classes + 1
        assert phase in ['train', 'inference']
        self.phase = phase

        # anchor parameters
        self.scales = scales
        self.aspect_ratios = aspect_ratios
        self.two_boxes_for_ar1 = two_boxes_for_ar1
        self.num_boxes = []
        for ar in self.aspect_ratios:
            if (1 in ar) and self.two_boxes_for_ar1:
                self.num_boxes.append(1 + len(ar))
            else:
                self.num_boxes.append(len(ar))

        self.steps = steps
        self.offsets = offsets
        self.clip_boxes = clip_boxes
        self.variances = variances

        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.top_k = top_k
        self.nms_max_output_size = nms_max_output_size
        self.normalize_coords = normalize_coords

        # build network
        self.end_points = self._build_backbone()
        self._build_head()
        self._build_target()

    def _build_backbone(self):
        # the preprocessed image tensor
        self.input = Input(shape=(self.image_height, self.image_width, self.channels), name='input_x')

        # block1
        conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(self.l2_regularization), name='conv1_1')(self.input)
        conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(self.l2_regularization), name='conv1_2')(conv1_1)
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1')(conv1_2)

        # block2
        conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(self.l2_regularization), name='conv2_1')(pool1)
        conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(self.l2_regularization), name='conv2_2')(conv2_1)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool2')(conv2_2)

        # block3
        conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(self.l2_regularization), name='conv3_1')(pool2)
        conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(self.l2_regularization), name='conv3_2')(conv3_1)
        conv3_3 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(self.l2_regularization), name='conv3_3')(conv3_2)
        pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool3')(conv3_3)

        # block4
        conv4_1 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(self.l2_regularization), name='conv4_1')(pool3)
        conv4_2 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(self.l2_regularization), name='conv4_2')(conv4_1)
        conv4_3 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(self.l2_regularization), name='conv4_3')(conv4_2)
        pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool4')(conv4_3)

        # block5
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

        # generate basic prediction
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
            'offset': self.offsets[index],
            'clip_box': self.clip_boxes,
            'variances': self.variances,
            'normalize_coords': True,
        }

        prior_box = AnchorBoxes(anchor_params)(feature)

        # reshape
        feature_map_shape = feature.get_shape().as_list()
        feat_size = feature_map_shape[1] * feature_map_shape[2]

        mbox_conf_reshape = Reshape((feat_size * nbox, self.num_classes),
                                    name='mbox_conf_reshape_{}'.format(index))(mbox_conf)
        mbox_loc_reshape = Reshape((feat_size * nbox, 4), name='mbox_loc_reshape_{}'.format(index))(mbox_loc)
        prior_box_reshape = Reshape((feat_size * nbox, 8), name='prior_box_reshape_{}'.format(index))(prior_box)

        return mbox_conf_reshape, mbox_loc_reshape, prior_box_reshape

    def _build_head(self):
        mbox_conf1, mbox_loc1, prior_box1 = self._single_head(0)
        mbox_conf2, mbox_loc2, prior_box2 = self._single_head(1)
        mbox_conf3, mbox_loc3, prior_box3 = self._single_head(2)
        mbox_conf4, mbox_loc4, prior_box4 = self._single_head(3)
        mbox_conf5, mbox_loc5, prior_box5 = self._single_head(4)
        mbox_conf6, mbox_loc6, prior_box6 = self._single_head(5)

        # concat in num_box axis
        self.mbox_conf = Concatenate(axis=1, name='mbox_conf')([mbox_conf1, mbox_conf2, mbox_conf3,
                                                                mbox_conf4, mbox_conf5, mbox_conf6])
        self.mbox_loc = Concatenate(axis=1, name='mbox_loc')([mbox_loc1, mbox_loc2, mbox_loc3,
                                                              mbox_loc4, mbox_loc5, mbox_loc6])
        self.mbox_priorbox = Concatenate(axis=1, name='prior_box')([prior_box1, prior_box2, prior_box3,
                                                                    prior_box4, prior_box5, prior_box6])

    def _build_target(self):
        mbox_conf_softmax = Activation('softmax', name='mbox_conf_softmax')(self.mbox_conf)
        predictions = Concatenate(axis=2, name='final_predictions')([mbox_conf_softmax,
                                                                     self.mbox_loc, self.mbox_priorbox])

        if self.phase == 'train':
            self.output = predictions
            self.model = Model(inputs=self.input, outputs=self.output)
        elif self.phase == 'inference':
            # append decode output layer
            decoded = SSDDecodeLayer(self.image_height, self.image_width, self.conf_thresh, self.iou_thresh,
                                     self.top_k, self.nms_max_output_size, self.normalize_coords)(predictions)
            self.output = decoded
            self.model = Model(inputs=self.input, outputs=self.output)
        else:
            raise ValueError('unsupported phase value')
