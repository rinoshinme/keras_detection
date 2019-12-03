from keras.engine.topology import InputSpec
from keras.engine.topology import Layer
import tensorflow as tf


class SSDDecode(Layer):
    """
    this class is used for generating results using single inference, without post-processing
    """
    def __init__(self, image_height, image_width, num_classes):
        self.confidence_thresh = 0.5
        self.iou_thresh = 0.45
        self.top_k = 200
        self.nms_output_num = 50
        self.image_height = image_height
        self.image_width = image_width
        self.normalize_coord = True

        self.num_classes = num_classes  # background included

        super(SSDDecode, self).__init__()

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        super(SSDDecode, self).build(input_shape)

    def call(self, prediction, mask=None):
        # prediction -> [num_classes + 4 + 8]
        pred_confidence = prediction[:, :, 0:self.num_classes]
        # cx, cy, w, h
        pred_loc = prediction[:, :, self.num_classes:self.num_classes+4]
        # ax, ay, w, h, 0.1, 0.1, 0.2, 0.2
        pred_anchor = prediction[:, :, self.num_classes+4:]

        cx = pred_loc[:, :, 0] * pred_anchor[:, :, 2] * pred_anchor[:, :, 4] + pred_anchor[:, :, 0]
        cy = pred_loc[:, :, 1] * pred_anchor[:, :, 3] * pred_anchor[:, :, 5] + pred_anchor[:, :, 1]
        w = tf.exp(pred_loc[:, :, 2] * pred_anchor[:, :, 6]) * pred_anchor[:, :, 2]
        h = tf.exp(pred_loc[:, :, 3] * pred_anchor[:, :, 7]) * pred_anchor[:, :, 3]

        xmin = cx - 0.5 * w
        xmax = cx + 0.5 * w
        ymin = cy - 0.5 * h
        ymax = cy + 0.5 * h

        if self.normalize_coord:
            xmin1 = tf.expand_dims(xmin * self.image_width, axis=-1)
            xmax1 = tf.expand_dims(xmax * self.image_width, axis=-1)
            ymin1 = tf.expand_dims(ymin * self.image_height, axis=-1)
            ymax1 = tf.expand_dims(ymax * self.image_height, axis=-1)
        else:
            xmin1 = tf.expand_dims(xmin, axis=-1)
            xmax1 = tf.expand_dims(xmax, axis=-1)
            ymin1 = tf.expand_dims(ymin, axis=-1)
            ymax1 = tf.expand_dims(ymax, axis=-1)

        y_pred = tf.concat([pred_confidence, xmin1, ymin1, xmax1, ymax1], axis=-1)

        # do box filtering
        # confidence thresholding, per-class nms and top-k filtering

    @staticmethod
    def filter_all():
        def filter_batch_item(batch_item):
            tf.map_fn()

    def filter_single_class(self, batch_item, index):
        # do threshold filtering
        confidences = batch_item[:, index]
        box_coordinates = batch_item[:, -4:]
        class_id = tf.fill(dims=tf.shape(confidences), value=tf.to_float(index))
        sample = tf.concat([class_id, confidences, box_coordinates], axis=-1)
        positive = confidences[:] > self.confidence_thresh

        positive_samples = tf.boolean_mask(tensor=sample,
                                           mask=positive)

        def perform_nms():
            positive_coordinates = positive_samples[:, -4:]
            positive_confidences = positive_samples[:, 1]

            # do nms filtering
            maxima_indices = tf.image.non_max_suppression(boxes=positive_coordinates,
                                                          scores=positive_confidences,
                                                          max_output_size=self.nms_output_num,
                                                          iou_threshold=self.iou_thresh,
                                                          name='non_maximum_suppression')

            maxima = tf.gather(params=positive_samples, indices=maxima_indices, axis=0)
            return maxima

        def no_confident_prediction():
            return tf.constant(value=0.0, shape=(1, 6))

        output = tf.cond(tf.equal(tf.size(positive_samples), 0), no_confident_prediction, perform_nms)

        padded_output = tf.pad(tensor=output,
                               paddings=[[0, self.nms_output_num - tf.shape(output)[0]],
                                         [0, 0]],
                               mode='CONSTANT',
                               constant_values=0.0)
        return padded_output

    def compute_output_shape(self, input_shape):
        batch_size, n_boxes, last_axis = input_shape
        return batch_size, self.top_k, 6  # Last axis: (class_ID, confidence, 4 box coordinates)

    # def get_config(self):
    #     config = {
    #         'confidence_thresh': self.confidence_thresh,
    #         'iou_threshold': self.iou_threshold,
    #         'top_k': self.top_k,
    #         'nms_max_output_size': self.nms_max_output_size,
    #         'coords': self.coords,
    #         'normalize_coords': self.normalize_coords,
    #         'img_height': self.img_height,
    #         'img_width': self.img_width,
    #     }
    #     base_config = super(DecodeDetections, self).get_config()
    #     return dict(list(base_config.items()) + list(config.items()))
