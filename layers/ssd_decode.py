from keras.engine.topology import InputSpec
from keras.engine.topology import Layer
import tensorflow as tf


class SSDDecodeLayer(Layer):
    """
    this class is used for generating results using single inference, without post-processing
    """
    def __init__(self, image_height, image_width, conf_thresh, iou_thresh, top_k,
                 nms_max_output_size, normalize_coords):
        self.image_height = image_height
        self.image_width = image_width
        self.confidence_thresh = conf_thresh
        self.iou_threshold = iou_thresh
        self.top_k = top_k
        self.nms_max_output_size = nms_max_output_size
        self.normalize_coords = normalize_coords

        # convert to tensorflow constants
        self.tf_confidence_thresh = tf.constant(self.confidence_thresh, name='confidence_thresh')
        self.tf_iou_threshold = tf.constant(self.iou_threshold, name='iou_threshold')
        self.tf_top_k = tf.constant(self.top_k, name='top_k')
        self.tf_normalize_coords = tf.constant(self.normalize_coords, name='normalize_coords')
        self.tf_image_height = tf.constant(self.image_height, dtype=tf.float32, name='image_height')
        self.tf_image_width = tf.constant(self.image_width, dtype=tf.float32, name='image_width')
        self.tf_nms_max_output_size = tf.constant(self.nms_max_output_size, name='nms_max_output_size')

        super(SSDDecodeLayer, self).__init__()

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        super(SSDDecodeLayer, self).build(input_shape)

    def call(self, prediction, mask=None):
        prediction_shape = prediction.get_shape().as_list()
        num_classes = prediction_shape[-1] - 12
        # print(num_classes)

        # 1. unpack prediction values
        # prediction -> [num_classes + 4 + 8]
        pred_confidence = prediction[:, :, 0:num_classes]
        # cx, cy, w, h
        pred_loc = prediction[:, :, num_classes:num_classes+4]
        # ax, ay, w, h, 0.1, 0.1, 0.2, 0.2
        pred_anchor = prediction[:, :, num_classes+4:num_classes+8]
        pred_variance = prediction[:, :, num_classes+8:]

        cx = pred_loc[:, :, 0] * pred_variance[:, :, 0] * pred_anchor[:, :, 2] + pred_anchor[:, :, 0]
        cy = pred_loc[:, :, 1] * pred_variance[:, :, 1] * pred_anchor[:, :, 3] + pred_anchor[:, :, 1]
        w = tf.exp(pred_loc[:, :, 2] * pred_variance[:, :, 2]) * pred_anchor[:, :, 2]
        h = tf.exp(pred_loc[:, :, 3] * pred_variance[:, :, 3]) * pred_anchor[:, :, 3]

        # convert centroids to minmax
        xmin = cx - 0.5 * w
        ymin = cy - 0.5 * h
        xmax = cx + 0.5 * w
        ymax = cy + 0.5 * h

        def normalize_coordinates():
            xmin1 = tf.expand_dims(xmin * self.image_width, axis=-1)
            xmax1 = tf.expand_dims(xmax * self.image_width, axis=-1)
            ymin1 = tf.expand_dims(ymin * self.image_height, axis=-1)
            ymax1 = tf.expand_dims(ymax * self.image_height, axis=-1)
            return xmin1, ymin1, xmax1, ymax1

        def non_normalize_coordinates():
            xmin1 = tf.expand_dims(xmin, axis=-1)
            xmax1 = tf.expand_dims(xmax, axis=-1)
            ymin1 = tf.expand_dims(ymin, axis=-1)
            ymax1 = tf.expand_dims(ymax, axis=-1)
            return xmin1, ymin1, xmax1, ymax1

        xmin, ymin, xmax, ymax = tf.cond(self.tf_normalize_coords, normalize_coordinates, non_normalize_coordinates)

        y_pred = tf.concat([pred_confidence, xmin, ymin, xmax, ymax], axis=-1)

        # 2. do thresholding, per-class nms, and top-k filtering
        # batch_size = tf.shape(y_pred)[0]
        # n_boxes = tf.shape(y_pred)[1]
        # class_indices = tf.range(1, num_classes)  # excluding background

        def filter_predictions(batch_item):
            def filter_single_class(index):
                confidences = tf.expand_dims(batch_item[..., index], axis=-1)
                box_coordinates = batch_item[..., -4:]
                class_id = tf.fill(dims=tf.shape(confidences), value=tf.to_float(index))
                single_class = tf.concat([class_id, confidences, box_coordinates], axis=-1)
                # apply confidence threshold
                threshold_met = single_class[:, 1] > self.tf_confidence_thresh
                single_class = tf.boolean_mask(tensor=single_class, mask=threshold_met)

                def perform_nms():
                    scores = single_class[..., 1]
                    xmin_single = tf.expand_dims(single_class[..., -4], axis=-1)
                    ymin_single = tf.expand_dims(single_class[..., -3], axis=-1)
                    xmax_single = tf.expand_dims(single_class[..., -2], axis=-1)
                    ymax_single = tf.expand_dims(single_class[..., -1], axis=-1)
                    boxes = tf.concat([ymin_single, xmin_single, ymax_single, xmax_single], axis=-1)
                    maxima_indices = tf.image.non_max_suppression(boxes=boxes,
                                                                  scores=scores,
                                                                  max_output_size=self.tf_nms_max_output_size,
                                                                  iou_threshold=self.iou_threshold,
                                                                  name='non_maximum_suppresion')
                    maxima = tf.gather(params=single_class,
                                       indices=maxima_indices,
                                       axis=0)
                    return maxima

                def perform_no_nms():
                    return tf.constant(value=0.0, shape=(1, 6))

                single_class_nms = tf.cond(tf.equal(tf.size(single_class), 0), perform_no_nms, perform_nms)
                padded_single_class = tf.pad(tensor=single_class_nms,
                                             paddings=[[0, self.tf_nms_max_output_size - tf.shape(single_class_nms)[0]],
                                                       [0, 0]],
                                             mode='CONSTANT',
                                             constant_values=0.0)
                return padded_single_class

            filtered_single_classes = tf.map_fn(fn=lambda i: filter_single_class(i),
                                                elems=tf.range(1, num_classes),
                                                dtype=tf.float32,
                                                parallel_iterations=128,
                                                back_prop=False,
                                                swap_memory=False,
                                                infer_shape=True,
                                                name='loop_over_classes')
            filtered_predictions = tf.reshape(tensor=filtered_single_classes, shape=(-1, 6))

            # do top-k filtering
            def top_k():
                tk = tf.nn.top_k(filtered_predictions[:, 1], k=self.tf_top_k, sorted=True)
                return tf.gather(params=filtered_predictions,
                                 indices=tk.indices,
                                 axis=0)

            def pad_and_top_k():
                padded_predictions = tf.pad(tensor=filtered_predictions,
                                            paddings=[[0, self.tf_top_k - tf.shape(filtered_predictions)[0]], [0, 0]],
                                            mode='CONSTANT',
                                            constant_values=0.0)
                return tf.gather(params=padded_predictions,
                                 indices=tf.nn.top_k(padded_predictions[:, 1], k=self.tf_top_k, sorted=True).indices,
                                 axis=0)

            top_k_boxes = tf.cond(tf.greater_equal(tf.shape(filtered_predictions)[0], self.tf_top_k), top_k,
                                  pad_and_top_k)
            return top_k_boxes

        # Iterate `filter_predictions()` over all batch items.
        output_tensor = tf.map_fn(fn=lambda x: filter_predictions(x),
                                  elems=y_pred,
                                  dtype=None,
                                  parallel_iterations=128,
                                  back_prop=False,
                                  swap_memory=False,
                                  infer_shape=True,
                                  name='loop_over_batch')

        return output_tensor

    def compute_output_shape(self, input_shape):
        batch_size, n_boxes, last_axis = input_shape
        return batch_size, self.top_k, 6  # Last axis: (class_ID, confidence, 4 box coordinates)

    def get_config(self):
        new_config = {
            'confidence_thresh': self.confidence_thresh,
            'iou_threshold': self.iou_threshold,
            'top_k': self.top_k,
            'nms_max_output_size': self.nms_max_output_size,
            'normalize_coords': self.normalize_coords,
            'img_height': self.image_height,
            'img_width': self.image_width,
        }

        base_config = super(SSDDecodeLayer, self).get_config()

        config = {}
        for k, v in base_config.items():
            config[k] = v
        for k, v in new_config.items():
            config[k] = v
        return config
