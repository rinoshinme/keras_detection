"""
encode ground truth labels into ssd model input format
"""
import numpy as np
from utils.anchors import generate_anchors
from utils.bbox import iou
from utils.box_matching import match_ground_truth, match_anchor_boxes


class SSDEncoder(object):
    def __init__(self, image_height, image_width, num_classes, feature_map_sizes,
                 scales, aspect_ratios, two_boxes_for_ar1, steps,
                 offsets, clip_boxes, variances, iou_thresh_high, iou_thresh_low, background_id=0):
        self.image_height = image_height
        self.image_width = image_width
        self.num_classes = num_classes + 1
        self.feature_map_sizes = feature_map_sizes
        self.scales = scales
        self.aspect_ratios = aspect_ratios
        self.two_boxes_for_ar1 = two_boxes_for_ar1
        self.steps = steps
        self.offsets = offsets
        self.clip_boxes = clip_boxes
        self.variances = variances
        self.iou_thresh_high = iou_thresh_high
        self.iouthresh_low = iou_thresh_low
        self.background_id = background_id

        # generate anchors
        self.boxes_list = []
        for i in range(len(self.feature_map_sizes)):
            box_tensor = generate_anchors(self.image_height, self.image_width,
                                          self.feature_map_sizes[i][0], self.feature_map_sizes[i][1],
                                          self.scales[i], self.scales[i+1], self.steps[i], self.aspect_ratios[i],
                                          two_boxes_for_ar1=self.two_boxes_for_ar1, offset=self.offsets[i],
                                          normalize_coord=True, clip_boundary=True)
            # each is of shape (feat_size, feat_size, nbox, 4) in (xmin, ymin, xmax, ymax) form
            self.boxes_list.append(box_tensor)

    def encode(self, gt_labels):
        all_encoded = []
        for gt_label in gt_labels:
            encoded = self.encode_single(gt_label)
            encoded = np.expand_dims(encoded, axis=0)
            all_encoded.append(encoded)

        return np.concatenate(all_encoded, axis=0)

    def encode_single(self, gt_labels):
        # gt_labels: (k, 5) if (class_id, xmin, ymin, xmax, ymax)

        class_id = 0
        xmin, ymin, xmax, ymax = [1, 2, 3, 4]

        classes_encoded, offset_encoded, anchors_encoded, variance_encoded = self.get_encoding_template_single()
        classes_encoded[:, self.background_id] = 1  # set background as default

        class_vectors = np.eye(self.num_classes)
        gt_classes_one_hot = class_vectors[gt_labels[:, class_id].astype(np.int)]
        gt_boxes = gt_labels[:, [xmin, ymin, xmax, ymax]]

        iou_mat = iou(gt_boxes, anchors_encoded)

        # match from ground truths
        gt_match = match_ground_truth(iou_mat)
        classes_encoded[gt_match] = gt_classes_one_hot
        offset_encoded[gt_match] = gt_boxes

        # match from anchors
        gt_indices, anchor_indices = match_anchor_boxes(iou_mat, self.iou_thresh_high)
        classes_encoded[anchor_indices] = gt_classes_one_hot[gt_indices]
        offset_encoded[anchor_indices] = gt_boxes[gt_indices]

        # calculate offset from anchors to boxes
        offset_encoded -= anchors_encoded
        offset_encoded[:, [0, 2]] /= np.expand_dims(anchors_encoded[:, 2] - anchors_encoded[:, 0], axis=-1)
        offset_encoded[:, [1, 3]] /= np.expand_dims(anchors_encoded[:, 3] - anchors_encoded[:, 1], axis=-1)
        offset_encoded /= variance_encoded

        return np.concatenate([classes_encoded, offset_encoded, anchors_encoded, variance_encoded], axis=-1)

    def get_encoding_template_single(self):
        boxes_batch = []
        for box in self.boxes_list:
            box = np.reshape(box, (-1, 4))
            boxes_batch.append(box)
        anchors_tensor = np.concatenate(boxes_batch, axis=0)

        num_boxes = anchors_tensor.shape[0]
        classes_tensor = np.zeros((num_boxes, self.num_classes))
        variance_tensor = np.zeros_like(anchors_tensor)
        variance_tensor += self.variances
        offset_tensor = np.zeros_like(anchors_tensor)
        return classes_tensor, offset_tensor, anchors_tensor, variance_tensor

    def get_encoding_template(self, batch_size):
        boxes_batch = []
        for boxes in self.boxes_list:
            boxes = np.expand_dims(boxes, axis=0)
            boxes = np.tile(boxes, (batch_size, 1, 1, 1, 1))
            boxes = np.reshape(boxes, (batch_size, -1, 4))
            boxes_batch.append(boxes)
        boxes_tensor = np.concatenate(boxes_batch, axis=1)

        classes_tensor = np.zeros((batch_size, boxes_tensor.shape[1], self.num_classes))

        variance_tensor = np.zeros_like(boxes_tensor)
        variance_tensor += self.variances

        offset_tensor = np.zeros_like(boxes_tensor)

        # boxes here mean anchor boxes
        return classes_tensor, offset_tensor, boxes_tensor, variance_tensor

        # template = np.concatenate((classes_tensor, boxes_tensor, boxes_tensor, variance_tensor), axis=2)
        # return template


def test_encoding():
    image_height = 300
    image_width = 300
    feature_map_sizes = ((38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1))
    scales = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05]
    aspect_ratios = [[1, 2.0, 0.5],
                     [1, 2.0, 0.5, 3.0, 1.0 / 3.0],
                     [1, 2.0, 0.5, 3.0, 1.0 / 3.0],
                     [1, 2.0, 0.5, 3.0, 1.0 / 3.0],
                     [1, 2.0, 0.5],
                     [1, 2.0, 0.5]]
    two_boxes_for_ar1 = True
    steps = [8, 16, 32, 64, 100, 300]
    offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    clip_boxes = True
    variances = [0.1, 0.1, 0.2, 0.2]
    iou_thresh_high = 0.5
    iou_thresh_low = 0.3

    encoder = SSDEncoder(image_height, image_width, 20, feature_map_sizes,
                         scales, aspect_ratios, two_boxes_for_ar1, steps,
                         offsets, clip_boxes, variances, iou_thresh_high, iou_thresh_low)
    temp = encoder.get_encoding_template(batch_size=8)
    print(len(temp))


if __name__ == '__main__':
    test_encoding()
