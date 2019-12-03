"""
input encoding and output decoding for ssd object detection
"""
import numpy as np
from utils.bbox import nms


class SSDDecoder(object):
    def __init__(self, conf_thresh, iou_thresh, top_k):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.top_k = top_k

    def decode_output(self, predictions):
        # prediction: network output for single image, shape: [nbox, nclass+4+8]
        pred_conf = predictions[:, 0:-12]
        pred_loc = predictions[:, -12:-8]
        anchors = predictions[:, -8:]
        num_classes = pred_conf.shape[1]

        all_boxes = []
        all_scores = []
        for i in range(1, num_classes):
            pred_conf_class = pred_conf[:, i]
            positive_mask = pred_conf_class > self.conf_thresh

            pred_conf_positive = pred_conf[positive_mask]
            pred_loc_positive = pred_loc[positive_mask]
            anchors_positive = anchors[positive_mask]

            pred_loc_xy = pred_loc_positive[..., 0:2]
            pred_loc_wh = pred_loc_positive[..., 2:4]
            anchors_xy = anchors_positive[..., 0:2]
            anchors_wh = anchors_positive[..., 2:4]
            anchors_var_xy = anchors_positive[..., 4:6]
            anchors_var_wh = anchors_positive[..., 6:8]

            xy = pred_loc_xy * anchors_var_xy * anchors_wh + anchors_xy
            wh = np.exp(pred_loc_wh * anchors_var_wh) * anchors_wh

            pred_boxes_positive = np.concatenate([xy, wh], axis=-1)

            # do nms
            keep_index = nms(pred_boxes_positive, pred_conf_positive, self.iou_thresh)
            all_boxes.append(pred_boxes_positive[keep_index])
            all_scores.append(pred_conf_positive[keep_index])

        all_boxes = np.concatenate(all_boxes, axis=-1)
        all_confs = np.concatenate(all_scores, axis=-1)

        # top k filtering
        total_indices = np.argsort(all_scores)
        if total_indices.size > self.top_k:
            total_indices = total_indices[:self.top_k]

        result_boxes = all_boxes[total_indices]
        result_confs = all_confs[total_indices]

        return result_boxes, result_confs


if __name__ == '__main__':
    pass
