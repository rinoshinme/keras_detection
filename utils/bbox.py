import numpy as np


def center2minmax(box):
    if len(box.shape) == 1:
        xmin = box[0] - box[2] / 2
        ymin = box[1] - box[3] / 2
        xmax = box[0] + box[2] / 2
        ymax = box[1] + box[3] / 2
        return np.array([xmin, ymin, xmax, ymax])
    else:
        cxcy = box[..., 0:2]
        wh = box[..., 2:4]
        mins = cxcy - wh / 2
        maxs = cxcy + wh / 2
        return np.concatenate([mins, maxs], axis=-1)


def minmax2center(box):
    if len(box.shape) == 1:
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        w = box[2] - box[0]
        h = box[3] - box[1]
        return np.array([cx, cy, w, h])
    else:
        mins = box[..., 0:2]
        maxs = box[..., 2:4]
        centers = (mins + maxs) / 2
        sizes = maxs - mins
        return np.concatenate([centers, sizes], axis=-1)


def iou(boxes1, boxes2):
    """
    boxes are of format (xmin, ymin, xmax, ymax)
    """
    if len(boxes1.shape) == 1:
        boxes1 = np.expand_dims(boxes1, axis=0)
    if len(boxes2.shape) == 1:
        boxes2 = np.expand_dims(boxes2, axis=0)
    n1 = boxes1.shape[0]
    n2 = boxes2.shape[0]
    boxes1 = np.expand_dims(boxes1, axis=1)
    boxes1 = np.tile(boxes1, (1, n2, 1))
    boxes2 = np.expand_dims(boxes2, axis=0)
    boxes2 = np.tile(boxes2, (n1, 1, 1))

    xy_min = np.maximum(boxes1[..., 0:2], boxes2[..., 0:2])
    xy_max = np.minimum(boxes1[..., 2:4], boxes2[..., 2:4])
    interp_wh = xy_max - xy_min
    interp_wh[interp_wh < 0] = 0
    interp_area = interp_wh[..., 0] * interp_wh[..., 1]

    wh1 = boxes1[..., 2:4] - boxes1[..., 0:2]
    wh1[wh1 < 0] = 0
    wh2 = boxes2[..., 2:4] - boxes2[..., 0:2]
    wh2[wh2 < 0] = 0
    area1 = wh1[..., 0] * wh1[..., 1]
    area2 = wh2[..., 0] * wh2[..., 1]

    return np.divide(interp_area, area1 + area2 - interp_area)


def nms(boxes, scores, thresh):
    # assume all boxes are for same category
    # boxes: (N, 4)
    # scores: (N, )
    xmin = boxes[:, 0]
    ymin = boxes[:, 1]
    xmax = boxes[:, 2]
    ymax = boxes[:, 3]
    areas = (xmax - xmin + 1) * (ymax - ymin + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(xmin[i], xmin[order[1:]])
        yy1 = np.maximum(ymin[i], ymin[order[1:]])
        xx2 = np.minimum(xmax[i], xmax[order[1:]])
        yy2 = np.minimum(ymax[i], ymax[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ious = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ious <= thresh)[0]
        order = order[inds + 1]
    return keep


if __name__ == '__main__':
    bb1 = np.array([[1, 1, 2, 2], [0, 0, 1, 1]])
    bb2 = minmax2center(bb1)
    iou_values = iou(bb1, bb2)

    print(iou_values)
