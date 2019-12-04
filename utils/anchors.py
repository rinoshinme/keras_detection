import numpy as np
from utils.bbox import minmax2center, center2minmax


def generate_anchors(image_height, image_width, feature_map_height, feature_map_width,
                     scale, next_scale, step, aspect_ratios, two_boxes_for_ar1=True,
                     offset=0.5, normalize_coord=True, clip_boundary=True):
    """
    result: np.ndarray of shape (feat_height, feat_width, nbox, 4) with (cx, cy, w, h) box coordinates
    """
    # generate anchors for each pixel position
    size = min(image_height, image_width)
    wh_list = []
    for ar in aspect_ratios:
        if ar == 1:
            box_width = scale * size
            box_height = scale * size
            wh_list.append((box_width, box_height))
            if two_boxes_for_ar1:
                s = np.sqrt(scale * next_scale)
                box_width = s * size
                box_height = s * size
                wh_list.append((box_width, box_height))
        else:
            box_width = scale * size * np.sqrt(ar)
            box_height = box_width / ar
            wh_list.append((box_width, box_height))
    nboxes = len(wh_list)
    wh_list = np.array(wh_list)

    # sweep anchors over entire feature map
    cx = [(offset + i) * step for i in range(feature_map_width)]
    cy = [(offset + i) * step for i in range(feature_map_height)]
    cx_grid, cy_grid = np.meshgrid(cx, cy)

    cx_grid = np.expand_dims(cx_grid, axis=-1)
    cy_grid = np.expand_dims(cy_grid, axis=-1)

    box_tensor = np.zeros((feature_map_height, feature_map_width, nboxes, 4))
    box_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, nboxes))
    box_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, nboxes))
    box_tensor[:, :, :, 2] = wh_list[:, 0]
    box_tensor[:, :, :, 3] = wh_list[:, 1]

    if clip_boundary:
        minmax = center2minmax(box_tensor)

        xmins = minmax[..., 0]
        xmins[xmins < 0] = 0
        ymins = minmax[..., 1]
        ymins[ymins < 0] = 0
        xmaxs = minmax[..., 2]
        xmaxs[xmaxs > image_width - 1] = image_width - 1
        ymaxs = minmax[..., 3]
        ymaxs[ymaxs > image_height - 1] = image_height - 1

        box_tensor = minmax2center(minmax)

    if normalize_coord:
        box_tensor[:, :, :, 0] /= image_width
        box_tensor[:, :, :, 1] /= image_height
        box_tensor[:, :, :, 2] /= image_width
        box_tensor[:, :, :, 3] /= image_height

    return box_tensor


def anchor_clustering(boxes):
    pass


def test_generate_anchors():
    image_height = 300
    image_width = 300

    feat_height = 19
    feat_width = 19
    scale = 0.1
    next_scale = 0.3
    step = 16
    aspect_ratios = [1, 2.0, 0.5, 3.0, 1.0/3.0]
    box_tensor = generate_anchors(image_height, image_width, feat_height, feat_width,
                                  scale, next_scale, step, aspect_ratios,
                                  normalize_coord=False, clip_boundary=True)
    print(box_tensor)
    print(box_tensor.shape)


if __name__ == '__main__':
    test_generate_anchors()
