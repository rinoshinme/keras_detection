import numpy as np


def match_ground_truth(weight_matrix):
    # first axis: ground truth
    # second axis: anchor boxes
    weight_matrix = np.copy(weight_matrix)
    num_gt_boxes = weight_matrix.shape[0]
    all_gt_indices = list(range(num_gt_boxes))

    matches = np.zeros(num_gt_boxes, dtype=np.int)
    # do this cause single anchor box cannot match 2 gt boxes
    for _ in range(num_gt_boxes):
        anchor_indices = np.argmax(weight_matrix, axis=1)
        overlaps = weight_matrix[all_gt_indices, anchor_indices]
        ground_truth_index = np.argmax(overlaps)
        anchor_index = anchor_indices[ground_truth_index]
        matches[ground_truth_index] = anchor_index

        weight_matrix[ground_truth_index, :] = 0
        weight_matrix[:, anchor_index] = 0
    return matches


def match_anchor_boxes(weight_matrix, threshold):
    # weight_matrix: (num_gt, num_anchors)
    num_anchor_boxes = weight_matrix.shape[1]
    all_anchor_indices = list(range(num_anchor_boxes))

    # find best ground truth match for each anchor box
    ground_truth_indices = np.argmax(weight_matrix, axis=0)
    overlaps = weight_matrix[ground_truth_indices, all_anchor_indices]

    anchor_indices_positive = np.nonzero(overlaps >= threshold)[0]
    gt_indices_positive = ground_truth_indices[anchor_indices_positive]
    return gt_indices_positive, anchor_indices_positive
