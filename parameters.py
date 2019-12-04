# anchor params should be pre-computed
anchor_params_ssd300 = {
    'scales': [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05],
    'feature_map_sizes': [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
    'aspect_ratios': [[1, 2.0, 0.5],
                      [1, 2.0, 0.5, 3.0, 1.0 / 3.0],
                      [1, 2.0, 0.5, 3.0, 1.0 / 3.0],
                      [1, 2.0, 0.5, 3.0, 1.0 / 3.0],
                      [1, 2.0, 0.5],
                      [1, 2.0, 0.5]],
    'two_boxes_for_ar1': True,
    # 'num_boxes': [4, 6, 6, 6, 4, 4],
    'steps': [8, 16, 32, 64, 100, 300],
    'offset': 0.5,
    'clip_boxes': True,
    'variances': [0.1, 0.1, 0.2, 0.2],
}

anchor_params_ssd512 = {
}

encode_params = {

}

decode_params = {
    'confidence_thresh': 0.01,
    'iou_threshold': 0.45,
    'top_k': 200,
    'normalize_coords': True,
    'nms_max_output_size': 40,
}

train_config = {
    'anchor_params': anchor_params_ssd300,
    'decode_params': decode_params,
    'encode_params': encode_params,

    # model parameters
    'model_name': 'ssd300',
    'image_height': 300,
    'image_width': 300,
    'class_names': ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                    'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                    'sofa', 'train', 'tvmonitor'),
    'num_classes': 20,
    'pretrained_weights': None,

    'conf_thresh': 0.5,
    'iou_thresh_high': 0.5,
    'iou_thresh_low': 0.3,

    # dataset parameters
    'dataset_dirs': ['D:/data/VOCdevkit/VOC2007', 'D:/data/VOCdevkit/VOC2012'],
    'dataset_pickle': './data/voc0712.pkl',
    'input_transform': None,
    'batch_size': 1,

    # training parameters
    'optimizer': 'Adam',
    'learn_rate': 1.0e-4,
    'train_epochs': 100,
    'train_steps': 20000,
    'display_step': 10,
    'checkpoint_step': 2000,
}
