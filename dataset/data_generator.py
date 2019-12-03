import cv2
import numpy as np
from dataset.dataset_pascalvoc import DatasetPascalvoc
from dataset.transform import ImageTransformer, Resize, Normalizer


class DataGenerator(object):
    def __init__(self, dataset, transform, batch_size, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.transform = transform
        self.shuffle = shuffle

    def generate(self):
        """
        generate batch of data samples
        """
        if self.dataset.size == 0:
            raise ValueError('empty dataset')

        current = 0
        while True:
            batch_x = []  # NHWC
            batch_y = []  # SSD label input type

            if current > self.dataset.size:
                current = 0
                if self.shuffle:
                    self.dataset.shuffle()

            batch_indices = [current + i for i in range(self.batch_size)]
            current += self.batch_size

            for idx in batch_indices:
                filename, labels = self.dataset[idx]

                # parse data and do transformation
                img, labels = self.parse_sample(filename, labels)
                img = np.expand_dims(img, axis=0)
                batch_x.append(img)
                batch_y.append(labels)

            batch_x = np.concatenate(batch_x, axis=0)
            yield batch_x, batch_y

    def parse_sample(self, filename, labels):
        # label coordinates are normalized into [0, 1]
        image = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), -1)
        height = image.shape[0]
        width = image.shape[1]
        image = self.transform.apply(image)
        for lbl in labels:
            lbl[1] /= width
            lbl[2] /= height
            lbl[3] /= width
            lbl[4] /= height
        labels = np.array(labels)
        return image, labels


def _get_voc_generator(dataset_dirs, transform, batch_size, phase='train', class_names=None):
    assert phase in ['train', 'test']
    dataset = DatasetPascalvoc(class_names)
    dataset.build(dataset_dirs, phase=phase)
    if phase == 'train':
        dataset.shuffle()
    data_generator = DataGenerator(dataset, transform=transform, batch_size=batch_size, shuffle=True)
    return data_generator


def get_ssd_input_transform():
    return ImageTransformer([Resize((300, 300)), Normalizer()])


def test():
    voc2007_dir = r'D:\data\VOCdevkit\VOC2007'
    transformer = get_ssd_input_transform()
    data_generator = _get_voc_generator([voc2007_dir], transformer, batch_size=2)

    for batchx, batchy in data_generator.generate():
        print(batchx.shape)
        print(batchy)
        break


if __name__ == '__main__':
    test()
