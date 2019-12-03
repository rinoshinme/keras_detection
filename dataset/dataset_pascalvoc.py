import os
import random
from bs4 import BeautifulSoup
import h5py
try:
    import cPickle as pickle
except ImportError:
    import pickle

PASCAL_VOC_CLASS_NAMES = ('aeroplane', 'bicycle', 'bird', 'boat',
                          'bottle', 'bus', 'car', 'cat',
                          'chair', 'cow', 'diningtable', 'dog',
                          'horse', 'motorbike', 'person', 'pottedplant',
                          'sheep', 'sofa', 'train', 'tvmonitor')


class DatasetPascalvoc(object):
    def __init__(self, class_names=PASCAL_VOC_CLASS_NAMES):
        if class_names is None:
            self.class_names = PASCAL_VOC_CLASS_NAMES
        else:
            self.class_names = class_names
        self.filenames = []
        self.labels = []
        self.size = 0

    def build(self, dataset_dirs, phase):
        # generate
        self.filenames = []
        self.labels = []
        for dataset_dir in dataset_dirs:
            if dataset_dir is None:
                continue
            image_dir = os.path.join(dataset_dir, 'JPEGImages')
            annotation_dir = os.path.join(dataset_dir, 'Annotations')
            if phase == 'train':
                imageset = 'trainval.txt'
            else:
                imageset = 'test.txt'
            imageset_path = os.path.join(dataset_dir, 'ImageSets', 'Main', imageset)
            filenames, labels = self._parse(image_dir, annotation_dir, imageset_path)
            self.filenames.extend(filenames)
            self.labels.extend(labels)
        self.size = len(self.filenames)

    def _parse(self, image_dir, annotation_dir, imageset_filename):
        with open(imageset_filename, 'r') as f:
            image_ids = [line.strip() for line in f]

        image_paths = []
        labels = []
        for idx in image_ids:
            image_path = os.path.join(image_dir, '{}.jpg'.format(idx))
            image_paths.append(image_path)
            annotation_path = os.path.join(annotation_dir, '{}.xml'.format(idx))

            # parse xml
            with open(annotation_path, 'r') as f:
                soup = BeautifulSoup(f, 'xml')

            objects = soup.find_all('object')
            boxes = []
            for obj in objects:
                name = obj.find('name').text
                if name not in self.class_names:
                    continue
                class_id = self.class_names.index(name)
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.xmin.text)
                ymin = int(bndbox.ymin.text)
                xmax = int(bndbox.xmax.text)
                ymax = int(bndbox.ymax.text)
                box = [class_id, xmin, ymin, xmax, ymax]
                boxes.append(box)
            labels.append(boxes)
        return image_paths, labels

    def shuffle(self):
        samples = [(path, label) for path, label in zip(self.filenames, self.labels)]
        random.shuffle(samples)
        self.filenames = [v[0] for v in samples]
        self.labels = [v[1] for v in samples]

    def __getitem__(self, idx):
        return self.filenames[idx], self.labels[idx]

    def save_to_hdf5(self, filepath):
        hf = h5py.File(filepath, 'w')
        hf.create_dataset('filenames', data=self.filenames)
        hf.create_dataset('labels', data=self.labels)
        hf.close()

    def load_from_hdf5(self, filepath):
        hf = h5py.File(filepath, 'r')
        print(hf.keys())
        self.filenames = hf.get('filenames')
        self.labels = hf.get('labels')
        hf.close()

    def pickle(self, filepath):
        with open(filepath, 'wb') as f:
            info_str = pickle.dumps([self.filenames, self.labels])
            f.write(info_str)

    def unpickle(self, filepath):
        with open(filepath, 'rb') as f:
            data = f.read()
        items = pickle.loads(data)
        self.filenames = items[0]
        self.labels = items[1]
        assert len(self.filenames) == len(self.labels)
        self.size = len(self.filenames)


if __name__ == '__main__':
    voc2007_dir = r'D:\data\VOCdevkit\VOC2007'
    voc2012_dir = r'D:\data\VOCdevkit\VOC2012'
    dataset = DatasetPascalvoc()
    dataset.build([voc2007_dir, voc2012_dir], 'train')
    dataset.pickle('../data/voc0712.pkl')

    print(dataset.size)
