import gc

import numpy as np
from pathlib import Path
from ucrloader import UCRLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import OneHotEncoder

from PIL import Image


class UCRImageLoader(UCRLoader):

    def load(self, name, lbl_dict=None):
        try:
            return UCRImage.from_path(self._data_dirs[name], lbl_dict)
        except KeyError:
            raise FileNotFoundError(name)

    @classmethod
    def from_path(cls, path, size, color_depth, image_cls):
        full_path = cls.dir_name(path, size, color_depth, image_cls)
        if not full_path.exists():
            raise FileNotFoundError(full_path)
        return cls(str(full_path))

    @classmethod
    def convert(cls, ucr_data_loader, path, size, color_depth, image_cls, names=None, z_norm=False):

        for name in ucr_data_loader.names if names is None else names:
            print("Creating images for {}".format(name))
            data = ucr_data_loader.load(name, z_norm)
            UCRImage.convert(data, path, size, color_depth, image_cls)
            del data
            gc.collect()

    @classmethod
    def dir_name(cls, path, size, color_depth, image_cls):
        return Path(path) / image_cls.__name__ / "x".join(map(str, (*size, color_depth)))


class UCRImage(object):
    FILE_EXTENSION = ".png"
    LABELS_FILE = "labels.csv"
    TEST_DIR = "test"
    TRAIN_DIR = "train"

    def __init__(self, train_data, train_labels, test_data, test_labels, num_classes=None):
        categories = [range(num_classes)] if num_classes is not None else 'auto'
        self._label_encoder = OneHotEncoder(categories=categories, sparse=False)

        self.train_data = train_data
        self.test_data = test_data

        self.train_labels = self._label_encoder.fit_transform(train_labels.reshape(-1, 1))
        self.test_labels = self._label_encoder.transform(test_labels.reshape(-1, 1))

        self.train_class_weight = dict(zip(
            self._label_encoder.categories_[0].tolist(),
            compute_class_weight('balanced', np.unique(train_labels), train_labels).tolist()
        ))

    @staticmethod
    def translate_lbls(lbl_dict, ds_name, labels):
        return np.array([lbl_dict[ds_name][l] for l in labels])

    @staticmethod
    def translate_lbls_reverse(lbl_dict, ds_name, labels):
        ret = []
        local_dict = {v: k for k, v in lbl_dict[ds_name].items()}
        for l in labels:
            ret.append(local_dict.get(l, l))
        return ret

    @classmethod
    def from_path(cls, path, lbl_dict=None):
        def read(is_train):
            image_dir = path / (cls.TRAIN_DIR if is_train else cls.TEST_DIR)
            labels = list(map(int, (image_dir / cls.LABELS_FILE).read_text().split("\n")))
            images = []
            for p in sorted(image_dir.glob("*" + cls.FILE_EXTENSION)):
                # im = np.asarray(Image.open(str(p.absolute())))
                # TODO: Currently always converts to bw image
                im = np.asarray(Image.open(str(p.absolute())).convert('LA'))
                color_depth = im.shape[2]
                images.append(im[:, :, :color_depth - 1])  # image without alpha

            return np.array(images), np.array(labels)

        train_data, train_labels = read(True)
        test_data, test_labels = read(False)
        if lbl_dict is None:
            return cls(train_data, train_labels, test_data, test_labels)
        else:
            train_labels = cls.translate_lbls(lbl_dict, path.name, train_labels)
            test_labels = cls.translate_lbls(lbl_dict, path.name, test_labels)
            return cls(train_data, train_labels,
                       test_data, test_labels,
                       num_classes=sum(map(len, lbl_dict.values())))

    @classmethod
    def convert(cls, data, path, size, color_depth, image_cls):
        # Convert labels to range: [0, NUM_CLASSES]
        unique_labels = sorted(np.unique(data.train_labels))
        test_labels = np.array([unique_labels.index(v) for v in data.test_labels])
        train_labels = np.array([unique_labels.index(v) for v in data.train_labels])

        cls.save(data.name, data.test_data, test_labels, path, size, color_depth, False, image_cls)
        cls.save(data.name, data.train_data, train_labels, path, size, color_depth, True, image_cls)

    @classmethod
    def dir_name(cls, path, name, size, color_depth, is_train, image_cls):
        return UCRImageLoader.dir_name(path, size, color_depth, image_cls) / name / (
            cls.TRAIN_DIR if is_train else cls.TEST_DIR)

    @classmethod
    def save(cls, name, data, labels, path, size, color_depth, is_train, image_cls):

        dir_name = cls.dir_name(path, name, size, color_depth, is_train, image_cls)
        dir_name.mkdir(parents=True, exist_ok=True)
        # save labels
        (dir_name / cls.LABELS_FILE).write_text("\n".join(map(str, labels)))

        # save images
        fn_format = "{:0%sd}{}" % len(str(len(data)))
        for idx, d in enumerate(data):
            file_name = dir_name / fn_format.format(idx, cls.FILE_EXTENSION)
            image_cls(d, size=(size[0], size[1]), color_depth=color_depth).save(file_name)

    @classmethod
    def _class_weights(cls, labels):
        return np.bincount(labels)
