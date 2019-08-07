from click import progressbar
from pathlib import Path
from tensorflow.train import summary_iterator
import matplotlib.pyplot as plt
import logging
from .data.image.loader import UCRImageLoader


class SingleResultsParser:

    def __init__(self, loader, network, label_dict):
        self.loader = loader
        self.network = network
        self.label_dict = label_dict

    @staticmethod
    def _score_dict_to_csv(score_dict):
        csv = "Dataset;TestAccuracy\n"
        for k, v in score_dict.items():
            csv += "{};{}\n".format(k, v)
        return csv

    def as_csv(self, datasets=None, out_file=None):
        scores = {}
        datasets = self.loader.names if datasets is None else datasets
        with progressbar(datasets) as dsi:
            for name in dsi:
                try:
                    data = self.loader.load(name, lbl_dict=self.label_dict)
                    score = self.network.score(data.test_data, data.test_labels)
                    scores[name] = score[1]
                except FileNotFoundError:
                    logging.error("Images not found: {}".format(name))
                except Exception as e:
                    logging.exception(e)

        if out_file is None:
            print(SingleResultsParser._score_dict_to_csv(scores))
        else:
            Path(out_file).write_text(
                SingleResultsParser._score_dict_to_csv(scores)
            )


class ResultParser:

    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        if not self.log_dir.exists():
            raise FileNotFoundError(log_dir)

    def max_val_acc(self, event_file_name):
        max_val_acc = 0
        for e in summary_iterator(event_file_name):
            for v in e.summary.value:
                if v.tag == 'val_acc':
                    if v.simple_value > max_val_acc:
                        max_val_acc = v.simple_value

        return max_val_acc

    def as_dict(self):
        result = {}
        directories = [d for d in self.log_dir.glob('*') if d.is_dir()]
        logging.info("Parsing log files of {} datasets".format(len(directories)))
        with progressbar(directories) as di:
            for d in di:
                if d.is_dir():
                    try:
                        file_name = str(next(d.glob('events.out.tfevents.*')))  # take the first file
                        max_val_acc = self.max_val_acc(file_name)
                        result[d.name] = max_val_acc
                    except StopIteration:
                        pass
        return result

    def as_csv(self):
        csv = ""
        for k, v in sorted(self.as_dict().items()):
            csv += "{};{}\n".format(k, v)
        return csv


class GraphicalResults:

    def __init__(self, experiment_dir, image_dir, network_model, image_cls, resolution, color_depth, lbl_dict=None):
        """
        :param image_dir: UCR base image dir
        :param experiment_dir: path to dir with trainings
        """
        from timage.data.image import image_classes
        from .network import network_models_classes
        self.image_dir = Path(image_dir)
        self.network_model_cls = network_models_classes[network_model]
        self.image_cls = image_classes[image_cls]
        self.img_rows, self.img_cols = resolution.split('x')
        self.color_depth = color_depth
        self.lbl_dict = lbl_dict
        self.loader = UCRImageLoader.from_path(self.image_dir, (self.img_rows, self.img_cols), color_depth,
                                               image_classes[image_cls])
        self.training_data_dir = UCRImageLoader.dir_name(self.image_dir, resolution, self.color_depth, self.image_cls)
        self.base_path = Path(experiment_dir) / self.network_model_cls.NAME / self.image_cls.__name__ / "x".join((
            resolution, str(color_depth)))

    @property
    def is_allnet(self):
        return self.lbl_dict is not None

    def confusion_matrix(self, out_file_folder, data_sets=None):
        out_file_folder = Path(out_file_folder)
        if not out_file_folder.exists():
            out_file_folder.mkdir(parents=True)
        if data_sets is None:
            data_sets = self.loader.names
        if self.is_allnet:
            self._confusion_matrix_all(out_file_folder, data_sets)
        else:
            self._confusion_matrix_single(out_file_folder, data_sets)

    @staticmethod
    def get_latest_weights_from_path(path):
        return sorted(str(m) for m in path.iterdir() if m.suffix == '.h5')[-1]

    def _confusion_matrix_all(self, out_file_folder, data_sets):
        from sklearn.metrics import confusion_matrix

        weights_dir = self.base_path / 'all' / 'model'
        number_classes = sum([len(x) for x in self.lbl_dict.values()])
        model = self.network_model_cls(self.image_cls,
                                       int(self.img_rows),
                                       int(self.img_cols),
                                       self.color_depth,
                                       number_classes,
                                       weights_path=self.get_latest_weights_from_path(weights_dir))
        model.load_weights(False)
        with progressbar(data_sets) as dsi:
            for ds in dsi:
                data = self.loader.load(ds)
                out_file = (out_file_folder / (ds + '_' +
                                               self.image_cls.__name__ + '_' +
                                               self.network_model_cls.NAME +
                                               '_all.png'))

                y_pred = model.predict(data.test_data, verbose=1).argmax(axis=1)
                y_pred_translated = data.translate_lbls_reverse(self.lbl_dict, ds, y_pred)
                y_true = data.test_labels.argmax(axis=1)
                conf_matrix = confusion_matrix(y_true, y_pred_translated)

                labels = sorted(list(set(y_true)))
                GraphicalResults.plot_confusion_matrix(conf_matrix, labels, out_file, title=ds)

    def _confusion_matrix_single(self, out_file_folder, data_sets):
        from sklearn.metrics import confusion_matrix

        with progressbar(data_sets) as dsi:
            for ds in dsi:
                data = self.loader.load(ds)

                out_file = (out_file_folder / (
                            ds + '_' + self.image_cls.__name__ + '_' + self.network_model_cls.NAME +
                            '_single.png'))
                weights_dir = self.base_path / ds / 'model'
                try:
                    weights_path = self.get_latest_weights_from_path(weights_dir)
                except FileNotFoundError:
                    logging.error("No weights found for {}".format(ds))
                    continue

                network = self.network_model_cls(self.image_cls,
                                                 int(self.img_rows),
                                                 int(self.img_cols),
                                                 self.color_depth,
                                                 num_classes=data.train_labels.shape[1],
                                                 weights_path=weights_path
                                                 )

                network.load_weights(by_name=False)

                y_pred = network.predict(data.test_data, verbose=1).argmax(axis=1)
                y_true = data.test_labels.argmax(axis=-1)
                conf_matrix = confusion_matrix(y_true, y_pred)

                labels = sorted(list(set(y_true)))
                GraphicalResults.plot_confusion_matrix(conf_matrix, labels, out_file, title=ds, plot_legend=True)
                del network

    @staticmethod
    def plot_confusion_matrix(cm, classes, out_file, normalize=False, title='Confusion matrix', cmap=plt.cm.plasma,
                              plot_legend=False):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        import numpy as np
        import itertools

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)
        np.set_printoptions(precision=2)
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        if plot_legend:
            plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, '',
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(str(out_file))
        plt.close('all')
