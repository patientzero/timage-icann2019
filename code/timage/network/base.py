from sklearn.base import BaseEstimator, ClassifierMixin


class BaseNetworkModel(BaseEstimator, ClassifierMixin):
    NAME = None

    def __init__(self, image_cls, img_rows, img_cols, color_depth=1, num_classes=1,
                 weights_path=None):
        self.image_cls = image_cls
        self.img_cols = img_cols
        self.img_rows = img_rows
        self.color_depth = color_depth
        self.num_classes = num_classes
        self.weights_path = weights_path
        self._model = None

    @property
    def model(self):
        raise NotImplementedError('Model needs to be implemented')

    def log_dir(self, base_log_dir, name):
        resolution_dir = "x".join(map(str, (self.img_rows, self.img_cols, self.color_depth)))
        return base_log_dir / self.NAME / self.image_cls.__name__ / resolution_dir / name

    def load_weights(self, by_name=True):
        if self.weights_path is not None:
            self.model.load_weights(self.weights_path, by_name=by_name)

    def fit(self, x, y, name, validation_data=[], class_weight=None, nb_epoch=100, batch_size=8, base_log_dir="."):
        import logging

        from keras.callbacks import TensorBoard
        from keras.callbacks import ModelCheckpoint
        from timage.network import config as conf

        conf.configure_seed()

        log_dir = self.log_dir(base_log_dir, name)

        pmodel = log_dir / 'model'
        pmodel.mkdir(parents=True, exist_ok=True)

        tensorboard = TensorBoard(log_dir=str(log_dir))

        logging.info("Loading Model for {}".format(name))
        mc = ModelCheckpoint(str(pmodel / (name + '_weights_{epoch:04d}-{val_acc:03.02f}.h5')),
                             save_best_only=True,
                             save_weights_only=True,
                             monitor='val_acc',
                             mode='max')

        logging.info("Fit Model for {}".format(name))
        # Start Fine-tuning
        self.model.fit(x, y,
                       batch_size=batch_size,
                       epochs=nb_epoch,
                       shuffle=True,
                       verbose=1,
                       validation_data=validation_data,
                       callbacks=[tensorboard, mc],
                       class_weight=class_weight
                      )

        self.save(pmodel, name)

    def save(self, path, name):
        self.model.save(str(path / '{}.h5'.format(name)))

    def score(self, x, y, verbose=0):
        return self.model.evaluate(x, y, verbose=0)

    def predict(self, x, verbose=0):
        return self.model.predict(x, verbose=verbose)

    def clear_model(self):
        from keras.backend import clear_session
        if self._model is not None:
            del self._model
        clear_session()

    def __del__(self):
        self.clear_model()


class LoadedNetworkModel(BaseNetworkModel):

    def __init__(self):
        pass

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    def fit(self):
        raise NotImplementedError("Loaded Model can not be refitted")
