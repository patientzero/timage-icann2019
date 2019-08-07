import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler

from timage.data.image.image import ImageMixin
from matplotlib.pyplot import gca


class ReccurenceImage(ImageMixin):
    _MAX_DIV = 3  # 3 * std of z-normalized data

    def __init__(self, *args, **kwargs):
        self._min_max_scaler = MinMaxScaler()
        self._min_max_scaler.fit([[0], [ReccurenceImage._MAX_DIV]])
        super().__init__(*args, **kwargs)

    def transform(self):
        def recurrence_plot(s):
            d = pairwise_distances(np.nan_to_num(s))
            d[d > self._MAX_DIV] = self._MAX_DIV
            return self._min_max_scaler.transform(d)

        ax = gca()

        DPI = float(ax.figure.get_dpi())
        ax.figure.set_size_inches(self.size[0] / DPI, self.size[1] / DPI)
        cmap = 'gray' if self.color_depth == 1 else None
        im = ax.imshow(recurrence_plot(self.x[:, None]), cmap=cmap)
        return im
