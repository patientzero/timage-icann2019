from matplotlib import use
import matplotlib.pyplot as plt

use('Agg')


class ImageMixin(object):

    def __init__(self, x, size=(640, 480), color_depth=3):
        self.x = x
        self.size = size
        self.color_depth = color_depth

    def transform(self):
        raise NotImplementedError()

    def save(self, file_name):
        im = self.transform()
        im.figure.subplots_adjust(bottom=0, top=1, left=0, right=1)
        im.figure.savefig(str(file_name), bbox_inches=0, pad_inches=0, dpi="figure")
        plt.close("all")
