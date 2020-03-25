import matplotlib.pyplot as plt
import uuid
from abc import abstractmethod, ABC


class StaticPlot(ABC):
    def show(self, save_out=None, show_on_window=True):
        plt.close()
        self._plot()

        if save_out:
            plt.savefig(save_out)

        if show_on_window:
            plt.show()

    def merge(self):
        pass

    @abstractmethod
    def _plot(self):
        pass

class LineChart(StaticPlot):
    def __init__(self, x, y):
        self._x = x
        self._y = y

    def _plot(self):
        plt.plot(self._x, self._y)

class Hist1D(StaticPlot):
    def __init__(self, x, bins=256, value_range=(0, 255)):
        if len(x.shape) > 1:
            total = 1
            for i in range(len(x.shape)):
                total *= x.shape[i]
            x = np.reshape(x, total)
        self._x = x
        self._bins = bins
        self._range = value_range

    def _plot(self):
        plt.hist(self._x, bins=self._bins, range=self._range)


class Hist2D(StaticPlot):
    def __init__(self, x,y):
        self._x = x
        self._y = y

    def _plot(self):
        plt.hist2d(self._x, self._y, bins=[50, 50],range=[(0,255), [0,255]], cmap=plt.cm.Reds)
        plt.colorbar()

