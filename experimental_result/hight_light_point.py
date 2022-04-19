import sys

import numpy as np

from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure


class Snapper:
    """Snaps to data points"""

    def __init__(self, data, callback):
        self.data = data
        self.callback = callback

    def snap(self, x, y):
        pos = np.array([x, y])
        distances = np.linalg.norm(self.data - pos, axis=1)
        dataidx = np.argmin(distances)
        datapos = self.data[dataidx,:]
        self.callback(datapos[0], datapos[1])
        return datapos


class SnappingNavigationToolbar(NavigationToolbar2QT):
    """Navigation toolbar with data snapping"""

    def __init__(self, canvas, parent, coordinates=True):
        super().__init__(canvas, parent, coordinates)
        self.snapper = None

    def set_snapper(self, snapper):
        self.snapper = snapper

    def mouse_move(self, event):
        if self.snapper and event.xdata and event.ydata:
            event.xdata, event.ydata = self.snapper.snap(event.xdata, event.ydata)
        super().mouse_move(event)


class Highlighter:
    def __init__(self, ax):
        self.ax = ax
        self.marker = None
        self.markerpos = None

    def draw(self, x, y):
        """draws a marker at plot position (x,y)"""
        if (x, y) != self.markerpos:
            if self.marker:
                self.marker.remove()
                del self.marker
            self.marker = self.ax.scatter(x, y, color='yellow')
            self.markerpos = (x, y)
            self.ax.figure.canvas.draw()


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        layout = QtWidgets.QVBoxLayout(self._main)
        canvas = FigureCanvas(Figure(figsize=(5,3)))
        layout.addWidget(canvas)
        toolbar = SnappingNavigationToolbar(canvas, self)
        self.addToolBar(toolbar)

        data = np.random.randn(100, 2)
        ax = canvas.figure.subplots()
        ax.scatter(data[:,0], data[:,1])

        self.highlighter = Highlighter(ax)
        snapper = Snapper(data, self.highlighter.draw)
        toolbar.set_snapper(snapper)


if __name__ == "__main__":
    qapp = QtWidgets.QApplication(sys.argv)
    app = ApplicationWindow()
    app.show()
    qapp.exec_()