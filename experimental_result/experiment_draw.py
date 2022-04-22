import collections
import matplotlib.pyplot as plt
import numpy as np

import json
import pathlib

import scipy.spatial


class BlittedCursor:
    """
    A cross hair cursor using blitting for faster redraw.
    """

    def __init__(self, ax, points):
        self.ax = ax
        self.background = None
        self.horizontal_line = ax.axhline(color='k', lw=0.8, ls='--')
        self.vertical_line = ax.axvline(color='k', lw=0.8, ls='--')
        # text location in axes coordinates
        self.text = ax.text(0.62, 0.9, '', transform=ax.transAxes)
        
        self.ckdtree = scipy.spatial.cKDTree(points)
        
        self._creating_background = False
        ax.figure.canvas.mpl_connect('draw_event', self.on_draw)

    def closest_point_distance(self,x, y):
        # returns distance to closest point
        distance = self.ckdtree.query([x, y])[0]
        return distance

    def closest_point_id(self,x, y):
        # returns index of closest point
        index = self.ckdtree.query([x, y])[1]
        return index

    def closest_point_coords(self,x, y):
        # returns coordinates of closest point
        coord = self.ckdtree.data[self.closest_point_id( x, y)]
        return coord
        # ckdtree.data is the same as points

    def on_draw(self, event):
        self.create_new_background()

    def set_cross_hair_visible(self, visible):
        need_redraw = self.horizontal_line.get_visible() != visible
        self.horizontal_line.set_visible(visible)
        self.vertical_line.set_visible(visible)
        self.text.set_visible(visible)
        return need_redraw

    def create_new_background(self):
        if self._creating_background:
            # discard calls triggered from within this function
            return
        self._creating_background = True
        self.set_cross_hair_visible(False)
        self.ax.figure.canvas.draw()
        self.background = self.ax.figure.canvas.copy_from_bbox(self.ax.bbox)
        self.set_cross_hair_visible(True)
        self._creating_background = False

    def on_mouse_move(self, event):
        if self.background is None:
            self.create_new_background()
        if not event.inaxes:
            
            need_redraw = self.set_cross_hair_visible(False)
            if need_redraw:
                self.ax.figure.canvas.restore_region(self.background)
                self.ax.figure.canvas.blit(self.ax.bbox)
        else:
            self.set_cross_hair_visible(True)
            # update the line positions
            x, y = event.xdata, event.ydata
            x = self.closest_point_coords(x,y)[0]
            y = self.closest_point_coords(x,y)[1]
            self.horizontal_line.set_ydata(y)
            self.vertical_line.set_xdata(x)
            self.text.set_text('Epoch: %d, BLEU = %1.2f' % (x, y))
            
            self.ax.figure.canvas.restore_region(self.background)
            self.ax.draw_artist(self.horizontal_line)
            self.ax.draw_artist(self.vertical_line)
            self.ax.draw_artist(self.text)           
            
            self.ax.figure.canvas.blit(self.ax.bbox)


def draw_data(filename, x_tick):

    # Read file
    data_path = str(pathlib.Path(__file__).parent.resolve()) + str(filename)
    with open(data_path, 'rb') as handle:
        data = handle.read()

    # reconstructing the data as dictionary
    d = json.loads(data)

    # convert key in dict to integers and floats
    d = {int(k): float(v) for k, v in d.items()}

    # sort the dictionary
    od = collections.OrderedDict(sorted(d.items()))
    # print(od)
    x = list(od.keys())
    y = list(od.values())

    points = np.column_stack([x, y])

    fig, ax = plt.subplots()
    ax.set_title('BLEU score after training, max {0} at epoch {1}'.format(max(y),np.argmax(y)+1))

    markers_on = [np.argmin(y), np.argmax(y)]
    #markers_on = [y.index(min(y))]
    ax.plot(x, y, linestyle='-', marker='H', markevery=markers_on, color='b', markerfacecolor='r')
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(min(y)-2, max(y)+2)
    ax.set_xticks(np.arange(min(x), max(x)+1, x_tick))  # tick
    ax.set_ylabel('BLEU Score in test dataset')
    ax.set_xlabel('Epoch')
    ax.grid()

    blitted_cursor = BlittedCursor(ax, points)
    fig.canvas.mpl_connect('motion_notify_event', blitted_cursor.on_mouse_move)

    plt.show()


if __name__ == "__main__":
    #draw_data("/result/transformer_base.json",50)
    #draw_data("/result/actor_critic_test1.json", 1)
    draw_data("/result/test.json", 1)
