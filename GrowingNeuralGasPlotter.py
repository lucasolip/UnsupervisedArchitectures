import matplotlib.pyplot as plt

import tensorflow as tf

class GrowingNeuralGasPlotter(object):
    @staticmethod
    def plotGraphConnectedComponent(pathFigure, nameFigure, growingNeuralGas):
        figure = plt.figure()
        ax = figure.add_subplot(projection='3d')

        ax.scatter(growingNeuralGas.A[:, 0], growingNeuralGas.A[:, 1], growingNeuralGas.A[:, 2], 'r')
        edges = growingNeuralGas.getEdges()
        for edge in edges:
            ax.plot(edge[:, 0], edge[:, 1], edge[:, 2], 'r-')

        # matplotlib.pyplot.show()
        figure.savefig(pathFigure + '//' + nameFigure + '.png', transparent=False, dpi=80, bbox_inches="tight")
        plt.close('all')