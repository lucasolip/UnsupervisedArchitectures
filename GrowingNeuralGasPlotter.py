import matplotlib.pyplot as plt

import tensorflow as tf

class GrowingNeuralGasPlotter(object):
    @staticmethod
    def plotGraphConnectedComponent(pathFigure, nameFigure, A, N, X, edges):
        figure = plt.figure()
        axis = figure.add_subplot(projection='3d')
        axis.scatter(X[:, 0], X[:, 1], X[:, 2])

        x = [A[index][0].numpy() for index in tf.range(A.shape[0])]
        y = [A[index][1].numpy() for index in tf.range(A.shape[0])]
        z = [A[index][2].numpy() for index in tf.range(A.shape[0])]

        graphZero = axis.scatter(x, y, z)
        for edge in edges:
            axis.plot(edge[:, 0], edge[:, 1], edge[:, 2], 'r-')

        # matplotlib.pyplot.show()
        figure.savefig(pathFigure + '//' + nameFigure + '.png', transparent=False, dpi=80, bbox_inches="tight")
        figure.savefig(pathFigure + '//' + nameFigure + '.svg')
        plt.close(figure)