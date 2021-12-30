import matplotlib.pyplot as plt

import tensorflow as tf

class GrowingNeuralGasPlotter(object):
    @staticmethod
    def plotGraphConnectedComponent(pathFigure, nameFigure, A, N, X, edges):
        if len(X[0]) == 3:
            figure = plt.figure()
            axis = figure.add_subplot(projection='3d')
            axis.scatter(X[:, 0], X[:, 1], X[:, 2])

            x = [A[index][0].numpy() for index in tf.range(A.shape[0])]
            y = [A[index][1].numpy() for index in tf.range(A.shape[0])]
            z = [A[index][2].numpy() for index in tf.range(A.shape[0])]

            graphZero = axis.scatter(x, y, z)
            for edge in edges:
                axis.plot(edge[:, 0], edge[:, 1], edge[:, 2], 'r-')
        elif len(X[0]) == 2:
            figure = plt.figure()
            axis = figure.add_subplot(projection='3d')
            axis.scatter(X[:, 0], X[:, 1])

            x = [A[index][0].numpy() for index in tf.range(A.shape[0])]
            y = [A[index][1].numpy() for index in tf.range(A.shape[0])]

            graphZero = axis.scatter(x, y)
            for edge in edges:
                axis.plot(edge[:, 0], edge[:, 1], 'r-')
        # matplotlib.pyplot.show()
        figure.savefig(pathFigure + '//' + nameFigure + '.png', transparent=False, dpi=80, bbox_inches="tight")
        plt.close(figure)

    @staticmethod
    def plotNetworkStructure2D(A, X, edges, title="", save=False, pathFigure=".//", nameFigure="networkStructure2D"):
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.scatter(X[:, 0], X[:, 1])
        ax.scatter(A[:, 0], A[:, 1], c='r')
        for edge in edges:
            ax.plot(edge[:, 0], edge[:, 1], c='r')
        ax.set_title(title)

        if save:
            fig.savefig(pathFigure + '//' + nameFigure + '.png', transparent=False, dpi=80, bbox_inches="tight")

    @staticmethod
    def plotNetworkStructure3D(A, X, edges, title="", save=False, pathFigure=".//", nameFigure="networkStructure2D"):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2])
        ax.scatter(A[:, 0], A[:, 1], A[:, 2], 'r')
        for edge in edges:
            ax.plot(edge[:, 0], edge[:, 1], edge[:, 2], 'r-')
        ax.set_title(title)
        if save:
            fig.savefig(pathFigure + '//' + nameFigure + '.png', transparent=False, dpi=80, bbox_inches="tight")

    @staticmethod
    def plotClusters2D(growingNeuralGas, X, title=""):
        fig = plt.figure()
        ax = fig.add_subplot()
        clusters = [0 for i in range(X.shape[0])]
        for i in range(X.shape[0]):
            clusters[i] = growingNeuralGas.predict(X[i])
        ax.scatter(X[:, 0], X[:, 1], c=clusters)
        ax.set_title(title)

    @staticmethod
    def plotClusters3D(growingNeuralGas, X, title=""):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        clusters = [0 for i in range(X.shape[0])]
        for i in range(X.shape[0]):
            clusters[i] = growingNeuralGas.predict(X[i])
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=clusters)
        ax.set_title(title)

    @staticmethod
    def show():
        plt.show()