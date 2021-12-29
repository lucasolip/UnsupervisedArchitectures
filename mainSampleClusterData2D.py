import tensorflow as tf
import matplotlib.pyplot as plt
from csvController import CsvController

from GrowingNeuralGas import GrowingNeuralGas

# Model
epochs = 10
eta = 25

# Test set
standard_deviation = .1

def test():

    csvController = CsvController("Sample_Cluster_Data_2D.csv", hasHeader= True)
    data, header = csvController.getCsvToTensor()

    growingNeuralGas = GrowingNeuralGas(eta=25)
    growingNeuralGas.fit(data, epochs)
    print(growingNeuralGas.countClusters())
    print(growingNeuralGas.A.shape)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(data[:, 0], data[:, 1])

    ax.scatter(growingNeuralGas.A[:, 0], growingNeuralGas.A[:, 1], 'r')
    edges = growingNeuralGas.getEdges()
    for edge in edges:
        ax.plot(edge[:, 0], edge[:, 1], 'r-')

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(projection='3d')
    clusters = [0 for i in range(X.shape[0])]
    for i in range(X.shape[0]):
        clusters[i] = growingNeuralGas.predict(X[i])
    ax2.scatter(X[:, 0], X[:, 1], c=clusters)
    plt.show()



test()