import tensorflow as tf
import matplotlib.pyplot as plt

from GrowingNeuralGas import GrowingNeuralGas

def test():
    X = tf.concat([tf.random.normal([50, 3], 0.0, .1, dtype=tf.float32) + tf.constant([0.0, 0.0, 1.0]),
                   tf.random.normal([50, 3], 0.0, .1, dtype=tf.float32) + tf.constant([1.0, 0.0, 1.0]),
                   tf.random.normal([50, 3], 0.0, .1, dtype=tf.float32) + tf.constant([1.0, 1.0, 1.0])], 0)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2])

    growingNeuralGas = GrowingNeuralGas(eta=25)
    growingNeuralGas.fit(X, 1000)
    print(growingNeuralGas.countClusters())

    ax.scatter(growingNeuralGas.A[:, 0], growingNeuralGas.A[:, 1], growingNeuralGas.A[:, 2], 'r')
    edges = growingNeuralGas.getEdges()
    for edge in edges:
        ax.plot(edge[:, 0], edge[:, 1], edge[:, 2], 'r-')
    plt.show()



test()