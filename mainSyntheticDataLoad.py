import tensorflow as tf
import matplotlib.pyplot as plt

from GrowingNeuralGas import GrowingNeuralGas

# Model
epochs = 10
eta = 25

# Test set
standard_deviation = .1

def test():
    X = tf.concat([tf.random.normal([50, 3], 0.0, standard_deviation, dtype=tf.float32) + tf.constant([0.0, 0.0, 1.0]),
                   tf.random.normal([50, 3], 0.0, standard_deviation, dtype=tf.float32) + tf.constant([1.0, 0.0, 1.0]),
                   tf.random.normal([50, 3], 0.0, standard_deviation, dtype=tf.float32) + tf.constant([1.0, 1.0, 1.0])], 0)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2])

    growingNeuralGas = GrowingNeuralGas(eta=25)
    growingNeuralGas.loadModel("model.h")
    growingNeuralGas.fit(X, epochs, modelLoaded= True)
    print(growingNeuralGas.countClusters())
    print(growingNeuralGas.A.shape)

    ax.scatter(growingNeuralGas.A[:, 0], growingNeuralGas.A[:, 1], growingNeuralGas.A[:, 2], 'r')
    edges = growingNeuralGas.getEdges()
    for edge in edges:
        ax.plot(edge[:, 0], edge[:, 1], edge[:, 2], 'r-')

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(projection='3d')
    clusters = [0 for i in range(X.shape[0])]
    for i in range(X.shape[0]):
        clusters[i] = growingNeuralGas.predict(X[i])
    ax2.scatter(X[:, 0], X[:, 1], X[:, 2], c=clusters)
    plt.show()

test()