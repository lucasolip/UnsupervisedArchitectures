import tensorflow as tf

from GrowingNeuralGas import GrowingNeuralGas

# Model
from GrowingNeuralGasPlotter import GrowingNeuralGasPlotter

epochs = 10
eta = 25

# Test set
standard_deviation = .1

def test():
    X = tf.concat([tf.random.normal([50, 3], 0.0, standard_deviation, dtype=tf.float32) + tf.constant([0.0, 0.0, 1.0]),
                   tf.random.normal([50, 3], 0.0, standard_deviation, dtype=tf.float32) + tf.constant([1.0, 0.0, 1.0]),
                   tf.random.normal([50, 3], 0.0, standard_deviation, dtype=tf.float32) + tf.constant([1.0, 1.0, 1.0])], 0)



    growingNeuralGas = GrowingNeuralGas(eta=25)
    growingNeuralGas.fit(X, epochs)
    print(growingNeuralGas.countClusters())
    print(growingNeuralGas.A.shape)

    GrowingNeuralGasPlotter.plotNetworkStructure3D(growingNeuralGas.A, X, growingNeuralGas.getEdges(), title="Estructura de la red")
    GrowingNeuralGasPlotter.plotClusters3D(growingNeuralGas, X, title="Agrupamientos encontrados")
    GrowingNeuralGasPlotter.show()


test()