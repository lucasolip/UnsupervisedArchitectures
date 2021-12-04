import tensorflow as tf

from GrowingNeuralGas import GrowingNeuralGas

def test():
    X = tf.concat([tf.random.normal([50, 3], 0.0, 0.25, dtype=tf.float32) + tf.constant([0.0, 0.0, 1.0]),
                   tf.random.normal([50, 3], 0.0, 0.25, dtype=tf.float32) + tf.constant([1.0, 0.0, 1.0]),
                   tf.random.normal([50, 3], 0.0, 0.25, dtype=tf.float32) + tf.constant([1.0, 1.0, 1.0])], 0)

    growingNeuralGas = GrowingNeuralGas()
    growingNeuralGas.fit(X, 5)
