import unittest
import tensorflow as tf

from GrowingNeuralGas import GrowingNeuralGas
from Graph import Graph

class ut_GrowingNeuralGas(unittest.TestCase):
    def test_findNearestUnit_and_findSecondNearestUnit(self):
        A = tf.constant([[0., 0., 0.], [0., 1., 0.], [0.5, 0.1, 0.], [2., 3., 0.5], [0.1, 0.5, 0.5]], dtype=tf.float32)
        xi = tf.expand_dims(tf.constant([0.45, 0.15, 0.05]), 0)

        growingNeuralGas = GrowingNeuralGas()
        indexNearestUnit = growingNeuralGas.findNearestUnit(xi, A)
        indexSecondNearestUnit = growingNeuralGas.findSecondNearestUnit(xi, A)

        self.assertEqual(indexNearestUnit, 2)
        self.assertEqual(indexSecondNearestUnit, 0)

    def test_pruneA(self):
        aBase = tf.constant([[0., 0., 0.], [0.5, 0.1, 0.], [0.1, 0.5, 0.5]], dtype=tf.float32)
        nBase = []
        nBase.append(Graph(0, [2, 4], [71, 31]))
        nBase.append(Graph(1, [0, 4], [21, 41]))
        nBase.append(Graph(2, [0, 2], [11, 32]))

        aTest = tf.constant([[0., 0., 0.], [0., 1., 0.], [0.5, 0.1, 0.], [2., 3., 0.5], [0.1, 0.5, 0.5]], dtype=tf.float32)
        nTest = []
        nTest.append(Graph(0, [2, 4], [71, 31]))
        nTest.append(Graph(1))
        nTest.append(Graph(2, [0, 4], [21, 41]))
        nTest.append(Graph(3))
        nTest.append(Graph(4, [0, 2], [11, 32]))

        growingNeuralGas = GrowingNeuralGas()
        growingNeuralGas.A = aTest
        growingNeuralGas.N = nTest

        growingNeuralGas.pruneA()

        self.assertTrue(tf.math.reduce_all(aBase == growingNeuralGas.A))
        for graphBase, graphTest in zip(nBase, growingNeuralGas.N):
            self.assertEqual(graphBase, graphTest)


if __name__ == '__main__':
    unittest.main()
