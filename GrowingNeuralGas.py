import numpy as np
import tensorflow as tf

from Graph import Graph

class GrowingNeuralGas(object):

    def __init__(self, epsilon_a=.1, epsilon_n=.05, a_max=25, eta=25, alpha=.1, delta=.1, maxNumberUnits=10):
        self.A = None
        self.N = []
        self.error_ = None
        self.epsilon_a = epsilon_a
        self.epsilon_n = epsilon_n
        self.a_max = a_max
        self.eta = eta
        self.alpha = alpha
        self.delta = delta
        self.maxNumberUnits = maxNumberUnits

    def incrementAgeNeighborhood(self, indexNearestUnit):
        self.N[indexNearestUnit].incrementAgeNeighborhood(1)

    def findNearestUnit(self, xi, A):
        return tf.math.argmin(tf.math.reduce_sum(tf.math.pow(A - xi, 2), 1))

    def findSecondNearestUnit(self, xi, A):
        indexNearestUnit = self.findNearestUnit(xi, A)
        error_ = tf.constant(tf.math.reduce_sum(tf.math.pow(A - xi, 2), 1), dtype=tf.float32).numpy()
        error_[indexNearestUnit] = np.Inf
        return tf.math.argmin(tf.constant(error_))

    def findIndexNeighbourMaxError(self, indexUnitWithMaxError_):
        index = tf.squeeze(tf.math.argmax(tf.gather(self.error_, self.N[indexUnitWithMaxError_].neighborhood)), 0)
        indexNeighbourMaxError = self.N[indexUnitWithMaxError_].neighborhood[index]
        return indexNeighbourMaxError

    def pruneA(self):
        indexToNotRemove = [index for index in tf.range(self.N.__len__()) if self.N[index].neighborhood.__len__() > 0]
        newN = []
        for index in tf.range(indexToNotRemove.__len__()):
            newN.append(Graph(index, self.N[indexToNotRemove[index]].neighborhood, self.N[indexToNotRemove[index]].ageNeighborhood))
        self.N = newN
        self.A.assign(tf.gather(self.A, indexToNotRemove, axis=0))

    def countClusters(self):
        visited = [False for i in range(len(self.N))]
        stack = []

        count = 0
        for unit in self.N:
            if not visited[unit.id]:
                count += 1
                stack.append(unit)

                while len(stack):
                    current = stack[-1]
                    stack.pop()

                    if not visited[current.id]:
                        print(current, end=' ')
                        visited[current.id] = True

                    for node in current.neighborhood:
                        for checkNode in self.N:
                            if tf.cast(checkNode.id, dtype=tf.int64) == node:
                                node = checkNode
                                break
                        if not visited[node.id]:
                            stack.append(node)
                print()

        return count

    def getEdges(self):
        visited = [False for i in range(len(self.N))]
        stack = []
        edges = []

        for unit in self.N:
            if not visited[unit.id]:
                stack.append(unit)

                while len(stack):
                    current = stack[-1]
                    stack.pop()

                    if not visited[current.id]:
                        visited[current.id] = True

                    for node in current.neighborhood:
                        edge = tf.stack([self.A[current.id], self.A[node]])
                        if len(edges) == 0 or (not tf.reduce_any(tf.reduce_all(tf.equal(edge, edges), axis=(1, 2)))):
                            edges.append(edge)
                        for checkNode in self.N:
                            if tf.cast(checkNode.id, dtype=tf.int64) == node:
                                node = checkNode
                                break
                        if not visited[node.id]:
                            stack.append(node)
        return edges


    def fit(self, trainingX, numberEpochs):
        self.A = tf.Variable(tf.random.normal([2, trainingX.shape[1]], 0.0, 1.0, dtype=tf.float32))
        self.N.append(Graph(0))
        self.N.append(Graph(1))
        self.error_ = tf.Variable(tf.zeros([2, 1]), dtype=tf.float32)
        epoch = 0
        numberProcessedRow = 0
        while epoch < numberEpochs and self.A.shape[0] < self.maxNumberUnits:
                shuffledTrainingX = tf.random.shuffle(trainingX)
                for row_ in tf.range(shuffledTrainingX.shape[0]):
                    xi = shuffledTrainingX[row_]

                    indexNearestUnit = self.findNearestUnit(xi, self.A)
                    self.incrementAgeNeighborhood(indexNearestUnit)
                    indexSecondNearestUnit = self.findSecondNearestUnit(xi, self.A)

                    self.error_[indexNearestUnit].assign(self.error_[indexNearestUnit] + tf.math.reduce_sum(tf.math.squared_difference(xi, self.A[indexNearestUnit])))

                    self.A[indexNearestUnit].assign(self.A[indexNearestUnit] + self.epsilon_a * (xi - self.A[indexNearestUnit]))
                    for indexNeighbour in self.N[indexNearestUnit].neighborhood:
                        self.A[indexNeighbour].assign(self.A[indexNeighbour] + self.epsilon_n * (xi - self.A[indexNeighbour]))

                    if indexSecondNearestUnit in self.N[indexNearestUnit].neighborhood:
                        self.N[indexNearestUnit].setAge(indexSecondNearestUnit, 0.0)
                        self.N[indexSecondNearestUnit].setAge(indexNearestUnit, 0.0)
                    else:
                        self.N[indexNearestUnit].addNeighbour(indexSecondNearestUnit, 0.0)
                        self.N[indexSecondNearestUnit].addNeighbour(indexNearestUnit, 0.0)

                    for graph in self.N:
                        graph.pruneGraph(self.a_max)

                    self.pruneA()

                    if not (numberProcessedRow + 1) % self.eta:
                        indexUnitWithMaxError_ = tf.squeeze(tf.math.argmax(self.error_), 0)
                        indexNeighbourWithMaxError_ = self.findIndexNeighbourMaxError(indexUnitWithMaxError_)

                        self.A = tf.Variable(tf.concat([self.A, tf.expand_dims(0.5 * (self.A[indexUnitWithMaxError_] + self.A[indexNeighbourWithMaxError_]), 0)], 0))

                        self.N.append(Graph(self.A.shape[0] - 1, [indexUnitWithMaxError_, indexNeighbourWithMaxError_], [0.0, 0.0]))
                        self.N[indexUnitWithMaxError_].removeNeighbour(indexNeighbourWithMaxError_)
                        self.N[indexUnitWithMaxError_].addNeighbour(tf.constant(self.A.shape[0] - 1, dtype=tf.int64), 0.0)
                        self.N[indexNeighbourWithMaxError_].removeNeighbour(indexUnitWithMaxError_)
                        self.N[indexNeighbourWithMaxError_].addNeighbour(tf.constant(self.A.shape[0] - 1, dtype=tf.int64), 0.0)

                        self.error_[indexUnitWithMaxError_].assign(self.error_[indexUnitWithMaxError_] * self.alpha)
                        self.error_[indexNeighbourWithMaxError_].assign(self.error_[indexNeighbourWithMaxError_] * self.alpha)
                        self.error_ = tf.Variable(tf.concat([self.error_,  tf.expand_dims(self.error_[indexUnitWithMaxError_], 0)], 0))

                    self.error_.assign(self.error_ * self.delta)
                    numberProcessedRow += 1

                epoch += 1






