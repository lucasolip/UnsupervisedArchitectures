import functools

import numpy
import numpy as np
import tensorflow as tf
import pickle

from Graph import Graph
from GrowingNeuralGasPlotter import GrowingNeuralGasPlotter


class GrowingNeuralGas(object):

    def __init__(self, epsilon_a=.1, epsilon_n=.05, a_max=10, eta=5, alpha=.1, delta=.1, maxNumberUnits=1000):
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

        self.clusters = []

    def incrementAgeNeighborhood(self, indexNearestUnit):
        self.N[indexNearestUnit].incrementAgeNeighborhood(1.0)
        for indexNeighbour in self.N[indexNearestUnit].neighborhood:
            self.N[indexNeighbour].incrementAgeNeighbour(indexNearestUnit, 1.0)

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
        self.A = tf.Variable(tf.gather(self.A, indexToNotRemove, axis=0))

        for graphIndex in reversed(range(self.N.__len__())):
            if self.N[graphIndex].neighborhood.__len__() == 0:
                for pivot in range(graphIndex + 1, self.N.__len__()):
                    self.N[pivot].id -= 1
                    for indexN in range(self.N.__len__()):
                        for indexNeighbothood in range(self.N[indexN].neighborhood.__len__()):
                            if self.N[indexN].neighborhood[indexNeighbothood] == pivot:
                                self.N[indexN].neighborhood[indexNeighbothood] -= 1
                self.N.pop(graphIndex)

    def getGraphConnectedComponents(self):
        connectedComponentIndeces = list(range(self.N.__len__()))
        for graphIndex in range(self.N.__len__()):
            for neighbourIndex in self.N[graphIndex].neighborhood:
                if connectedComponentIndeces[graphIndex] <= connectedComponentIndeces[neighbourIndex]:
                    connectedComponentIndeces[neighbourIndex] = connectedComponentIndeces[graphIndex]
                else:
                    aux = connectedComponentIndeces[graphIndex]
                    for pivot in range(graphIndex, self.N.__len__()):
                        if connectedComponentIndeces[pivot] == aux:
                            connectedComponentIndeces[pivot] = connectedComponentIndeces[neighbourIndex]
        uniqueConnectedComponentIndeces = functools.reduce(
            lambda cCI, index: cCI.append(index) or cCI if index not in cCI else cCI, connectedComponentIndeces, [])
        connectedComponents = []
        for connectedComponentIndex in uniqueConnectedComponentIndeces:
            connectedComponent = []
            for index in range(connectedComponentIndeces.__len__()):
                if connectedComponentIndex == connectedComponentIndeces[index]:
                    connectedComponent.append(self.N[index])
            connectedComponents.append(connectedComponent)
        return uniqueConnectedComponentIndeces.__len__(), connectedComponents

    def fit(self, trainingX, numberEpochs, modelLoaded=False):
        if (modelLoaded == False):
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

                self.error_[indexNearestUnit].assign(self.error_[indexNearestUnit] + tf.math.reduce_sum(
                    tf.math.squared_difference(xi, self.A[indexNearestUnit])))

                self.A[indexNearestUnit].assign(
                    self.A[indexNearestUnit] + self.epsilon_a * (xi - self.A[indexNearestUnit]))
                for indexNeighbour in self.N[indexNearestUnit].neighborhood:
                    self.A[indexNeighbour].assign(
                        self.A[indexNeighbour] + self.epsilon_n * (xi - self.A[indexNeighbour]))

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

                    self.A = tf.Variable(tf.concat([self.A, tf.expand_dims(
                        0.5 * (self.A[indexUnitWithMaxError_] + self.A[indexNeighbourWithMaxError_]), 0)], 0))

                    self.N.append(
                        Graph(self.A.shape[0] - 1, [indexUnitWithMaxError_, indexNeighbourWithMaxError_], [0.0, 0.0]))
                    self.N[indexUnitWithMaxError_].removeNeighbour(indexNeighbourWithMaxError_)
                    self.N[indexUnitWithMaxError_].addNeighbour(tf.constant(self.A.shape[0] - 1, dtype=tf.int64), 0.0)
                    self.N[indexNeighbourWithMaxError_].removeNeighbour(indexUnitWithMaxError_)
                    self.N[indexNeighbourWithMaxError_].addNeighbour(tf.constant(self.A.shape[0] - 1, dtype=tf.int64),
                                                                     0.0)

                    self.error_[indexUnitWithMaxError_].assign(self.error_[indexUnitWithMaxError_] * self.alpha)
                    self.error_[indexNeighbourWithMaxError_].assign(
                        self.error_[indexNeighbourWithMaxError_] * self.alpha)
                    self.error_ = tf.Variable(
                        tf.concat([self.error_, tf.expand_dims(self.error_[indexUnitWithMaxError_], 0)], 0))

                self.error_.assign(self.error_ * self.delta)
                numberProcessedRow += 1

            numberGraphConnectedComponents, _ = self.getGraphConnectedComponents()
            print("GrowingNeuralGas::numberUnits: {} - GrowingNeuralGas::numberGraphConnectedComponents: {}".format(
                self.A.shape[0], numberGraphConnectedComponents))
            GrowingNeuralGasPlotter.plotGraphConnectedComponent('.//data',
                                                                'graphConnectedComponents_' + '{}_{}'.format(
                                                                    self.A.shape[0], numberGraphConnectedComponents),
                                                                self.A, self.N, trainingX, self.getEdges())
            epoch += 1
            print("GrowingNeuralGas::epoch: {}".format(epoch))
            print("Saving model...")
            self.saveModel("model.h")

    def predict(self, X):
        if not self.clusters:
            self.countClusters()

        squared_distances = tf.reduce_sum(tf.square(X - self.A), 1)
        num_clusters = tf.reduce_max(self.clusters) + 1
        cluster_distances = [0 for i in range(num_clusters)]
        cluster_sizes = [0 for i in range(num_clusters)]
        for i in range(len(self.clusters)):
            cluster_distances[self.clusters[i]] += squared_distances[i]
            cluster_sizes[self.clusters[i]] += 1
        cluster_distances = tf.divide(cluster_distances, cluster_sizes)
        return tf.argmin(cluster_distances)

    def countClusters(self):
        self.clusters = [-1 for i in range(self.A.shape[0])]

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
                        visited[current.id] = True
                        self.clusters[current.id] = count - 1
                    for node in current.neighborhood:
                        for checkNode in self.N:
                            if tf.cast(checkNode.id, dtype=tf.int64) == node:
                                node = checkNode
                                break
                        if not visited[node.id]:
                            stack.append(node)
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
                        if len(edges) == 0 or (
                                not tf.reduce_any(tf.reduce_all(tf.equal(edge, edges), axis=(1, 2)))):
                            edges.append(edge)
                        for checkNode in self.N:
                            if tf.cast(checkNode.id, dtype=tf.int64) == node:
                                node = checkNode
                                break
                        if not visited[node.id]:
                            stack.append(node)
        return edges

    def saveModel(self, path):
        file = open(path, 'wb')
        pickle.dump(self, file)
        file.close()

    def loadModel(self, path):
        file = open(path, 'rb')
        loadedGNG = pickle.load(file)
        file.close()

        self.A = loadedGNG.A
        self.N = loadedGNG.N
        self.error_ = loadedGNG.error_
        self.epsilon_a = loadedGNG.epsilon_a
        self.epsilon_n = loadedGNG.epsilon_n
        self.a_max = loadedGNG.a_max
        self.eta = loadedGNG.eta
        self.alpha = loadedGNG.alpha
        self.delta = loadedGNG.delta
        self.maxNumberUnits = loadedGNG.maxNumberUnits
        self.clusters = loadedGNG.clusters
