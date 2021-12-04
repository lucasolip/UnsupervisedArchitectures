import unittest

from Graph import Graph

class ut_Graph(unittest.TestCase):
    def test_addNeighborhood(self):
        graphBase = Graph(7, [1, 2, 3], [10, 20, 30])

        graphTest = Graph(7)
        graphTest.addNeighbour(1, 10)
        graphTest.addNeighbour(2, 20)
        graphTest.addNeighbour(3, 30)

        self.assertEqual(graphBase, graphTest)

    def test_removeNeighborhood(self):
        graphBase = Graph(7, [1, 3], [10, 30])

        graphTest = Graph(7, [1, 2, 3], [10, 20, 30])
        graphTest.removeNeighbour(2)

        self.assertEqual(graphBase, graphTest)

    def test_incrementAgeNeighborhood(self):
        graphBase = Graph(7, [1, 2, 3], [11, 21, 31])

        graphTest = Graph(7, [1, 2, 3], [10, 20, 30])
        graphTest.incrementAgeNeighborhood(1)

        self.assertEqual(graphBase, graphTest)

    def test_incrementAgeNeighborhood_when_graph_have_not_neighborhood(self):
        graphBase = Graph(7, [], [])

        graphTest = Graph(7, [], [])
        graphTest.incrementAgeNeighborhood(1)

        self.assertEqual(graphBase, graphTest)

    def test_pruneGraph(self):
        graphBase = Graph(7, [1, 2, 3, 4, 5], [25, 20, 30, 25, 20])

        graphTest_30 = Graph(7, [1, 2, 4, 5], [25, 20, 25, 20])
        graphTest_25 = Graph(7, [2, 5], [20, 20])

        graphBase.pruneGraph(30)

        self.assertEqual(graphBase, graphTest_30)

        graphBase.pruneGraph(25)

        self.assertEqual(graphBase, graphTest_25)

if __name__ == '__main__':
    unittest.main()