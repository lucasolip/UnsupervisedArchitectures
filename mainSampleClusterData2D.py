from GrowingNeuralGasPlotter import GrowingNeuralGasPlotter
from csvController import CsvController

from GrowingNeuralGas import GrowingNeuralGas

# Model
epochs = 2
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

    GrowingNeuralGasPlotter.plotNetworkStructure2D(growingNeuralGas.A, data, growingNeuralGas.getEdges())
    GrowingNeuralGasPlotter.plotClusters2D(growingNeuralGas, data)
    GrowingNeuralGasPlotter.show()

def parameter_study():
    csvController = CsvController("Sample_Cluster_Data_2D.csv", hasHeader=True)
    data, header = csvController.getCsvToTensor()

    # for l in [55, 45, 35, 25, 15, 5]:
    #     print("eta = {}".format(l))
    #     growingNeuralGas = GrowingNeuralGas(eta=l)
    #     growingNeuralGas.fit(data, epochs)
    #     print(growingNeuralGas.countClusters())
    #     print(growingNeuralGas.A.shape)
    #
    #     print("Guardando imagen de la estructura de la red...")
    #     GrowingNeuralGasPlotter.plotNetworkStructure2D(growingNeuralGas.A, data, growingNeuralGas.getEdges(),
    #                                                    title="eta = {}".format(l), save=True,
    #                                                    pathFigure=".//data", nameFigure="eta_{}".format(l))
    #     print("Imagen guardada")
    #
    # for a_max in [5, 15, 25, 35, 45, 55]:
    #     print("a_max = {}".format(a_max))
    #     growingNeuralGas = GrowingNeuralGas(eta=25, a_max=a_max)
    #     growingNeuralGas.fit(data, epochs)
    #     print(growingNeuralGas.countClusters())
    #     print(growingNeuralGas.A.shape)
    #
    #     print("Guardando imagen de la estructura de la red...")
    #     GrowingNeuralGasPlotter.plotNetworkStructure2D(growingNeuralGas.A, data, growingNeuralGas.getEdges(),
    #                                                    title="Edad máxima = {}".format(a_max), save=True,
    #                                                    pathFigure=".//data", nameFigure="a_max_{}".format(a_max))
    #     print("Imagen guardada")

    for epsilon_b in [0.05, .1, 0.2, 0.25, 0.5, 0.9]:
        print("epsilon_b = {}".format(epsilon_b))
        growingNeuralGas = GrowingNeuralGas(eta=25, epsilon_b=epsilon_b)
        growingNeuralGas.fit(data, epochs)
        print(growingNeuralGas.countClusters())
        print(growingNeuralGas.A.shape)

        print("Guardando imagen de la estructura de la red...")
        GrowingNeuralGasPlotter.plotNetworkStructure2D(growingNeuralGas.A, data, growingNeuralGas.getEdges(),
                                                       title="ε_b = {}".format(epsilon_b), save=True,
                                                       pathFigure=".//data", nameFigure="epsilon_b_{}".format(epsilon_b))
        print("Imagen guardada")

    for epsilon_n in [0.05, .1, 0.2, 0.25, 0.5, 0.9]:
        print("epsilon_n = {}".format(epsilon_n))
        growingNeuralGas = GrowingNeuralGas(eta=25, epsilon_n=epsilon_n)
        growingNeuralGas.fit(data, epochs)
        print(growingNeuralGas.countClusters())
        print(growingNeuralGas.A.shape)

        print("Guardando imagen de la estructura de la red...")
        GrowingNeuralGasPlotter.plotNetworkStructure2D(growingNeuralGas.A, data, growingNeuralGas.getEdges(),
                                                       title="ε_n = {}".format(epsilon_n), save=True,
                                                       pathFigure=".//data", nameFigure="epsilon_n_{}".format(epsilon_n))
        print("Imagen guardada")

parameter_study()