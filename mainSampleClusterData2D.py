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



test()