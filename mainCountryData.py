import csvController
from GrowingNeuralGas import GrowingNeuralGas
from GrowingNeuralGasPlotter import GrowingNeuralGasPlotter
import matplotlib as plt

epochs = 20

def test():

    #Data extraction
    countryData = csvController.load_data("Datos_Paises.csv")
    csvController.removeColumns(countryData, ["Country", "Region"])
    countryData = csvController.replaceNaNWithMedian(countryData)
#
    #Principal component analysis
    countryData = csvController.pcaByVarianceRetention(countryData, 0.95)
#
    #Transform to tensor
    countryData = csvController.fromDataFrameToTensor(countryData)
#
    growingNeuralGas = GrowingNeuralGas(eta=25)
    #growingNeuralGas.fit(countryData, epochs)
    #print(growingNeuralGas.countClusters())
    #print(growingNeuralGas.A.shape)
#
    #GrowingNeuralGasPlotter.plotNetworkStructure3D(growingNeuralGas.A, countryData, growingNeuralGas.getEdges(), title="Estructura de los datos paises.\n"
    #                                                                                                                   "PCA aplicado para obtener 3 dimensiones\n"
    #                                                                                                                    "Retención del 50% de la información")
    #GrowingNeuralGasPlotter.show()

    growingNeuralGas.loadModel("model.h")
    print(growingNeuralGas.countClusters())
    print(growingNeuralGas.predict(countryData[0]))
    print(growingNeuralGas.predict(countryData[1]))
    print(growingNeuralGas.predict(countryData[2]))
    print(growingNeuralGas.predict(countryData[3]))
    print(growingNeuralGas.predict(countryData[4]))
    print(growingNeuralGas.predict(countryData[5]))
    print(growingNeuralGas.predict(countryData[6]))
    print(growingNeuralGas.predict(countryData[7]))

test()