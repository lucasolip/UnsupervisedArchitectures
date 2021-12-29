import csvController
from GrowingNeuralGas import GrowingNeuralGas
import matplotlib as plt

epochs = 10

def test():

    #Data extraction
    countryData = csvController.load_data("Datos_Paises.csv")
    csvController.removeColumns(countryData, ["Country", "Region"])
    countryData = csvController.replaceNaNWithMedian(countryData)

    #Principal component analysis
    countryData = csvController.pcaByVarianceRetention(countryData, 0.95)

    #Transform to tensor
    countryData = csvController.fromDataFrameToTensor(countryData)

    growingNeuralGas = GrowingNeuralGas(eta=25)
    growingNeuralGas.fit(countryData, epochs)
    print(growingNeuralGas.countClusters())
    print(growingNeuralGas.A.shape)

test()