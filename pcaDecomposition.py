from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class PcaDecomposition(object):
    def __init__(self):
        pass

    def getDecompositionByNComponentsRemaining(self, NComponents):
        self.pca = PCA(NComponents)

    def getDecompositionByPercentOfDataRetained(self, percent):
        self.pca = PCA(percent)

    def fitDataWithoutHeader(self, data):
        X = StandardScaler().fit_transform(data)
        principalComponents = self.pca.fit_transform(X)
        return principalComponents