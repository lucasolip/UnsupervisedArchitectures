import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from pcaDecomposition import PcaDecomposition

def load_data(dataPath):
    data = pd.read_csv(dataPath)
    return data

def removeColumns(data, cols = []):
    for col in cols:
        data.drop(col, inplace=True, axis=1)

def replaceNaNWithMedian(data):
    return data.fillna(data.median())

def fromDataFrameToTensor(data):
    return tf.convert_to_tensor(data, dtype= 'float32')

def getDataframeValues(data):
    return data.values

def getDataframeHeader(data):
    return list(data.columns)

def pcaByVarianceRetention(data, percent):
    pca = PcaDecomposition()
    pca.getDecompositionByPercentOfDataRetained(percent)
    return pca.fitDataWithoutHeader(data)

def pcaByNComponents(data, components):
    pca = PcaDecomposition()
    pca.getDecompositionByNComponentsRemaining(components)
    return pca.fitDataWithoutHeader(data)
