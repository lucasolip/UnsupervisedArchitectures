import csv
import tensorflow as tf

class CsvController(object):

    def __init__(self, directory, hasHeader = True):
        self.file = open(directory)
        self.fileType = type(self.file)
        self.csvreader = csv.reader(self.file)
        self.hasHeader = hasHeader

    def __del__(self):
        self.file.close()

    #shape=(150, 3), dtype=float32)
    def getCsvToTensor(self):
        rows = []
        header = []
        if self.hasHeader:
            header = next(self.csvreader)
        for row in self.csvreader:
            processedRow = []
            for col in row:
                processedRow.append(float(col))
            rows.append(processedRow)

        tensor = tf.convert_to_tensor(value = rows, dtype = 'float32')
        return tensor, header