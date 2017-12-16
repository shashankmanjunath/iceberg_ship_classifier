import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt

class loader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.json_data = pd.read_json(data_path)
        self.band_1 = np.zeros((75, 75, len(self.json_data)))
        self.band_2 = np.zeros((75, 75, len(self.json_data)))
        self.id = []
        self.labels = []
        self.load_data()
        self.band_3 = self.band_1 + self.band_2

        self.band_1_norm = (self.band_1 - np.mean(self.band_1))/(np.max(self.band_1) - np.min(self.band_1))
        self.band_2_norm = (self.band_2 - np.mean(self.band_2))/(np.max(self.band_2) - np.min(self.band_2))
        self.band_3_norm = (self.band_3 - np.mean(self.band_3))/(np.max(self.band_3) - np.min(self.band_3))

    def load_data(self):
        for i in range(len(self.json_data)):
            self.band_1[:, :, i] = np.asarray(self.json_data['band_1'][i]).reshape((75, 75))
            self.band_2[:, :, i] = np.asarray(self.json_data['band_2'][i]).reshape((75, 75))
            self.id += [self.json_data['id'][i]]

            if 'is_iceberg' in self.json_data.keys():
                self.labels += [self.json_data['is_iceberg'][i]]

    def train_test_split(self, split_pct):
        imgStack = np.zeros((len(self.labels), 75, 75, 3))
        for i in range(len(self.labels)):
            imgStack[i, :, :, 0] = self.band_1_norm[:, :, i]
            imgStack[i, :, :, 1] = self.band_2_norm[:, :, i]
            imgStack[i, :, :, 2] = self.band_3_norm[:, :, i]

        trainImg, valImg, trainLabel, valLabel = train_test_split(imgStack, self.labels, random_state=1,
                                                                  train_size=split_pct)
        trainLabel = np.asarray(trainLabel)
        valLabel = np.asarray(valLabel)
        return trainImg, valImg, trainLabel, valLabel
