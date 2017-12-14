import pandas as pd
import numpy as np


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
            self.labels += [self.json_data['is_iceberg'][i]]

    def train_test_split(self, split_pct): # Only loads np_data_1
        split_num = int(len(self.labels)*split_pct)

        trainImg = np.zeros((split_num, 75, 75, 3))
        valImg = np.zeros((len(self.labels) - split_num, 75, 75, 3))

        trainLabel = np.zeros((split_num, 1))
        valLabel = np.zeros((len(self.labels) - split_num, 1))

        for i in range(split_num):
            trainImg[i, :, :, 0] = self.band_1_norm[:, :, i]
            trainImg[i, :, :, 1] = self.band_2_norm[:, :, i]
            trainImg[i, :, :, 2] = self.band_3_norm[:, :, i]
            trainLabel[i] = self.labels[i]

        count = 0

        for i in range(split_num, len(self.labels)):
            valImg[count, :, :, 0] = self.band_1_norm[:, :, i]
            valImg[count, :, :, 1] = self.band_2_norm[:, :, i]
            valImg[count, :, :, 2] = self.band_3_norm[:, :, i]
            trainLabel[count] = self.labels[i]
            count += 1

        return trainImg, valImg, trainLabel, valLabel

if __name__ == '__main__':
    x = loader('../icebergClassifier/data_train/train.json')
