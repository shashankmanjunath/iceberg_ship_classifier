import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class loader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.json_data = pd.read_json(data_path)
        self.band_1 = []
        self.band_2 = []
        self.id = []
        self.labels = []
        self.load_data()

        self.np_data_1 = np.zeros((self.band_1[0].shape[0], self.band_1[0].shape[1], len(self.band_1)))
        self.np_data_2 = np.zeros((self.band_1[0].shape[0], self.band_1[0].shape[1], len(self.band_1)))
        self.to_np_array()

    def load_data(self):
        for i in range(len(self.json_data)):
            self.band_1 += [np.asarray(self.json_data['band_1'][i]).reshape((75, 75))]
            self.band_2 += [np.asarray(self.json_data['band_2'][i]).reshape((75, 75))]
            self.id += [self.json_data['id'][i]]
            self.labels += [self.json_data['is_iceberg'][i]]

    def train_test_split(self, split_pct): # Only loads np_data_1
        split_num = int(len(self.labels)*split_pct)

        trainImg = np.zeros((split_num, self.np_data_1.shape[0], self.np_data_1.shape[1], 1))
        valImg = np.zeros((len(self.labels) - split_num, self.np_data_1.shape[0], self.np_data_1.shape[1], 1))

        trainLabel = np.zeros((split_num, 1))
        valLabel = np.zeros((len(self.labels) - split_num, 1))

        for i in range(split_num):
            trainImg[i, :, :, 0] = self.np_data_1[:, :, i]
            trainLabel[i] = self.labels[i]

        count = 0

        for i in range(split_num, len(self.labels)):
            valImg[count, :, :, 0] = self.np_data_1[:, :, i]
            trainLabel[count] =  self.labels[i]
            count += 1

        print(trainImg.shape)
        print(' ')

        return trainImg, valImg, trainLabel, valLabel

    def to_np_array(self):
        for i in range(len(self.band_1)):
            self.np_data_1[:, :, i] = np.asarray(self.band_1[i])
            self.np_data_2[:, :, i] = np.asarray(self.band_2[i])

    def normalize_data(self):
        for i in range(self.np_data_1.shape[0]):
            for j in range(self.np_data_1.shape[1]):
                for k in range(self.np_data_1.shape[2]):
                    self.np_data_1[i, j, k] = l_function(self.np_data_1[i, j, k])
                    self.np_data_2[i, j, k] = l_function(self.np_data_2[i, j, k])


def l_function(x): # Bentes et al, Ship-Iceberg Discrimination with Convolutional Neural Networks
    if x > 1:
        return 1 + np.log(x)
    else:
        return x


if __name__ == '__main__':
    x = loader('../icebergClassifier/data_train/train.json')
    x.to_np_array()
    x.normalize_data()

    print(np.sum(x.band_1[0] == x.np_data_1[:, :, 0])/float(75*75))
