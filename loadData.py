import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
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

    def train_test_more_images(self, split_pct):



def get_more_images(imgs):
    more_images = []
    vert_flip_imgs = []
    hori_flip_imgs = []

    for i in range(0, imgs.shape[0]):
        a = imgs[i, :, :, 0]
        b = imgs[i, :, :, 1]
        c = imgs[i, :, :, 2]

        av = cv2.flip(a, 1)
        ah = cv2.flip(a, 0)
        bv = cv2.flip(b, 1)
        bh = cv2.flip(b, 0)
        cv = cv2.flip(c, 1)
        ch = cv2.flip(c, 0)

        vert_flip_imgs.append(np.dstack((av, bv, cv)))
        hori_flip_imgs.append(np.dstack((ah, bh, ch)))

    v = np.array(vert_flip_imgs)
    h = np.array(hori_flip_imgs)

    more_images = np.concatenate((imgs, v, h))

    return more_images
