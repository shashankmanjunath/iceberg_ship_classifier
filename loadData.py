import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
# import matplotlib.pyplot as plt

class loader:
    def __init__(self, data_path, split_pct=0.75):
        self.split_pct = split_pct
        self.data_path = data_path
        self.json_data = pd.read_json(data_path)
        self.X_band_1 = np.asarray([np.array(img).astype(np.float32).reshape(75, 75) for img
                                    in self.json_data['band_1']])
        self.X_band_2 = np.asarray([np.array(img).astype(np.float32).reshape(75, 75) for img
                                    in self.json_data['band_2']])
        self.X_train = np.concatenate([self.X_band_1[:, :, :, np.newaxis], self.X_band_2[:, :, :, np.newaxis],
                                     ((self.X_band_1 + self.X_band_2) / 2)[:, :, :, np.newaxis]], axis=-1)
        self.id = self.json_data['id']
        self.inc_angle = pd.to_numeric(self.json_data['inc_angle'], errors='coerce')
        self.inc_angle = np.asarray(self.inc_angle.fillna(method='pad'))

        if 'is_iceberg' in self.json_data.keys():
            self.labels = np.asarray(self.json_data['is_iceberg'])

    def clean_inc_angle(self):
        ind = np.where(self.inc_angle > 0)
        self.X_train = self.X_train[ind[0]]
        if 'is_iceberg' in self.json_data.keys():
            self.labels = self.labels[ind[0]]

    def train_test_split(self):
        trainImg, valImg, trainLabel, valLabel, trainAngle, valAngle = train_test_split(self.X_train,
                                                                                        self.labels,
                                                                                        self.inc_angle,
                                                                                        random_state=1,
                                                                                        train_size=self.split_pct)
        return trainImg, valImg, trainLabel, valLabel, trainAngle, valAngle

    def train_test_more_images(self):
        trainImg, valImg, trainLabel, valLabel, trainAngle, valAngle = self.train_test_split()

        trainImg_more = get_more_images(trainImg)
        valImg_more = get_more_images(valImg)

        trainLabel_more = np.concatenate([trainLabel, trainLabel, trainLabel])
        valLabel_more = np.concatenate([valLabel, valLabel, valLabel])

        trainAngle_more = np.concatenate([trainAngle, trainAngle, trainAngle])
        valAngle_more = np.concatenate([valAngle, valAngle, valAngle])

        return trainImg_more, valImg_more, trainLabel_more, valLabel_more, trainAngle_more, valAngle_more

    def median_filter(self):
        for i in range(self.X_train.shape[0]):
            self.X_train[i, :, :, 0] = cv2.medianBlur(self.X_train[i, :, :, 0], 3)
            self.X_train[i, :, :, 1] = cv2.medianBlur(self.X_train[i, :, :, 1], 3)
            self.X_train[i, :, :, 2] = cv2.medianBlur(self.X_train[i, :, :, 2], 3)



def get_more_images(imgs):
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
