from loadData import loader
import pandas as pd
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, Dropout, Activation, GlobalMaxPooling2D
from keras.layers import concatenate
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
from keras.applications.vgg16 import VGG16


class iceberg_model:
    def __init__(self, dataPath):
        self.dataPath = dataPath
        self.dataLoader = loader(self.dataPath)
        self.dataLoader.clean_inc_angle()

        self.model = []
        self.train_test_split_val = 0.8
        self.run_weight_name = 'model_weights_1219.hdf5'

    def create_model(self):
        # Conv Layer 1
        self.model = Sequential()
        self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(75, 75, 3)))
        self.model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        self.model.add(Dropout(0.2))

        # Conv Layer 2
        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Dropout(0.2))

        # Conv Layer 3
        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Dropout(0.2))

        # Conv Layer 4
        self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Dropout(0.2))

        # Flatten the data for upcoming dense layers
        self.model.add(Flatten())

        # Dense Layers
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))

        # Dense Layer 2
        self.model.add(Dense(256))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))

        # Sigmoid Layer
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))

        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        self.model.compile(loss='binary_crossentropy',
                           optimizer=opt,
                           metrics=['accuracy'])

    def callbacks(self):
        stop = EarlyStopping("val_loss", patience=10, mode="min")
        check = ModelCheckpoint(self.run_weight_name, save_best_only=True)
        return [stop, check]

    def train_model(self):
        print('Training Model...')
        self.create_model()
        trainImg, valImg, trainLabels, valLabels, _, _ = self.dataLoader.train_test_more_images()


        self.model.fit(trainImg, trainLabels,
                       epochs=50,
                       validation_data=(valImg, valLabels),
                       verbose=1,
                       callbacks=self.callbacks())

    def test_model(self):
        print('Testing model...')

        _, valImg, _, valLabels, _, _ = self.dataLoader.train_test_more_images()

        if not self.model:
            self.create_model()

        self.model.load_weights(self.run_weight_name)
        score = self.model.evaluate(valImg, valLabels, verbose=1)
        print('Test loss: ', score[0])
        print('Test accuracy: ', score[1])
        loss, accuracy = score[0], score[1]
        return loss, accuracy

    def kFoldValidation(self):
        print('Validating Model...')

        trainImg, valImg, trainLabel, valLabel, _, _ = self.dataLoader.train_test_more_images()

        n_split = 10
        kfold = StratifiedKFold(n_splits=n_split, shuffle=True)
        loss = []
        count = 0

        for train_k, test_k in kfold.split(trainImg, trainLabel):
            print('Run ' + str(count + 1) + ' out of ' + str(n_split))
            self.create_model()

            self.model.fit(trainImg[train_k], trainLabel[train_k],
                           epochs=100,
                           validation_data=(valImg, valLabel),
                           verbose=1,
                           callbacks=self.callbacks())

            scores = self.model.evaluate(trainImg[test_k], trainLabel[test_k])
            print(scores)
            loss.append(scores[0])
            count += 1

        print("")
        print("Length of scores: " + str(len(loss)))

        for i in range(len(loss)):
            print("Run " + str(i+1) + ": " + str(loss[i]))

        print("Loss: " + str(np.mean(loss)), str(np.std(loss)))
        print("")
        return 0

    def submission(self, testpath):
        print('Generating submission...')
        testLoader = loader(testpath)

        if not self.model:
            self.create_model()

        self.model.load_weights(self.run_weight_name)

        pred = self.model.predict_proba(testLoader.X_train)

        submission = pd.DataFrame()
        submission['id'] = testLoader.id
        submission['is_iceberg'] = pred.reshape((pred.shape[0]))
        submission.to_csv('sub_' + self.run_weight_name[:-5] + '.csv', index=False)


if __name__ == '__main__':
    # data_path = '../iceberg_ship_classifier/data_train/train.json'
    data_path = '../icebergClassifier/data_train/train.json'
    x = iceberg_model(data_path)
    # x.train_model()
    # x.test_model()
    x.kFoldValidation()
    # x.submission('../iceberg_ship_classifier/data_test/test.json')

