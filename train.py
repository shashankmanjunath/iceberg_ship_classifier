from loadData import loader
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, Dropout, Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedKFold


seed = 1337
np.random.seed(seed)


class iceberg_model:
    def __init__(self, dataPath):
        self.dataPath = dataPath
        self.dataLoader = []
        self.model = []
        self.train_test_split_val = 0.8
        self.run_weight_name = 'model_weights_1214_up_3.hdf5'

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

        # self.model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
        # self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # self.model.add(Dropout(0.2))

        # self.model.add(Conv2D(384, kernel_size=(3, 3), activation='relu'))
        # self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # self.model.add(Dropout(0.2))

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

        opt = Adam(lr=0.001)

        self.model.compile(loss='binary_crossentropy',
                           optimizer=opt,
                           metrics=['accuracy'])

    def callbacks(self):
        stop = EarlyStopping(monitor='val_loss', patience=7, mode='min')
        check = ModelCheckpoint(self.run_weight_name, save_best_only=True)
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, epsilon=1e-4,
                                           mode='min')
        return stop, check, reduce_lr_loss

    def train_model(self):
        print('Training Model...')
        self.create_model()

        self.dataLoader = loader(self.dataPath)
        trainImg, valImg, trainLabels, valLabels = self.dataLoader.train_test_split(self.train_test_split_val)

        earlyStop, modelCheck, reduce = self.callbacks(self.run_weight_name)

        datagen = ImageDataGenerator(horizontal_flip=True,
                                     vertical_flip=True,
                                     width_shift_range=0.3,
                                     height_shift_range=0.3,
                                     zoom_range=0.1,
                                     rotation_range=20)

        val_datagen = ImageDataGenerator(horizontal_flip=True,
                                         vertical_flip=True,
                                         width_shift_range=0.3,
                                         height_shift_range=0.3,
                                         zoom_range=0.1,
                                        rotation_range=20)

        history = self.model.fit_generator(datagen.flow(trainImg, trainLabels),
                                           epochs=2000,
                                           steps_per_epoch=1,
                                           verbose=1,
                                           validation_data=(valImg, valLabels),
                                           callbacks=[modelCheck, earlyStop, reduce])

        # plt.figure()
        # plt.plot(history.history['acc'])
        # plt.plot(history.history['val_acc'])
        # plt.title('Model Accuracy')
        # plt.xlabel('Epoch')
        # plt.ylabel('Accuracy')
        # plt.legend(['Train', 'Test'], loc='upper left')
        # plt.show()

    def test_model(self, wpath):
        print('Testing model...')
        if not self.dataLoader:
            self.dataLoader = loader(self.dataPath)
        _, valImg, _, valLabels = self.dataLoader.train_test_split(self.train_test_split_val)

        if not self.model:
            self.create_model()

        self.model.load_weights(filepath=wpath)
        score = self.model.evaluate(valImg, valLabels, verbose=1)
        print('Test loss: ', score[0])
        print('Test accuracy: ', score[1])
        loss, accuracy = score[0], score[1]
        return loss, accuracy

    def kFoldValidation(self):
        print('Validating Model...')
        if not self.dataLoader:
            self.dataLoader = loader(self.dataPath)

        _, valImg, _, valLabels = self.dataLoader.train_test_split(self.train_test_split_val)

        earlyStop, _, _ = self.callbacks(self.run_weight_name)
        n_split = 5
        kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=seed)
        loss = []
        count = 0

        for train, test in kfold.split(valImg, valLabels):
            print('Run ' + str(count + 1) + ' out of ' + str(n_split))
            self.create_model()
            self.model.fit(valImg[train], valLabels[train],
                           epochs=50,
                           batch_size=1,
                           verbose=1,
                           callbacks=[earlyStop])

            scores = self.model.evaluate(valImg[test], valLabels[test])
            loss.append(scores[0])
            count += 1

        print("")
        print("Length of scores: " + str(len(loss)))

        for i in range(len(loss)):
            print("Run " + str(i+1) + ": " + str(loss[i]))

        print("Loss: " + str(np.mean(loss), str(np.std(loss))))
        print("")
        return 0

    def submission(self, testpath, wname):
        print('Generating submission...')
        testLoader = loader(testpath)
        testImg = np.zeros((len(testLoader.json_data), 75, 75, 3))

        for i in range(len(testLoader.json_data)):
            testImg[i, :, :, 0] = testLoader.band_1_norm[:, :, i]
            testImg[i, :, :, 1] = testLoader.band_2_norm[:, :, i]
            testImg[i, :, :, 2] = testLoader.band_3_norm[:, :, i]

        if not self.model:
            self.create_model()

        self.model.load_weights(wname)

        pred = self.model.predict(testImg, verbose=1)

        submission = pd.DataFrame()
        submission['id'] = testLoader.id
        submission['is_iceberg'] = pred.reshape((pred.shape[0]))
        submission.to_csv('sub_' + self.run_weight_name[:-5] + '.csv', index=False)


if __name__ == '__main__':
    data_path = '../iceberg_ship_classifier/data_train/train.json'
    x = iceberg_model(data_path)
    # x.train_model()
    x.kFoldValidation()
    # x.test_model('../iceberg_ship_classifier/' + x.run_weight_name)
    # x.submission('../iceberg_ship_classifier/data_test/test.json', '../iceberg_ship_classifier/' + x.run_weight_name)

