from loadData import loader
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, Dropout, Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint



class iceberg_model:
    def __init__(self, dataPath):
        self.dataPath = dataPath
        self.model = Sequential()

    def create_model(self):
        # Conv Layer 1
        self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(75, 75, 1)))
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

    def callbacks(self, fname):
        stop = EarlyStopping('val_loss', mode='min')
        check = ModelCheckpoint(fname, save_best_only=True)
        return stop, check

    def train_model(self):
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0)
        self.create_model()
        self.model.compile(loss='binary_crossentropy',
                           optimizer=opt,
                           metrics=['accuracy'])

        dataLoader = loader(self.dataPath)
        trainImg, valImg, trainLabels, valLabels = dataLoader.train_test_split(0.8)

        earlyStop, modelCheck = self.callbacks('model_weights.hdf5')

        self.model.fit(trainImg, trainLabels,
                       batch_size=24,
                       epochs=50,
                       verbose=1,
                       validation_data=(valImg, valLabels),
                       callbacks=[modelCheck])


if __name__ == '__main__':
    data_path = '../icebergClassifier/data_train/train.json'
    x = iceberg_model(data_path)
    x.train_model()
