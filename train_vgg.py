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
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

class iceberg_model:
    def __init__(self, dataPath):
        self.dataPath = dataPath
        self.dataLoader = loader(self.dataPath)
        self.dataLoader.clean_inc_angle()

        self.model = self.vgg_model()
        self.train_test_split_val = 0.8
        self.run_weight_name = 'vgg_1221.hdf5'
        self.gen = ImageDataGenerator(horizontal_flip=True,
                                      vertical_flip=True,
                                      width_shift_range=0.,
                                      height_shift_range=0.,
                                      channel_shift_range=0,
                                      zoom_range=0.2,
                                      rotation_range=10)

        self.loss = []

    def vgg_model(self):
        # input_2 = Input(shape=[1], name="angle")
        # angle_layer = Dense(1, )(input_2)
        base_model = VGG16(weights='imagenet', include_top=False,
                           input_shape=self.dataLoader.X_train.shape[1:], classes=1)
        x = base_model.get_layer('block5_pool').output

        x = GlobalMaxPooling2D()(x)
        # merge_one = concatenate([x, angle_layer])
        merge_one = Dense(512, activation='relu', name='fc2')(x)
        merge_one = Dropout(0.3)(merge_one)
        merge_one = Dense(512, activation='relu', name='fc3')(merge_one)
        merge_one = Dropout(0.3)(merge_one)

        predictions = Dense(1, activation='sigmoid')(merge_one)

        # model = Model(input=[base_model.input, input_2], output=predictions)
        model = Model(input=base_model.input, output=predictions)
        sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='binary_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])
        return model

    def gen_flow(self, X1, X2, y):
        batch_size=64
        genX1 = self.gen.flow(X1, y, batch_size=batch_size, seed=55)
        genX2 = self.gen.flow(X1, X2, batch_size=batch_size, seed=55)
        while True:
            X1i = genX1.next()
            X2i = genX2.next()
            # yield[X1i[0], X2i[1]], X1i[1]
            yield X1i[0], X1i[1]

    def callbacks(self, wname):
        es = EarlyStopping("val_loss", patience=20, mode="min")
        msave = ModelCheckpoint(filepath=wname, save_best_only=True)
        return es, msave

    def kFoldValidation(self):
        trainImg = self.dataLoader.X_train
        # trainAngle = self.dataLoader.inc_angle
        trainLabel = self.dataLoader.labels

        n_split = 10
        kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=16)
        count = 0
        loss = []

        for train_k, test_k in kfold.split(trainImg, trainLabel):
            print('Run ' + str(count + 1) + ' out of ' + str(n_split))
            self.vgg_model()

            # generator = self.gen_flow(trainImg[train_k], trainAngle[train_k], trainLabel[train_k])
            weight_name = "vgg_1220_weights_run_%s.hdf5" % count
            es, msave = self.callbacks(wname=weight_name)

            model = self.vgg_model()

            model.fit(trainImg[train_k], trainLabel[train_k],
                      epochs=100,
                      # steps_per_epoch=24,
                      verbose=1,
                      callbacks=[es, msave])

            scores = model.evaluate(trainImg[test_k], trainLabel[test_k])
            print(scores)
            loss.append(scores[0])
            count += 1

        self.loss = loss

        print("")
        print("Length of scores: " + str(len(loss)))

        for i in range(len(loss)):
            print("Run " + str(i + 1) + ": " + str(loss[i]))

        print("Loss: " + str(np.mean(loss)), str(np.std(loss)))
        print("")

    def train(self):
        trainImg, valImg, trainLabel, valLabel = train_test_split(self.dataLoader.X_train, self.dataLoader.labels)
        model = self.vgg_model()
        es, msave = self.callbacks(wname=self.run_weight_name)
        model.fit(trainImg, trainLabel,
                  epochs=50,
                  validation_data=(valImg, valLabel),
                  verbose=1,
                  callbacks=[es,msave])

    def submission(self):
        print('Generating submission...')
        testLoader = loader('../iceberg_ship_classifier/data_test/test.json')
        self.model.load_weights(self.run_weight_name)
        pred = self.model.predict(testLoader.X_train)
        submission = pd.DataFrame()
        submission['id'] = testLoader.id
        submission['is_iceberg'] = pred.reshape((pred.shape[0]))
        submission.to_csv('sub_vgg_1221.csv', index=False)

    def submission_on_best(self):
        print('Generating submission...')
        testLoader = loader('../iceberg_ship_classifier/data_test/test.json')

        minInd = self.loss.index(np.min(self.loss))
        weight_name = "vgg_1220_weights_run_" + str(minInd) + ".hdf5"
        self.model.load_weights(weight_name)

        pred = self.model.predict(testLoader.X_train)

        submission = pd.DataFrame()
        submission['id'] = testLoader.id
        submission['is_iceberg'] = pred.reshape((pred.shape[0]))
        submission.to_csv('sub_vgg_1220.csv', index=False)


if __name__ == '__main__':
    data_path = '../iceberg_ship_classifier/data_train/train.json'
    # data_path = '../icebergClassifier/data_train/train.json'
    x = iceberg_model(data_path)
    x.train()
    x.submission()

    # x.kFoldValidation()
    # x.submission_on_best()
    # x.submission('../iceberg_ship_classifier/data_test/test.json')