import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import cv2
import keras
from keras.applications .vgg19 import VGG19, preprocess_input, decode_predictions
#from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential, Model
from keras.preprocessing.image import load_img, img_to_array
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import image

WEIGHTS = '/media/soupayan/DATA/Projects/Pycharm/AnalyticsVidya/my_weights.model'
IMG_PATH = '/media/soupayan/DATA/Projects/Pycharm/AnalyticsVidya/Data/Train/'
batch_size = 170
validation_split = .2

data_gen = image.ImageDataGenerator(rotation_range=10,
                                    shear_range=0.1,
                                    horizontal_flip=False,
                                    vertical_flip=False,
                                    zoom_range=0.1,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    fill_mode='nearest',
                                    rescale=1./255,
                                    validation_split=validation_split
                                    )

train_generator = data_gen.flow_from_directory(IMG_PATH, target_size=(32, 32), subset='training', batch_size=batch_size)
valid_generator = data_gen.flow_from_directory(IMG_PATH, target_size=(32, 32), subset='validation',
                                               batch_size=batch_size)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=(32, 32, 3)))
model.add(BatchNormalization(axis=1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization(axis=1))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization(axis=1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization(axis=1))
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization(axis=1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dropout(0.5, seed=100))
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.2, seed=100))
model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5, seed=100))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5, seed=100))
model.add(Dense(46, activation='softmax'))

model.summary()


def train(model, weights=None):
    if weights and os.path.exists(WEIGHTS):
        print("weights found. Loading the weights")
        model.load_weights(weights)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    else:
        print("weights not found. Hence training the model")
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit_generator(train_generator, epochs=50, verbose=1, steps_per_epoch=(78200*validation_split)/batch_size,
                            validation_steps=(78200*validation_split)/batch_size,
                            validation_data=valid_generator, use_multiprocessing=True)
        model.save_weights('my_weights.model')
    return model


# model = train(model)
model = train(model, weights=WEIGHTS)

evaluation = model.evaluate_generator(generator=valid_generator,
                                      steps=(78200*validation_split)/batch_size)


test_images = pd.read_csv('/media/soupayan/DATA/Projects/Pycharm/AnalyticsVidya/Data/test_X.csv')


def to_rgb(im):
    im = np.array(im)
    im.resize((32, 32, 1))
    return np.repeat(im.astype(np.float32), 3, 2)


def get_image_for_prediction(index):
    im = test_images.iloc[index]
    im = im.divide(255)
    im = to_rgb(im)
    im = np.expand_dims(im, axis=0)
    return im


def show_image(im):
    # im = to_rgb5(im)
    try:
        cv2.imshow('image', im)
        k = cv2.waitKey(5000)
    finally:
        cv2.destroyAllWindows()


mappings = {v: k for k, v in train_generator.class_indices.items()}


def predict_from_dataframe(df):
    pred = []
    for index in df:
        prediction = model.predict_classes(get_image_for_prediction(int(index)))
        pred.append(mappings[prediction[0]])
    return pred


predictions = predict_from_dataframe(test_images)
predictions = pd.DataFrame({'Label': predictions})
predictions.to_excel('predictions.xlsx', index=False)
