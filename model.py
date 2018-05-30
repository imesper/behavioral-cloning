import sklearn
from sklearn.model_selection import train_test_split
from keras import Sequential, optimizers
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.layers import Flatten, Dense, Conv2D, Lambda, Cropping2D, Dropout, LeakyReLU
from random import shuffle, randint
import numpy as np
import cv2
import csv
import os

# Load data
# '../DrivingData02/


def loadData(path, changePathinCSV=False):
    samples = []
    if changePathinCSV:
        with open(path + 'driving_log.csv') as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                path = line[0]
                path_l = line[1]
                path_r = line[2]
                filename = path.split('/')[-1]
                filename_l = path_l.split('/')[-1]
                filename_r = path_r.split('/')[-1]
                line[0] = path + 'IMG/' + filename
                line[1] = path + 'IMG/' + filename_l
                line[2] = path + 'IMG/' + filename_r
                samples.append(line)
    else:
        with open(path + 'driving_log.csv') as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                samples.append(line)

    samples = sklearn.utils.shuffle(samples)
    return samples


def randomDarkener(image):
    alpha = 1
    beta = randint(-30, 0)
    res = cv2.addWeighted(image, alpha, np.zeros(
        image.shape, image.dtype), 0, beta)
    return res


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    name = batch_sample[i].strip()
                    if os.path.isfile(name):
                        image = cv2.cvtColor(
                            cv2.imread(name), cv2.COLOR_BGR2RGB)
                    else:
                        image = cv2.imread(name)
                        exit(1)
                    angle = float(batch_sample[3])
                    if i == 1:
                        angle += 0.20
                    if i == 2:
                        angle -= 0.20
                    images.append(image)
                    angles.append(angle)
                    if i == 0:
                        image_flipped = np.fliplr(image)
                        images.append(image_flipped)
                        measurement_flipped = -angle
                        angles.append(measurement_flipped)
                    if randint(0, 1000) % 30 == 0:
                        image = randomDarkener(image)
                        images.append(image)
                        angles.append(angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


def CNN(train_samples, validation_samples, batch_size):

    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((60, 25), (0, 0))))
    model.add(Conv2D(24, (5, 5), strides=(2, 2)))
    model.add(LeakyReLU(alpha=.01))
    model.add(Conv2D(36, (5, 5), strides=(2, 2)))
    model.add(LeakyReLU(alpha=.01))
    model.add(Conv2D(48, (5, 5), strides=(2, 2)))
    model.add(LeakyReLU(alpha=.01))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (3, 3)))
    model.add(LeakyReLU(alpha=.01))
    model.add(Conv2D(64, (3, 3)))
    model.add(LeakyReLU(alpha=.01))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.summary()

    return model


def main():
    samples = []
    # Loading Data from 3 different folders
    # Each folder has different runs on simulator
    samples += loadData('../DrivingData/')
    samples += loadData('../DrivingData02/')
    samples += loadData('data/')
    print(len(samples))
    # Spliting the data between trainnig (80%) and validation (20%)
    train_samples, validation_samples = train_test_split(
        samples, test_size=0.2)

    # Setting the batch size
    batch_size = 32
    # Getting the model
    model = CNN(train_samples, validation_samples, batch_size)

    # Running the model, saving only the best models based on validation loss
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='auto')

    model.compile(loss='mse', optimizer=Adam(lr=1.0e-4))

    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)

    model.fit_generator(train_generator, steps_per_epoch=len(train_samples)/batch_size, validation_data=validation_generator,
                        validation_steps=len(validation_samples)/batch_size, epochs=30, callbacks=[checkpoint])


if __name__ == '__main__':
    main()
