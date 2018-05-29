import cv2
import csv
import os
import sklearn
from sklearn.model_selection import train_test_split
from keras import Sequential, optimizers
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam 
from keras.layers import Flatten, Dense, Conv2D, Lambda, Cropping2D, Dropout, LeakyReLU
from random import shuffle
import numpy as np

samples = []
with open('../DrivingData/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        path = line[0]
        path_l = line[1]
        path_r = line[2]
        filename = path.split('/')[-1]
        filename_l = path_l.split('/')[-1]
        filename_r = path_r.split('/')[-1]
        line[0] = '../DrivingData/IMG/'+filename
        line[1] = '../DrivingData/IMG/'+filename_l
        line[2] = '../DrivingData/IMG/'+filename_r
        #print(line)
        samples.append(line)

with open('../DrivingData03/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        path = line[0]
        path_l = line[1]
        path_r = line[2]
        filename = path.split('/')[-1]
        filename_l = path_l.split('/')[-1]
        filename_r = path_r.split('/')[-1]
        line[0] = '../DrivingData03/IMG/'+filename
        line[1] = '../DrivingData03/IMG/'+filename_l
        line[2] = '../DrivingData03/IMG/'+filename_r
        #print(line)
        samples.append(line)

with open('../DrivingData02/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        path = line[0]
        path_l = line[1]
        path_r = line[2]
        filename = path.split('/')[-1]
        filename_l = path_l.split('/')[-1]
        filename_r = path_r.split('/')[-1]
        line[0] = '../DrivingData02/IMG/'+filename
        line[1] = '../DrivingData02/IMG/'+filename_l
        line[2] = '../DrivingData02/IMG/'+filename_r
        samples.append(line)

with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

samples  = sklearn.utils.shuffle(samples)
#images = []
#measurements = []
#for line in samples:
#    file_path = line[0]
#    image = cv2.imread(file_path)
#    images.append(image)
#    measurement = float(line[3])
#    measurements.append(measurement)
#    image_flipped = np.fliplr(image)
#    images.append(image_flipped)
#    measurement_flipped = -measurement
#    measurements.append(measurement_flipped)


##train_samples, validation_samples, train_meas, validation_meas = train_test_split(images, measurements, test_size=0.2)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
# import cv2
# import numpy as np
# import sklearn
import random
def randomDarkener(image):
    alpha = random.randint(0,2) 
    beta = random.randint(-30,0)
    res = cv2.addWeighted(image, alpha, np.zeros(image.shape, image.dtype), 0, beta)
    return res

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    name = batch_sample[i].strip()
                    if os.path.isfile(name):
                        image = cv2.cvtColor(cv2.imread(name),cv2.COLOR_BGR2RGB)
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
                    if random.randint(0, 1000) % 30 == 0:
                       image = randomDarkener(image)
                       images.append(image)
                       angles.append(angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

batch_size = 32
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)
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
#model.add(Dropout(0.8))
model.add(Dropout(0.5))
#model.add(Dense(400))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='auto')

model.compile(loss='mse', optimizer=Adam(lr=1.0e-4))

model.fit_generator(train_generator, steps_per_epoch=len(train_samples)/batch_size, validation_data=validation_generator, validation_steps=len(validation_samples)/batch_size, epochs=30, callbacks=[checkpoint])
