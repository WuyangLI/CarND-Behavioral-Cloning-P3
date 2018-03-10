import csv
import argparse

import numpy as np
import cv2
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Conv2D, Dropout, Dense, Flatten, Cropping2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2


NUM=61524
#the number of samples in the batch
NUM_BATCH_SAMPLE = 128
#the number of samples in the epoch
NUM_EPOCH_SAMPLE = 0.8*NUM
#steps per epoch
NUM_BATCHES = NUM_EPOCH_SAMPLE/NUM_BATCH_SAMPLE
# number of epoches
NUM_EPOCHES = 10
# validation step
NUM_VALIDATION_SAMPLE = 0.2*NUM
NUM_VALIDATION_STEPS = NUM_VALIDATION_SAMPLE/NUM_BATCH_SAMPLE


IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 160, 320, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


def navidia_model(keep_prob):
    print("keep probability is {}".format(keep_prob))
    model = Sequential()
    # crop the images
    model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=INPUT_SHAPE))
    # normalization layer
    model.add(BatchNormalization())
    # convolutional layers
    model.add(Conv2D(24, (5, 5), activation='elu', strides=(2,2)))
    model.add(Conv2D(36, (5, 5), activation='elu', strides=(2,2)))
    model.add(Conv2D(48, (5, 5), activation='elu', strides=(2,2)))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    # fully connected layers
    model.add(Flatten())
    model.add(Dropout(keep_prob))
    model.add(Dense(100, activation='elu'))
    model.add(Dropout(keep_prob))
    model.add(Dense(50, activation='elu'))
    model.add(Dropout(keep_prob))
    model.add(Dense(10, activation='elu', kernel_regularizer=l2(0.02)))
    model.add(Dense(1))
    return model


def generator(samples, batch_size=NUM_BATCH_SAMPLE):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = batch_sample[0]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[1])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)


def get_train_valid_samples(log_path):
    samples = []
    with open(log_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    return train_samples, validation_samples


def train_model(model, train_gen, valid_gen):
    print("start training model")
    checkpoint = ModelCheckpoint('model{epoch:02d}.h5')
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')
    model.fit_generator(train_gen, steps_per_epoch=NUM_BATCHES, epochs=NUM_EPOCHES, verbose=1, validation_data=valid_gen,
                  validation_steps=NUM_VALIDATION_STEPS, callbacks=[checkpoint])
    print(model.summary())
    print("save model")
    model.save('./model.h5')


def main(keep_prob):
    print("generate training and validation samples")
    train_samples, validation_samples = get_train_valid_samples('./processed_log.csv')
    train_gen = generator(train_samples)
    valid_gen = generator(validation_samples)
    model = navidia_model(keep_prob)
    train_model(model, train_gen, valid_gen)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tune keep prob')
    parser.add_argument('--keep_prob', type=float, default=0.5, help='keep probability for dropout')

    args = parser.parse_args()
    main(args.keep_prob)
