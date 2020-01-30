from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
from tensorflow.keras.datasets import mnist

from ray.tune.integration.keras import TuneReporterCallback

parser = argparse.ArgumentParser()
parser.add_argument(
    "--smoke-test", action="store_true", help="Finish quickly for testing")
args, _ = parser.parse_known_args()

def train_mnist(config, reporter):
    # https://github.com/tensorflow/tensorflow/issues/32159
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (Dense, Dropout, Flatten, Conv2D,
                                         MaxPooling2D, BatchNormalization)
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    batch_size = config["batch_size"]
    num_classes = 10
    epochs = 12

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    input_shape = [28, 28, 1]

    x_train = x_train/255.0
    x_test = x_test/255.0

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    # model = Sequential()
    # model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape))
    # model.add(BatchNormalization())
    # model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='Same', activation='relu'))
    # model.add(BatchNormalization())
    #
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    #
    # model.add(Conv2D(64, (3, 3), activation="relu"))
    # model.add(BatchNormalization())
    # model.add(Conv2D(64, (3, 3), activation="relu"))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.50))
    #
    # model.add(Flatten())
    # model.add(Dense(config["hidden"], activation="relu"))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.5))
    # model.add(Dense(num_classes, activation="softmax"))

    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(strides=2))
    model.add(Conv2D(filters=48, kernel_size=(5, 5), padding='valid', activation='relu'))
    model.add(MaxPooling2D(strides=2))
    model.add(Flatten())
    model.add(Dense(config["hidden"], activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adam(
            lr=config["lr"]),
        metrics=["accuracy"])

    # Data Augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1)
    train_datagen.fit(x_train)

    val_datagen = ImageDataGenerator()
    val_datagen.fit(x_test)

    model.fit_generator(
        train_datagen.flow(x_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(x_train)/batch_size,
        epochs=epochs,
        verbose=0,
        validation_data=val_datagen.flow(x_test, y_test, batch_size=256),
        callbacks=[TuneReporterCallback(reporter)])


if __name__ == "__main__":
    import ray
    from ray import tune
    from ray.tune import grid_search
    from ray.tune.schedulers import AsyncHyperBandScheduler
    mnist.load_data()  # we do this on the driver because it's not threadsafe

    ray.init()
    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration",
        metric="mean_accuracy",
        mode="max",
        max_t=4960,
        grace_period=20)

    tune.run(
        train_mnist,
        name="exp",
        scheduler=sched,
        # stop={
        #     "mean_accuracy": 0.99,
        #     "training_iteration": 5 if args.smoke_test else 300
        # },
        num_samples=5,
        resources_per_trial={
            "cpu": 4,
            "gpu": 0
        },
        config={
            "threads": 4,
            # "lr": tune.sample_from(lambda spec: np.random.uniform(0.001, 0.1)),
            # "momentum": tune.sample_from(
            #     lambda spec: np.random.uniform(0.1, 0.9)),
            # "hidden": tune.sample_from(
            #     lambda spec: np.random.randint(32, 512)),
            "lr": grid_search([5e-4]),
            "batch_size": grid_search([32, 64, 128]),
            # "momentum": grid_search([0.1, 0.9]),
            "hidden": grid_search([256]),
        })
