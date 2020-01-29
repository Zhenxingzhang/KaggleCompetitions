import numpy as np  # linear algebra
import math
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import LearningRateScheduler


def lenet_5():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPool2D(strides=2))

    model.add(Conv2D(filters=48, kernel_size=(5, 5), padding='valid', activation='relu'))
    model.add(MaxPool2D(strides=2))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.50))
    model.add(Dense(84, activation='relu'))
    # model.add(Dropout(0.25))
    model.add(Dense(10, activation='softmax'))
    return model


# learning rate schedule
def step_decay(epoch):
    initial_lrate = 5e-4
    drop = 0.5
    epochs_drop = 5.0
    return initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))


if __name__ == "__main__":
    df_train = pd.read_csv('data/train.csv')
    X_train = df_train.iloc[:, 1:]
    Y_train = df_train.iloc[:, 0]

    X_train = X_train / 255.0

    print(X_train.shape)

    # Train-Test Split
    X_dev, X_val, Y_dev, Y_val = train_test_split(X_train, Y_train, test_size=0.03, shuffle=True, random_state=2019)

    # Reshape the input
    X_dev = X_dev.values.reshape(X_dev.shape[0], 28, 28, 1)
    X_val = X_val.values.reshape(X_val.shape[0], 28, 28, 1)

    model = lenet_5()
    adam = Adam(lr=5e-4)
    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer=adam)

    # Data Augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1)
    train_datagen.fit(X_dev)

    val_datagen = ImageDataGenerator()
    val_datagen.fit(X_val)

    # learning schedule callback
    lr_sche = LearningRateScheduler(step_decay)

    checkpoint = ModelCheckpoint("saved_models/model-{val_accuracy:.4f}.h5",
                                 monitor="val_accuracy", verbose=1, save_best_only=True,
                                 save_weights_only=True, mode="max")

    model.fit_generator(
        train_datagen.flow(X_dev, Y_dev, batch_size=128),
        steps_per_epoch=len(X_dev) / 128,
        epochs=30,
        validation_data=val_datagen.flow(X_val, Y_val, batch_size=256),
        callbacks=[checkpoint, lr_sche]
    )

    print("training completed!")
