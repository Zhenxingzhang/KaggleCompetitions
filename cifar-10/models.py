from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras import regularizers

CLASSES_NUM = 10
INPUT_SHAPE = [32, 32, 3]


# Model from here: https://www.kaggle.com/c/cifar-10/discussion/40237
def Convnet():
    base_filter_num = 32
    weight_decay = 1e-4
    model = Sequential()
    model.add(Conv2D(base_filter_num, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),
                     input_shape=INPUT_SHAPE))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(base_filter_num, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(2 * base_filter_num, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(2 * base_filter_num, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(4 * base_filter_num, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(4 * base_filter_num, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(CLASSES_NUM, activation='softmax'))
    return model


def standard_cnn():
    cnn_model = Sequential([
        Conv2D(32, (3, 3), padding='same',
               input_shape=[32, 32, 3]),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(32, (3, 3)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(64, (3, 3)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(512),
        BatchNormalization(),
        Dense(10),
        BatchNormalization(),
        Activation('softmax')
    ])

    return cnn_model
