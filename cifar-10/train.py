import tensorflow_datasets as tfds
import tensorflow as tf
from datetime import datetime
from models import Convnet, standard_cnn
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorboard.plugins.hparams import api as hp


if __name__=="__main__":
    # Load the raw CIFAR-10 data
    train_ds, test_ds = tfds.load("cifar10",
                                  split=[tfds.Split.TRAIN, tfds.Split.TEST],
                                  batch_size=-1)

    train_images = train_ds["image"]
    test_images = test_ds["image"]

    train_labels = train_ds["label"]
    test_labels = test_ds["label"]

    x_train = tf.dtypes.cast(train_images, tf.float32)
    x_test = tf.dtypes.cast(test_images, tf.float32)

    x_train /= 255
    x_test /= 255

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    data_augmentation = True

    # define model architecture
    model = standard_cnn()

    # define optimizer
    sgd = SGD(lr=0.001, decay=1e-5, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    HP_LEARN_RATE = hp.HParam('learning_rate', hp.RealInterval([16, 32]))
    HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
    HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))

    if not data_augmentation:
        print('Not using data augmentation.')
        tensorboard_logdir = "/workspace/tensorboard/no_data_aug/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        checkpoint = ModelCheckpoint("/workspace/project/saved_models/no_data_aug/model-{val_accuracy:.2f}.h5",
                                     monitor="val_accuracy", verbose=1, save_best_only=True,
                                     save_weights_only=True, mode="max")

        early_stop = EarlyStopping(monitor="val_loss", patience=50, mode="min")
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=1e-7, verbose=1, mode="min")
        tensorboard_callback = TensorBoard(log_dir=tensorboard_logdir)

        model.fit(x_train, train_labels,
                  validation_data=(x_test, test_labels),
                  callbacks=[checkpoint, reduce_lr, early_stop, tensorboard_callback],
                  epochs=200,
                  verbose=1,
                  shuffle=True,
                  batch_size=32)
    else:
        print('Using real-time data augmentation.')
        tensorboard_logdir = "/workspace/tensorboard/data_aug/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        checkpoint = ModelCheckpoint("/workspace/project/saved_models/data_aug/model-{val_accuracy:.2f}.h5",
                                     monitor="val_accuracy", verbose=1, save_best_only=True,
                                     save_weights_only=True, mode="max")

        early_stop = EarlyStopping(monitor="val_loss", patience=50, mode="min")
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=1e-7, verbose=1,
                                      mode="min")
        tensorboard_callback = TensorBoard(log_dir=tensorboard_logdir)
        # This will do preprocessing and realtime data augmentation:
        train_datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            zca_epsilon=1e-06,  # epsilon for ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            # randomly shift images horizontally (fraction of total width)
            width_shift_range=0.1,
            # randomly shift images vertically (fraction of total height)
            height_shift_range=0.1,
            shear_range=0.,  # set range for random shear
            zoom_range=0.,  # set range for random zoom
            channel_shift_range=0.,  # set range for random channel shifts
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            cval=0.,  # value used for fill_mode = "constant"
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False,  # randomly flip images
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)

        test_datagen = ImageDataGenerator()

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        train_datagen.fit(x_train)

        model.fit_generator(train_datagen.flow(x_train, train_labels, batch_size=64),
                            validation_data=test_datagen.flow(x_test, test_labels, batch_size=256),
                            callbacks=[checkpoint, reduce_lr, early_stop, tensorboard_callback],
                            epochs=200,
                            verbose=1,
                            shuffle=True,
                            use_multiprocessing=True,
                            workers=4)

    print("Training completed!")
