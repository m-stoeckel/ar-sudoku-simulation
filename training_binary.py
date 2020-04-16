import os


import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense, Dropout

from generate_datasets import load_datasets, TRANSFORMED_DATASET_NAMES
from simulation.data.data_generator import ToBinaryGenerator
from training import evaluate_and_plot


def get_cnn_binary_model():
    model = Sequential()
    model.add(Conv2D(16, (5, 5), strides=2,
                     input_shape=(28, 28, 1)))  # 24x24x16
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(4, 4)))  # 6x6x16
    model.add(Conv2D(32, (2, 2)))  # 6x62x32
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # 3x3x32
    model.add(Flatten())  # 256
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='sigmoid'))
    return model


def train_binary_model():
    os.makedirs("model_binary_finetuning", exist_ok=True)
    concat_machine, concat_hand, concat_out, real_training, real_validation = load_datasets(TRANSFORMED_DATASET_NAMES)

    batch_size = 192
    train_generator = ToBinaryGenerator(
        concat_machine.train, concat_hand.train, concat_out.train,
        batch_size=batch_size,
        shuffle=True,
        truncate=True
    )

    dev_generator = ToBinaryGenerator(
        concat_machine.test, concat_hand.test, concat_out.test,
        batch_size=batch_size,
        shuffle=True,
        truncate=True
    )

    ft_train_generator = ToBinaryGenerator(
        real_training.train,
        batch_size=batch_size,
        shuffle=True,
        truncate=True
    )

    ft_dev_generator = ToBinaryGenerator(
        real_training.test,
        batch_size=batch_size,
        shuffle=True,
        truncate=True
    )

    test_generator = ToBinaryGenerator(
        real_validation.test,
        batch_size=batch_size,
        shuffle=False
    )

    # Run training on the GPU
    with tf.device('/GPU:0'):
        # Keras Model
        print("Creating model..")
        model = get_cnn_binary_model()

        # Hyperparameters
        epochs = 10
        ft_epochs = 10
        learning_rate = 0.001

        def mean_pred(_, y):
            return keras.backend.mean(y)

        print("Compiling model..")
        model.compile(
            loss=keras.losses.binary_crossentropy,
            optimizer=keras.optimizers.Adagrad(lr=learning_rate),
            metrics=[keras.metrics.binary_accuracy, 'mse', mean_pred],
        )
        print(model.summary())

        print("Training model")
        model.fit_generator(
            train_generator, validation_data=dev_generator,
            epochs=epochs,
            use_multiprocessing=True,
            workers=8,
            callbacks=[
                EarlyStopping(monitor='val_binary_accuracy', restore_best_weights=True),
            ]
        )

        print("Finetuning model")
        model.fit_generator(
            ft_train_generator, validation_data=ft_train_generator,
            epochs=ft_epochs,
            use_multiprocessing=True,
            workers=8,
            callbacks=[
                EarlyStopping(monitor='val_binary_accuracy', restore_best_weights=True),
            ]
        )

        model.save("model_binary_finetuning/binary_model.ft.final.hdf5")

        print("Evaluating")
        print("Training dev", list(zip(model.metrics_names, model.evaluate_generator(dev_generator))))
        print("Finetuning dev", list(zip(model.metrics_names, model.evaluate_generator(ft_dev_generator))))
        print("Test", list(zip(model.metrics_names, model.evaluate_generator(test_generator))))
        evaluate_and_plot(model, test_generator)


if __name__ == '__main__':
    # CharacterRenderer().prerender_all(mode='L')
    # generate_base_datasets()
    # generate_transformed_datasets()
    # create_data_overview()
    train_binary_model()
    # load_and_evaluate()
