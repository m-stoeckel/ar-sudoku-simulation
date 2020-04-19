import os
from typing import List, Union

import tensorflow as tf
import tensorflow.keras as keras
from h5py import File
from tensorflow.keras import Sequential
from tensorflow.keras import models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense, Dropout

from generate_datasets import load_datasets, TRANSFORMED_DATASET_NAMES
from simulation.data.data_generator import ToBinaryGenerator
from training import evaluate, convert_to_tflite


def train_binary_model(
        path,
        epochs=100,
        ft_epochs=100,
        learning_rate=0.01,
        classes_to_match: Union[int, List[int]] = 0,
        classes_to_drop: Union[int, List[int]] = None
):
    """
    Train a smaller binary model for empty/not empty classification and save it under the given path. The method first
    loads the models using :py:doc:`generate_datasets.py <training.generate_datasets.py>` methods. Then the model is
    trained, saved and finally evaluated.

    Training is run in two steps: It is first trained with synthetic data and then finetuned with real data. Early
    stopping is used to prevent overfitting.

    Args:
        path(str): The directory to save the trained model to.
        epochs(int): The number of epochs. (Default value = 100)
        ft_epochs: The number of finetuning epochs. (Default value = 100)
        learning_rate: The learning rate for the Adadelta optimizer. (Default value = 0.01)
        classes_to_match(Union[int, list[int]]): The classes to match as class 1. (Default value = 0)
        classes_to_drop(Union[int, list[int]]): The classes to drop from the dataset. (Default value = None)

    Returns:
        None

    """
    os.makedirs(path, exist_ok=True)
    concat_machine, concat_hand, concat_out, real_training, real_validation = load_datasets(TRANSFORMED_DATASET_NAMES)

    batch_size = 192
    train_generator = ToBinaryGenerator(
        concat_machine.train, concat_hand.train, concat_out.train,
        classes_to_match=classes_to_match,
        classes_to_drop=classes_to_drop,
        batch_size=batch_size,
        shuffle=True,
        truncate=True
    )

    dev_generator = ToBinaryGenerator(
        concat_machine.test, concat_hand.test, concat_out.test,
        classes_to_match=classes_to_match,
        classes_to_drop=classes_to_drop,
        batch_size=batch_size,
        shuffle=True,
        truncate=True
    )

    ft_train_generator = ToBinaryGenerator(
        real_training.train,
        classes_to_match=classes_to_match,
        classes_to_drop=classes_to_drop,
        batch_size=batch_size,
        shuffle=True,
        truncate=True
    )

    ft_dev_generator = ToBinaryGenerator(
        real_training.test,
        classes_to_match=classes_to_match,
        classes_to_drop=classes_to_drop,
        batch_size=batch_size,
        shuffle=True,
        truncate=True
    )

    test_generator = ToBinaryGenerator(
        real_validation.test,
        classes_to_match=classes_to_match,
        classes_to_drop=classes_to_drop,
        batch_size=batch_size,
        shuffle=False
    )

    # Run training on the GPU
    with tf.device('/GPU:0'):
        # Keras Model
        print("Creating model..")
        model = Sequential()
        model.add(Conv2D(16, (5, 5), strides=2,
                         input_shape=(28, 28, 1)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(4, 4)))
        model.add(Conv2D(32, (2, 2)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())  # 32
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        # def mean_pred(_, y):
        #     return keras.backend.mean(y)

        print("Compiling model..")
        model.compile(
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
            optimizer=keras.optimizers.Adadelta(learning_rate),
            metrics=[keras.metrics.binary_accuracy, 'mse'],
        )
        print(model.summary())

        print("Training model")
        model.fit_generator(
            train_generator, validation_data=dev_generator,
            epochs=epochs,
            callbacks=[
                EarlyStopping(monitor='val_accuracy', restore_best_weights=True, patience=3, min_delta=0.0001),
            ]
        )

        print("Finetuning model")
        model.fit_generator(
            ft_train_generator, validation_data=ft_train_generator,
            epochs=ft_epochs,
            callbacks=[
                EarlyStopping(monitor='val_accuracy', restore_best_weights=True, patience=3, min_delta=0.0001),
            ]
        )

        models.save_model(model, path + "model.h5", save_format='h5')

        print("Evaluating")
        print("Training dev", list(zip(model.metrics_names, model.evaluate_generator(dev_generator))))
        print("Finetuning dev", list(zip(model.metrics_names, model.evaluate_generator(ft_dev_generator))))
        print("Test", list(zip(model.metrics_names, model.evaluate_generator(test_generator))))
        evaluate(model, test_generator, binary=True)


if __name__ == '__main__':
    tf.get_logger().setLevel('ERROR')
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Train empty vs. not-empty classifier
    train_binary_model("model_empty_finetuning/")

    validation = load_datasets([TRANSFORMED_DATASET_NAMES[-1]])[0]
    test_generator = ToBinaryGenerator(
        validation.test,
        classes_to_match=0,
        batch_size=64,
        shuffle=False
    )
    model = models.load_model("model_empty_finetuning/model.h5")
    # evaluate(model, test_generator)
    convert_to_tflite(model, "model_empty_finetuning/", test_generator, binary=True)

    # Train handwritten vs. machine-written classifier
    classes_to_match = list(range(1, 10))
    train_binary_model("model_hand_finetuning/", classes_to_match=classes_to_match)

    validation = load_datasets([TRANSFORMED_DATASET_NAMES[-1]])[0]
    test_generator = ToBinaryGenerator(
        validation.test,
        classes_to_match=classes_to_match,
        classes_to_drop=0,
        batch_size=64,
        shuffle=False
    )
    model = models.load_model("model_hand_finetuning/model.h5")
    # evaluate(model, test_generator)
    convert_to_tflite(model, "model_hand_finetuning/", test_generator, binary=True)
