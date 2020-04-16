import os
from typing import Tuple

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.metrics import classification_report
from tensorflow.keras import Model, Sequential
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.callbacks import EarlyStopping

from generate_datasets import load_datasets, get_labels, plot_9x9_grid, TRANSFORMED_DATASET_NAMES
from simulation.data.data_generator import SimpleDataGenerator, BaseDataGenerator


def train_cnn(path="model/", to_simple_digit=False):
    """
    Train the CNN model and save it under the given path.
    
    The method first loads the models using *generate_datasets.py* methods. Then the model is trained and saved and
    finally evaluated.

    Args:
        path: The directory to save the trained model to. (Default value = "model/")
        to_simple_digit: If true, convert the datasets to simple 9 + 2 class digit recognition. (Default value = False)

    Returns:
        None

    """
    os.makedirs(path, exist_ok=True)

    print("Loading data..")
    concat_machine, concat_hand, concat_out, real_training, real_validation = load_datasets(TRANSFORMED_DATASET_NAMES)

    batch_size = 256
    train_generator = SimpleDataGenerator(
        concat_machine.train, concat_hand.train, concat_out.train,
        batch_size=batch_size,
        shuffle=True,
        to_simple_digit=to_simple_digit
    )

    dev_generator = SimpleDataGenerator(
        concat_machine.test, concat_hand.test, concat_out.test,
        batch_size=batch_size,
        shuffle=True,
        to_simple_digit=to_simple_digit
    )

    ft_train_generator = SimpleDataGenerator(
        real_training.train,
        batch_size=batch_size,
        shuffle=True,
        to_simple_digit=to_simple_digit
    )

    ft_dev_generator = SimpleDataGenerator(
        real_training.test,
        batch_size=batch_size,
        shuffle=True,
        to_simple_digit=to_simple_digit
    )

    test_generator = SimpleDataGenerator(
        real_validation.test,
        batch_size=batch_size,
        shuffle=False,
        to_simple_digit=to_simple_digit
    )

    # Run training on the GPU
    with tf.device('/GPU:0'):
        # Keras Model
        print("Creating model..")
        model = Sequential()
        model.add(layers.InputLayer(input_shape=(28, 28, 1)))
        model.add(layers.Conv2D(16, (3, 3)))  # 26x26x16
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))  # 13x13x16
        model.add(layers.Conv2D(32, (2, 2)))  # 12x12x32
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))  # 6x6x32
        model.add(layers.Conv2D(64, (3, 3)))  # 4x4x64
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))  # 2x2x64
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Flatten())  # 256
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.25))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(train_generator.num_classes))

        # Hyperparameters
        epochs = 100
        ft_epochs = 100

        print("Compiling model..")
        model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=keras.optimizers.Adadelta(0.01),
            metrics=['accuracy']
        )
        print(model.summary())

        print("Training model")
        model.fit(
            train_generator, validation_data=dev_generator,
            epochs=epochs,
            callbacks=[
                EarlyStopping(monitor='val_accuracy', restore_best_weights=True, patience=3, min_delta=0.0001),
            ]
        )

        print("Finetuning model")
        model.fit(
            ft_train_generator, validation_data=ft_train_generator,
            epochs=ft_epochs,
            callbacks=[
                EarlyStopping(monitor='val_accuracy', restore_best_weights=True, patience=3, min_delta=0.0001),
            ]
        )

        print("Saving keras model")
        models.save_model(model, path + "model.hdf5")

        print("Evaluating keras model")
        print("Training dev", dict(zip(model.metrics_names, model.evaluate(dev_generator))))
        print("Finetuning dev", dict(zip(model.metrics_names, model.evaluate(ft_dev_generator))))
        print("Test", dict(zip(model.metrics_names, model.evaluate(test_generator))))
        evaluate(model, test_generator)


def convert_to_tflite(model: Model, path: str, test_generator: BaseDataGenerator):
    """
    Converts a Keras model to a tf.lite byte model.

    Args:
        model(tensorflow.keras.Model): The Keras model to convert.
        path(str): The directory path for the model.
        test_generator(:py:class:`simulation.data.data_generator.BaseDataGenerator`): The generator for test files.

    Returns:
        None

    """
    print("Converting to TFLite model")
    converter: tf.lite.TFLiteConverter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_float_model = converter.convert()
    with open(path + "model.tflite", "wb") as f:
        f.write(tflite_float_model)
    print(f"TFLite model accurracy: {evaluate_tflite_model(tflite_float_model, test_generator):0.02f}")
    print("Converting to quantized TFLite model")
    converter: tf.lite.TFLiteConverter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quantized_model = converter.convert()
    with open(path + "model.quantized.tflite", "wb") as f:
        f.write(tflite_quantized_model)
    print(f"TFLite quantized model accurracy: {evaluate_tflite_model(tflite_quantized_model, test_generator):0.02f}")


def evaluate_tflite_model(tflite_model_content: bytes, test_generator: BaseDataGenerator) -> float:
    """
    Evaluate a tf.lite model with the given *test_generator*.

    Args:
        tflite_model_content(bytes): The tf.lite model content, output of TFLiteConverter.convert().
        test_generator(:py:class:`simulation.data.data_generator.BaseDataGenerator`): The generator for test files.

    Returns:
        The accuracy of the tf.lite model on the test files.

    """
    test_images = test_generator.get_data()
    test_labels = test_generator.get_labels()

    # Initialize TFLite interpreter using the model.
    interpreter = tf.lite.Interpreter(model_content=tflite_model_content)
    interpreter.allocate_tensors()
    input_tensor_index = interpreter.get_input_details()[0]["index"]
    output = interpreter.tensor(interpreter.get_output_details()[0]["index"])

    # Run predictions on every image in the "test" dataset.
    prediction_digits = []
    for test_image in test_images:
        # Pre-processing: add batch dimension and convert to float32 to match with
        # the model's input digit format.
        test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
        interpreter.set_tensor(input_tensor_index, test_image)

        # Run inference.
        interpreter.invoke()

        # Post-processing: remove batch dimension and find the digit with highest
        # probability.
        digit = np.argmax(output()[0])
        prediction_digits.append(digit)

    # Compare prediction results with ground truth labels to calculate accuracy.
    accurate_count = 0
    for index in range(len(prediction_digits)):
        if prediction_digits[index] == test_labels[index]:
            accurate_count += 1
    accuracy = accurate_count * 1.0 / len(prediction_digits)

    return accuracy


def evaluate(
        model: Model,
        test_generator: BaseDataGenerator,
        binary=False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate a given model with the given generator.

    Args:
        model(tensorflow.keras.Model): The Keras model to evaluate.
        test_generator(:py:class:`simulation.data.data_generator.BaseDataGenerator`): The generator for test files.
        binary(bool): If True, the given model is a binary recognition model. (Default value = False)

    Returns:
        tuple[:py:class:`numpy.ndarray`, :py:class:`numpy.ndarray`, :py:class:`numpy.ndarray`]: The

    """
    x = test_generator.get_data()
    y_true = test_generator.get_labels()
    if binary:
        y_pred = (model.predict(x) >= 0.5).astype(np.int).squeeze()
    else:
        y_pred = np.array(model.predict_classes(x))

    print(list(zip(model.metrics_names, model.evaluate_generator(test_generator))))
    print(classification_report(y_true, y_pred))
    print(f"{p}/{t}" + ("!" if p != t else "") for p, t in zip(y_pred, y_true))
    return x, y_true, y_pred


def evaluate_and_plot(model: Model, test_generator: BaseDataGenerator, binary=False):
    """
    Evaluate a given model and plot the results on the test_generator to a set of files.

    Args:
        model: The model to evaluate.
        test_generator: The generator for test files.
        binary: If True, the given model is a binary recognition model. (Default value = False)
        model: Model:
        test_generator: BaseDataGenerator:

    Returns:
      None

    """
    x, y_true, y_pred = evaluate(model, test_generator, binary)
    y = get_labels(y_true, y_pred)
    zipped = list(zip(x, y))
    for idx in range(0, len(zipped), 81):
        plot_9x9_grid(zipped[idx:idx + 81], f"{'binary' if binary else 'full'}_validation_set_{idx // 81 + 1}")


def load_and_evaluate(filepath="model_simple_finetuning/cnn_model.ft.final.hdf5"):
    """
    Load and evaluate a model at the given file path.

    Args:
      filepath: return: (Default value = "model_simple_finetuning/cnn_model.ft.final.hdf5")

    Returns:

    """
    model = models.load_model(filepath)
    concat_machine, concat_hand, concat_out, real_training, real_validation = load_datasets(TRANSFORMED_DATASET_NAMES)

    test_generator = SimpleDataGenerator(
        real_validation.test,
        batch_size=64,
        shuffle=False,
        to_simple_digit=True
    )

    evaluate(model, test_generator)


if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.get_logger().setLevel('ERROR')

    train_cnn("model_simple_finetuning/", True)
    train_cnn("model_full_finetuning/", False)

    validation = load_datasets([TRANSFORMED_DATASET_NAMES[-1]])[0]
    test_generator = SimpleDataGenerator(
        validation.test,
        batch_size=64,
        shuffle=False,
        to_simple_digit=True
    )
    model = models.load_model("model_simple_finetuning/model.hdf5")
    evaluate(model, test_generator)
    convert_to_tflite(model, "model_simple_finetuning/", test_generator)

    test_generator = SimpleDataGenerator(
        validation.test,
        batch_size=64,
        shuffle=False,
        to_simple_digit=False
    )
    model = models.load_model("model_full_finetuning/model.hdf5")
    evaluate(model, test_generator)
    convert_to_tflite(model, "model_full_finetuning/", test_generator)
