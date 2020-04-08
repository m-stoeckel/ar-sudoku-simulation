import os

import keras
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import *
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model

from generate_datasets import load_datasets, get_labels, plot_9x9_grid, TRANSFORMED_DATASET_NAMES
from simulation.data.data_generator import SimpleDataGenerator, BaseDataGenerator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def train_cnn(path="model/", to_simple_digit=False):
    os.makedirs(path, exist_ok=True)

    print("Loading data..")
    concat_machine, concat_hand, concat_out, real_training, real_validation = load_datasets(TRANSFORMED_DATASET_NAMES)

    batch_size = 192
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
        model.add(InputLayer(input_shape=(28, 28, 1)))
        model.add(Conv2D(16, (3, 3)))  # 26x26x16
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))  # 13x13x16
        model.add(Conv2D(32, (2, 2)))  # 12x12x32
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))  # 6x6x32
        model.add(Conv2D(64, (3, 3)))  # 4x4x64
        model.add(MaxPooling2D(pool_size=(2, 2)))  # 2x2x64
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Flatten())  # 256
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(train_generator.num_classes, activation='softmax'))

        # Hyperparameters
        epochs = 50
        ft_epochs = 30

        print("Compiling model..")
        model.compile(
            loss=keras.losses.CategoricalCrossentropy(from_logits=True),
            optimizer=keras.optimizers.Adagrad(),
            metrics=['accuracy'],
        )
        print(model.summary())

        print("Training model")
        model.fit_generator(
            train_generator, validation_data=dev_generator,
            epochs=epochs,
            use_multiprocessing=True,
            workers=8,
            callbacks=[
                EarlyStopping(monitor='accuracy', restore_best_weights=True, patience=3),
            ]
        )

        print("Finetuning model")
        model.fit_generator(
            ft_train_generator, validation_data=ft_train_generator,
            epochs=ft_epochs,
            use_multiprocessing=True,
            workers=8,
            callbacks=[
                EarlyStopping(monitor='accuracy', restore_best_weights=True),
            ]
        )

        print("Saving keras model")
        model.save(path + "model.hdf5", include_optimizer=False)

        print("Evaluating keras model")
        print("Training dev", list(zip(model.metrics_names, model.evaluate_generator(dev_generator))))
        print("Finetuning dev", list(zip(model.metrics_names, model.evaluate_generator(ft_dev_generator))))
        print("Test", list(zip(model.metrics_names, model.evaluate_generator(test_generator))))
        # evaluate_and_plot(model, test_generator)


def convert_to_tflite(model, path, test_generator):
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


def evaluate(model: Sequential, test_generator: BaseDataGenerator, binary=False):
    x = np.expand_dims(test_generator.get_data(), 3).astype(np.float) / 255
    y_true = test_generator.get_labels()
    if binary:
        y_pred = (model.predict(x) >= 0.5).astype(np.int).squeeze()
    else:
        y_pred = np.array(model.predict_classes(x))

    print(list(zip(model.metrics_names, model.evaluate_generator(test_generator))))
    print(classification_report(y_true, y_pred))
    print(f"{p}/{t}" + ("!" if p != t else "") for p, t in zip(y_pred, y_true))
    return x, y_true, y_pred


def evaluate_and_plot(model: Sequential, test_generator: BaseDataGenerator, binary=False):
    x, y_true, y_pred = evaluate(model, test_generator, binary)
    y = get_labels(y_true, y_pred)
    zipped = list(zip(x, y))
    for idx in range(0, len(zipped), 81):
        plot_9x9_grid(zipped[idx:idx + 81], f"{'binary' if binary else 'full'}_validation_set_{idx // 81 + 1}")


def load_and_evaluate():
    model = keras.models.load_model("model_simple_finetuning/cnn_model.ft.final.hdf5")
    concat_machine, concat_hand, concat_out, real_training, real_validation = load_datasets(TRANSFORMED_DATASET_NAMES)

    test_generator = SimpleDataGenerator(
        real_validation.test,
        batch_size=64,
        shuffle=False,
        to_simple_digit=True
    )

    evaluate(model, test_generator)


# A helper function to evaluate the TF Lite model using "test" dataset.
def evaluate_tflite_model(tflite_model, test_generator):
    test_images = np.expand_dims(test_generator.get_data(), 3).astype(np.float) / 255
    test_labels = test_generator.get_labels()

    # Initialize TFLite interpreter using the model.
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_tensor_index = interpreter.get_input_details()[0]["index"]
    output = interpreter.tensor(interpreter.get_output_details()[0]["index"])

    # Run predictions on every image in the "test" dataset.
    prediction_digits = []
    for test_image in test_images:
        # Pre-processing: add batch dimension and convert to float32 to match with
        # the model's input data format.
        test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
        interpreter.set_tensor(input_tensor_index, test_image)

        # Run inference.
        interpreter.invoke()

        # Post-processing: remove batch dimension and find the data with highest
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


if __name__ == '__main__':
    train_cnn("model_simple_finetuning/", True)
    train_cnn("model_full_finetuning/", False)

    real_validation, = load_datasets(TRANSFORMED_DATASET_NAMES[-1])

    test_generator = SimpleDataGenerator(
        real_validation.test,
        batch_size=64,
        shuffle=False,
        to_simple_digit=True
    )
    model = load_model("model_simple_finetuning/model.hdf5")
    convert_to_tflite(model, "model_simple_finetuning/", test_generator)

    test_generator = SimpleDataGenerator(
        real_validation.test,
        batch_size=64,
        shuffle=False,
        to_simple_digit=False
    )
    model = load_model("model_full_finetuning/model.hdf5")
    convert_to_tflite(model, "model_full_finetuning/", test_generator)
