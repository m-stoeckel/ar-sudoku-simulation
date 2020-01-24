import keras
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

from digit_dataset import BalancedMnistDigitDataGenerator, DigitDataset, RandomPerspectiveTransform, \
    RandomPerspectiveTransformY, FilteredMNIST

tf.get_logger().setLevel('ERROR')


def get_linear_model(n_classes=18):
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(n_classes, activation='softmax'))
    return model


def get_cnn_model(n_classes=18):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))
    return model


if __name__ == '__main__':
    print("Loading data..")
    # Load MNIST dataset, with zeros filtered out
    mnist_dataset = FilteredMNIST()

    # Crate large training dataset
    train_digit_dataset = DigitDataset(resolution=28)
    train_digit_dataset.add_transforms(RandomPerspectiveTransform())
    train_digit_dataset.add_transforms(RandomPerspectiveTransform(0.333))
    train_digit_dataset.add_transforms(RandomPerspectiveTransformY())
    train_digit_dataset.add_transforms(RandomPerspectiveTransformY(0.333))
    train_digit_dataset.apply_transforms(keep=False)
    train_generator = BalancedMnistDigitDataGenerator(train_digit_dataset, mnist_dataset.train)

    # Create separate, small validation dataset
    test_digit_dataset = DigitDataset(resolution=28)
    test_digit_dataset.add_transforms(RandomPerspectiveTransform())
    test_digit_dataset.apply_transforms(keep=True)
    test_generator = BalancedMnistDigitDataGenerator(test_digit_dataset, mnist_dataset.test)

    steps_per_epoch = len(train_generator)
    validation_steps = len(test_generator)

    # Keras Model
    print("Creating model..")
    model = get_cnn_model()

    print("Compiling model..")
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    print("Starting training..")
    model.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=10,
        validation_data=test_generator,
        validation_steps=validation_steps)
