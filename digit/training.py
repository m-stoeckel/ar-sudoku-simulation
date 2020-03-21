import keras
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

from digit.dataset import CuratedCharactersDataset, RandomPerspectiveTransform, \
    RandomPerspectiveTransformY, FilteredMNIST, MNIST, np
from digit import BalancedDataGenerator
from image.image_transforms import GaussianNoise, GaussianBlur

tf.get_logger().setLevel('ERROR')


# SGD or Adam work well
def get_linear_model(n_classes=18):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(28 ** 2,)))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(n_classes, activation='softmax'))
    return model


# Adadelta or Adagrad work well
def get_cnn_model(n_classes=18):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(n_classes, activation='softmax'))
    return model


def train_linear():
    batch_size = 128
    print("Loading data..")
    # Load MNIST dataset, with zeros filtered out
    mnist_dataset = FilteredMNIST()
    print(mnist_dataset.train_x.shape, mnist_dataset.test_x.shape)

    # Crate large training dataset
    train_digit_dataset = CuratedCharactersDataset()
    train_digit_dataset.add_transforms(RandomPerspectiveTransform(0.2))
    train_digit_dataset.add_transforms(RandomPerspectiveTransform(0.25))
    train_digit_dataset.add_transforms(RandomPerspectiveTransform(0.333))
    train_digit_dataset.add_transforms(RandomPerspectiveTransformY(0.2))
    train_digit_dataset.add_transforms(RandomPerspectiveTransformY(0.25))
    train_digit_dataset.add_transforms(RandomPerspectiveTransformY(0.333))
    train_digit_dataset.apply_transforms(keep=False)
    print(train_digit_dataset.train_x.shape)
    train_generator = BalancedDataGenerator(train_digit_dataset, mnist_dataset.train,
                                            batch_size=batch_size, flatten=True)

    # Create separate, small validation dataset
    test_digit_dataset = CuratedCharactersDataset()
    test_digit_dataset.add_transforms(RandomPerspectiveTransform())
    test_digit_dataset.apply_transforms(keep=True)
    print(test_digit_dataset.train_x.shape)
    test_generator = BalancedDataGenerator(test_digit_dataset, mnist_dataset.test,
                                           batch_size=batch_size, flatten=True)

    assert train_generator.num_classes == test_generator.num_classes

    steps_per_epoch = len(train_generator)
    validation_steps = len(test_generator)

    # Keras Model
    print("Creating model..")
    model = get_linear_model(n_classes=train_generator.num_classes)

    print("Compiling model..")
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    print(model.summary())

    print("Starting training..")
    model.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=20,
        validation_data=test_generator,
        validation_steps=validation_steps
    )


def train_cnn():
    batch_size = 64
    print("Loading data..")
    # Load MNIST dataset, with zeros filtered out
    mnist_dataset = FilteredMNIST()
    print(mnist_dataset.train_x.shape, mnist_dataset.test_x.shape)

    # Crate large training dataset
    train_digit_dataset = CuratedCharactersDataset()
    train_digit_dataset.add_transforms(GaussianNoise(), RandomPerspectiveTransform(0.1), GaussianBlur())
    train_digit_dataset.add_transforms(GaussianNoise(), RandomPerspectiveTransform(0.15), GaussianBlur())
    train_digit_dataset.add_transforms(GaussianNoise(), RandomPerspectiveTransform(0.2), GaussianBlur())
    train_digit_dataset.add_transforms(GaussianNoise(), RandomPerspectiveTransformY(0.1), GaussianBlur())
    train_digit_dataset.add_transforms(GaussianNoise(), RandomPerspectiveTransformY(0.15), GaussianBlur())
    train_digit_dataset.add_transforms(GaussianNoise(), RandomPerspectiveTransformY(0.2), GaussianBlur())
    train_digit_dataset.apply_transforms(keep=False)
    print(train_digit_dataset.train_x.shape)
    train_generator = BalancedDataGenerator(train_digit_dataset, mnist_dataset.train, batch_size=batch_size)

    # Create separate, small validation dataset
    test_digit_dataset = CuratedCharactersDataset()
    test_digit_dataset.add_transforms(GaussianNoise(sigma=4), GaussianBlur())
    test_digit_dataset.add_transforms(GaussianNoise(sigma=4), RandomPerspectiveTransform(), GaussianBlur())
    test_digit_dataset.add_transforms(GaussianNoise(), RandomPerspectiveTransform(), GaussianBlur())
    test_digit_dataset.apply_transforms(keep=False)
    print(test_digit_dataset.train_x.shape)
    test_generator = BalancedDataGenerator(test_digit_dataset, mnist_dataset.test, batch_size=batch_size)

    assert train_generator.num_classes == test_generator.num_classes

    steps_per_epoch = len(train_generator)
    validation_steps = len(test_generator)

    # Keras Model
    print("Creating model..")
    model = get_cnn_model(n_classes=train_generator.num_classes)

    print("Compiling model..")
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adagrad(),
                  metrics=['accuracy'])
    print(model.summary())

    print("Starting training..")
    model.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=10,
        validation_data=test_generator,
        validation_steps=validation_steps
    )


def train_mnist():
    batch_size = 64
    print("Loading data..")
    mnist_dataset = MNIST()
    print(mnist_dataset.train_x.shape, mnist_dataset.test_x.shape)

    # Convert native MNIST to trainable format
    train_x = mnist_dataset.train_x.astype(np.float32)
    train_x = train_x[:, :, :, np.newaxis]
    train_x /= 255.
    train_y = keras.utils.to_categorical(mnist_dataset.train_y, num_classes=10)

    test_x = mnist_dataset.test_x.astype(np.float32)
    test_x = test_x[:, :, :, np.newaxis]
    test_x /= 255.
    test_y = keras.utils.to_categorical(mnist_dataset.test_y, num_classes=10)

    # Keras Model
    print("Creating model..")
    model = get_cnn_model(n_classes=10)

    print("Compiling model..")
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adagrad(),
                  metrics=['accuracy'])
    print(model.summary())

    print("Starting training..")
    model.fit(
        train_x, train_y,
        epochs=10, batch_size=batch_size,
        validation_data=(test_x, test_y)
    )


if __name__ == '__main__':
    # train_linear()
    train_cnn()
    # train_mnist()
