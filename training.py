import keras
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

from simulation.digit import BalancedDataGenerator
from simulation.digit.dataset import CuratedCharactersDataset, RandomPerspectiveTransform, \
    RandomPerspectiveTransformY, FilteredMNIST, MNIST, np, PrerenderedDigitDataset, ClassSeparateMNIST, ConcatDataset, \
    PrerenderedCharactersDataset
from simulation.image.image_transforms import GaussianNoise, GaussianBlur, EmbedInGrid

tf.debugging.set_log_device_placement(True)
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
    train_digit_dataset.apply_transforms(keep=True)
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
    print("Loading data..")
    digit_characters = "123456789"
    non_digit_characters = "0abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.!?"

    # Digit dataset
    digit_dataset = PrerenderedDigitDataset(digits_path="datasets/digits/")
    digit_dataset.resize(28)

    # Prerendered character dataset - digits
    prerendered_digit_dataset = PrerenderedCharactersDataset(
        digits_path="datasets/characters/",
        load_chars=digit_characters
    )
    prerendered_digit_dataset.resize(28)

    # Prerendered character dataset - non-digits
    prerendered_nondigit_dataset = PrerenderedCharactersDataset(
        digits_path="datasets/characters/",
        load_chars=non_digit_characters
    )
    prerendered_nondigit_dataset.resize(28)

    # Handwritten non-digits
    curated_out = CuratedCharactersDataset(
        digits_path="datasets/curated/",
        load_chars=non_digit_characters
    )
    curated_out.resize(28)

    # Handwritten digits
    curated_digits = CuratedCharactersDataset(digits_path="datasets/curated/", load_chars=digit_characters)
    curated_digits.resize(28)

    # Mnist digits
    mnist = ClassSeparateMNIST(data_home="datasets/")

    # Concatenate datasets
    concat_machine = ConcatDataset([digit_dataset, prerendered_digit_dataset])
    concat_hand = ConcatDataset([mnist, curated_digits])
    concat_out = ConcatDataset([curated_out, prerendered_nondigit_dataset])

    # Remove old datasets, as concatenation creates a copy of all images
    del digit_dataset, prerendered_digit_dataset, mnist, curated_digits, curated_out, prerendered_nondigit_dataset

    # Apply some transforms
    for dataset in [concat_machine]:
        dataset.add_transforms(GaussianNoise(), GaussianBlur())
        dataset.add_transforms(GaussianNoise(), RandomPerspectiveTransform(0.1), GaussianBlur())
        dataset.add_transforms(EmbedInGrid(), GaussianNoise(), GaussianBlur())
        dataset.apply_transforms()

    # Apply some transforms
    for dataset in [concat_hand, concat_out]:
        dataset.add_transforms(EmbedInGrid(), GaussianNoise(), GaussianBlur())
        dataset.apply_transforms()

    batch_size = 48
    train_generator = BalancedDataGenerator(
        concat_machine.train, concat_hand.train, concat_out.train,
        batch_size=batch_size,
        shuffle=True,
        resolution=28
    )

    test_generator = BalancedDataGenerator(
        concat_machine.test, concat_hand.test, concat_out.test,
        batch_size=batch_size,
        shuffle=True,
        resolution=28
    )

    steps_per_epoch = len(train_generator)
    validation_steps = len(test_generator)

    # Run training on the GPU
    with tf.device('/GPU:0'):
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
