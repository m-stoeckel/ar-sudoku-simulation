import h5py
import keras
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tqdm import tqdm

from simulation.digit import BalancedDataGenerator
from simulation.digit.dataset import CuratedCharactersDataset, RandomPerspectiveTransform, \
    MNIST, np, PrerenderedDigitDataset, ClassSeparateMNIST, ConcatDataset, \
    PrerenderedCharactersDataset, CharacterDataset, EmptyDataset
from simulation.image.image_transforms import GaussianNoise, GaussianBlur, EmbedInGrid, Rescale, \
    RescaleIntermediateTransforms

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
    print("Loading data..")
    # concat_hand, concat_machine, concat_out = create_datasets()
    concat_hand, concat_machine, concat_out = load_datasets()

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
    concat_hand, concat_machine, concat_out = create_datasets()
    # concat_hand, concat_machine, concat_out = load_datasets()

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
            epochs=20,
            validation_data=test_generator,
            validation_steps=validation_steps
        )


def create_datasets():
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

    # Empty fields
    empty_dataset = EmptyDataset(28, 5000)

    # Concatenate datasets
    concat_machine = ConcatDataset([digit_dataset, prerendered_digit_dataset])
    concat_hand = ConcatDataset([mnist, curated_digits])
    concat_out = ConcatDataset([curated_out, prerendered_nondigit_dataset, empty_dataset])

    # Remove old datasets, as concatenation creates a copy of all images
    del digit_dataset, prerendered_digit_dataset, mnist, curated_digits, curated_out, prerendered_nondigit_dataset, \
        empty_dataset

    # Transforms
    noise = GaussianNoise()
    blur = GaussianBlur()
    embed = EmbedInGrid()
    perspective_transform = RandomPerspectiveTransform(0.1)
    rescale_intermediate_transforms = RescaleIntermediateTransforms((14, 14), [noise, blur])

    # Apply many transforms to machine digits
    for dataset in [concat_machine]:
        dataset.add_transforms(noise, blur)
        dataset.add_transforms(noise, perspective_transform, blur)
        dataset.add_transforms(embed, perspective_transform)
        dataset.add_transforms(embed, noise, perspective_transform, blur)
        dataset.add_transforms(embed, noise, blur)
        dataset.add_transforms(embed, rescale_intermediate_transforms)
        dataset.add_transforms(embed, noise, Rescale((14, 14)), blur)
        dataset.apply_transforms()

    # Apply some transforms to other digits
    for dataset in [concat_hand, concat_out]:
        dataset.add_transforms(embed, noise, blur)
        dataset.add_transforms(embed, rescale_intermediate_transforms)
        dataset.add_transforms(noise, perspective_transform, blur)
        dataset.apply_transforms()

    for dataset, name in tqdm(
            [(concat_machine, "concat_machine"), (concat_hand, "concat_hand"), (concat_out, "concat_out")],
            desc="Writing datasets to file"
    ):
        with h5py.File(f"datasets/{name}_dataset.hdf5", "w") as f:
            f.create_dataset("train_x", data=dataset.train_x)
            f.create_dataset("train_y", data=dataset.train_y)
            f.create_dataset("test_x", data=dataset.test_x)
            f.create_dataset("test_y", data=dataset.test_y)
    return concat_hand, concat_machine, concat_out


def load_datasets():
    concat_machine = CharacterDataset(28)
    concat_hand = CharacterDataset(28)
    concat_out = CharacterDataset(28)

    for dataset, name in tqdm(
            [(concat_machine, "concat_machine"), (concat_hand, "concat_hand"), (concat_out, "concat_out")],
            desc="Loading datasets from file"
    ):
        with h5py.File(f"datasets/{name}_dataset.hdf5", "r") as f:
            dataset.train_x = f["train_x"][:]
            dataset.train_y = f["train_y"][:]
            dataset.test_x = f["test_x"][:]
            dataset.test_y = f["test_y"][:]

    return concat_hand, concat_machine, concat_out


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
    # CharacterRenderer().prerender_all(mode='L')
    train_cnn()
