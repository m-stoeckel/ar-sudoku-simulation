import json
import os
import sys
from pathlib import Path

import h5py
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tqdm import tqdm

from simulation.digit import BalancedDataGenerator
from simulation.digit.data_generator import SimpleDataGenerator
from simulation.digit.dataset import PrerenderedDigitDataset, ClassSeparateMNIST, ConcatDataset, \
    PrerenderedCharactersDataset, CharacterDataset, EmptyDataset, ClassSeparateCuratedCharactersDataset
from simulation.transforms import *

tf.get_logger().setLevel('ERROR')


def create_datasets():
    base_names = ["base_machine_dataset", "base_hand_dataset", "base_out_dataset"]
    if not all(os.path.exists(f"datasets/{name}.hdf5") for name in base_names):
        digit_characters = "123456789"
        non_digit_characters = "0abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.!?"

        #########################
        # Machinewritten digits #
        #########################

        # "Digit" dataset
        digit_dataset = PrerenderedDigitDataset(digits_path="datasets/digits/")
        digit_dataset.resize(28)

        # Prerendered character dataset - digits
        prerendered_digit_dataset = PrerenderedCharactersDataset(
            digits_path="datasets/characters/",
            load_chars=digit_characters
        )
        prerendered_digit_dataset.resize(28)

        ######################
        # Handwritten digits #
        ######################

        # Curated digits
        curated_digits = ClassSeparateCuratedCharactersDataset(digits_path="datasets/curated/",
                                                               load_chars=digit_characters)
        curated_digits.resize(28)

        # Mnist digits
        mnist = ClassSeparateMNIST(data_home="datasets/")

        ##############
        # Non-digits #
        ##############

        # Prerendered non-digits
        prerendered_nondigit_dataset = PrerenderedCharactersDataset(
            digits_path="datasets/characters/",
            load_chars=non_digit_characters
        )
        prerendered_nondigit_dataset.resize(28)

        # Curated non-digits
        curated_out = ClassSeparateCuratedCharactersDataset(
            digits_path="datasets/curated/",
            load_chars=non_digit_characters
        )
        curated_out.resize(28)

        # Empty fields
        empty_dataset = EmptyDataset(28, 1000)

        # Concatenate datasets
        concat_machine = ConcatDataset([digit_dataset, prerendered_digit_dataset])
        concat_hand = ConcatDataset([mnist, curated_digits])
        # concat_out = ConcatDataset([curated_out, prerendered_nondigit_dataset, empty_dataset])
        # concat_out = ConcatDataset([curated_out, empty_dataset])
        concat_out = ConcatDataset([empty_dataset])

        # Save base datasets for later
        save_datsets([(concat_machine, "base_machine_dataset"), (concat_hand, "base_hand_dataset"),
                      (concat_out, "base_out_dataset")])

        # Remove old datasets, as concatenation creates a copy of all images
        del digit_dataset, prerendered_digit_dataset, mnist, curated_digits, empty_dataset, \
            curated_out, prerendered_nondigit_dataset
    else:
        concat_hand, concat_machine, concat_out = load_datasets(base_names)

    # Transforms
    perspective_transform = RandomPerspectiveTransform(0.1, background_color=0)
    downscale_intermediate_transforms = RescaleIntermediateTransforms(
        (14, 14),
        [perspective_transform, JPEGEncode()],
        inter_consecutive=cv2.INTER_NEAREST
    )
    upscale_and_salt = RescaleIntermediateTransforms(
        (92, 92),
        [SaltAndPepperNoise(amount=0.002, ratio=1), Dilate()],
        inter_initial=cv2.INTER_LINEAR, inter_consecutive=cv2.INTER_AREA
    )

    # Apply many transforms to machine digits
    print("Applying transforms to machine digits")
    for dataset in [concat_machine]:
        dataset.add_transforms(EmbedInGrid())
        dataset.apply_transforms()

        dataset.add_transforms(PoissonNoise())
        dataset.add_transforms(upscale_and_salt)
        dataset.add_transforms(GrainNoise())
        dataset.add_transforms(GaussianNoise())
        dataset.apply_transforms()

        dataset.add_transforms(perspective_transform)
        dataset.add_transforms(perspective_transform, JPEGEncode())
        dataset.add_transforms(downscale_intermediate_transforms)
        dataset.add_transforms(JPEGEncode())
        dataset.apply_transforms()

    # Apply some transforms to other digits
    print("Applying transforms to handwritten digits")
    for dataset in [concat_hand, ]:
        dataset.add_transforms(EmbedInGrid())
        dataset.apply_transforms()

        dataset.add_transforms(upscale_and_salt)
        dataset.add_transforms(GrainNoise())
        dataset.apply_transforms()

    print("Applying transforms to out characters")
    for dataset in [concat_out]:
        dataset.add_transforms(EmbedInGrid(), upscale_and_salt)
        dataset.add_transforms(EmbedInGrid(), PoissonNoise(), perspective_transform)
        dataset.add_transforms(EmbedInGrid(), GrainNoise())
        dataset.apply_transforms(keep=False)

        dataset.add_transforms(JPEGEncode())
        dataset.add_transforms(perspective_transform, JPEGEncode())
        dataset.apply_transforms()

    save_datsets(list(zip([concat_machine, concat_hand, concat_out],
                          ["concat_machine_dataset", "concat_hand_dataset", "concat_out_dataset"])))
    return concat_hand, concat_machine, concat_out


def load_datasets(file_names=None):
    if file_names is None:
        file_names = ["concat_machine_dataset", "concat_hand_dataset", "concat_out_dataset"]

    machine = CharacterDataset(28)
    hand = CharacterDataset(28)
    out = CharacterDataset(28)

    for dataset, name in tqdm(zip([machine, hand, out], file_names), desc="Loading datasets from file", total=3):
        with h5py.File(f"datasets/{name}.hdf5", "r") as f:
            dataset.train_x = f["train_x"][:]
            dataset.train_y = f["train_y"][:]
            dataset.test_x = f["test_x"][:]
            dataset.test_y = f["test_y"][:]

    return hand, machine, out


def load_validation(to_simple_digit=False):
    labels = []
    images = []
    for i in range(10):
        base_path = Path("datasets/validation/") / str(i)
        labels_path = base_path / "labels.json"
        if base_path.exists() and base_path.is_dir() and labels_path.exists():
            with open(labels_path, 'r', encoding="utf8") as fp:
                _labels = json.load(fp)["labels"]
                if len(_labels) != 81:
                    print(f"{labels_path}: len={len(labels)}", file=sys.stderr)
                    continue
                labels.extend(_labels)
            for i in range(81):
                img_path = base_path / f"box_{i}.png"
                if not img_path.exists():
                    print(f"{img_path} does not exist", file=sys.stderr)
                    raise RuntimeError
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_LANCZOS4)
                images.append(img)

    images = np.expand_dims(np.array(images, dtype=np.uint8), axis=3)
    labels = np.array(labels, dtype=int)
    assert images.shape[0] == labels.shape[0]

    num_classes = 20
    if to_simple_digit:
        indices = np.logical_and(labels > 9, labels != 10)
        labels[indices] -= 10
        num_classes = 11

    labels = keras.utils.to_categorical(labels, num_classes=num_classes)

    return images, labels


def save_datsets(datasets):
    for dataset, name in tqdm(datasets, desc="Writing datasets to file"):
        with h5py.File(f"datasets/{name}.hdf5", "w") as f:
            f.create_dataset("train_x", data=dataset.train_x)
            f.create_dataset("train_y", data=dataset.train_y)
            f.create_dataset("test_x", data=dataset.test_x)
            f.create_dataset("test_y", data=dataset.test_y)


def create_data_overview(samples=(20, 20)):
    concat_hand, concat_machine, concat_out = load_datasets()
    concat_all = ConcatDataset([concat_hand, concat_machine, concat_out], delete=False)

    non_zero_classes = set()
    for i in range(samples[0]):
        if concat_all.train_indices_by_number[i].size > 0: non_zero_classes.add(i)
    indices = np.array(
        [np.random.choice(concat_all.train_indices_by_number[i], samples[1]) for i in non_zero_classes]
    ).reshape(-1)
    image = concat_all.train_x[indices] \
        .reshape(len(non_zero_classes), samples[1], 28, 28) \
        .swapaxes(1, 2) \
        .reshape(len(non_zero_classes) * 28, samples[1] * 28)
    cv2.imwrite(f"train_samples.png", image)

    non_zero_classes = set()
    for i in range(samples[0]):
        if concat_all.test_indices_by_number[i].size > 0: non_zero_classes.add(i)
    indices = np.array(
        [np.random.choice(concat_all.test_indices_by_number[i], samples[1]) for i in non_zero_classes]
    ).reshape(-1)
    image = concat_all.test_x[indices] \
        .reshape(len(non_zero_classes), samples[1], 28, 28) \
        .swapaxes(1, 2) \
        .reshape(len(non_zero_classes) * 28, samples[1] * 28)
    cv2.imwrite(f"test_samples.png", image)


def get_labels(y_true, y_pred):
    return [f"{y_pred[i]}" if y_pred[i] == y_true[i] else f"{y_pred[i]}/{y_true[i]}" for i in range(y_true.shape[0])]


def plot_9x9_grid(zipped, title):
    plt.tight_layout(0.1, rect=(0.2, 0.2, 1, 1))
    fig, axes = plt.subplots(9, 9, figsize=(9, 12))
    fig.suptitle(title, y=0.995)
    tuples = iter(zipped)
    for i in range(9):
        for j in range(9):
            img, label = tuples.__next__()
            axes[i][j].imshow(img.squeeze(), cmap="gray")
            axes[i][j].axis('off')
            axes[i][j].set_title(str(label))
    plt.show()


# SGD or Adam work well
def get_linear_model(n_classes=18):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(28 ** 2,)))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
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
        shuffle=True
    )

    test_generator = BalancedDataGenerator(
        concat_machine.test, concat_hand.test, concat_out.test,
        batch_size=batch_size,
        shuffle=True
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


# Adadelta or Adagrad work well
def get_cnn_model(n_classes=18):
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu',
                     input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(n_classes, activation='softmax'))
    return model


def train_cnn(to_simple_digit=True):
    print("Loading data..")
    concat_hand, concat_machine, concat_out = load_datasets()

    batch_size = 192
    train_generator = SimpleDataGenerator(
        # concat_machine.train, concat_hand.train,
        concat_machine.train, concat_hand.train, concat_out.train,
        batch_size=batch_size,
        shuffle=True,
        # data_align=1
        to_simple_digit=to_simple_digit
    )
    unique, coeffs = np.unique(train_generator.labels, return_counts=True)
    coeffs = dict(zip(unique, coeffs.astype(np.float) / np.sum(coeffs)))

    test_generator = SimpleDataGenerator(
        # concat_machine.test, concat_hand.test,
        concat_machine.test, concat_hand.test, concat_out.test,
        batch_size=batch_size,
        shuffle=True,
        to_simple_digit=to_simple_digit
    )

    steps_per_epoch = len(train_generator)
    validation_steps = len(test_generator)

    # Run training on the GPU
    with tf.device('/GPU:0'):
        # Keras Model
        print("Creating model..")
        model = get_cnn_model(n_classes=train_generator.num_classes)

        print("Compiling model..")
        model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adagrad(),
            metrics=[keras.metrics.categorical_accuracy],
            weighted_metrics=['accuracy']
        )
        print(model.summary())

        print("Starting training..")
        model.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=5,
            validation_data=test_generator,
            validation_steps=validation_steps,
            use_multiprocessing=True,
            workers=8,
            class_weight=coeffs
        )

        x, y = train_generator[0]
        y_true = np.argmax(y, axis=-1)
        y_pred = model.predict_classes(x)
        y = get_labels(y_true, y_pred)
        plot_9x9_grid(list(zip(x, y))[:81], "Training sample")

        x, y = test_generator[0]
        y_true = np.argmax(y, axis=-1)
        y_pred = model.predict_classes(x)
        y = get_labels(y_true, y_pred)
        plot_9x9_grid(list(zip(x, y))[:81], "Development sample")

        evaluate(model, to_simple_digit)


def evaluate(model: Sequential, to_simple_digit=False):
    print("Evaluating")
    x, y = load_validation(to_simple_digit)
    print(dict(zip(model.metrics_names, model.evaluate(x, y))))
    y_true = np.argmax(y, axis=-1)
    y_pred = model.predict_classes(x)
    y = get_labels(y_true, y_pred)
    zipped = list(zip(x, y))
    for idx in range(0, len(zipped), 81):
        plot_9x9_grid(zipped[idx:idx + 81], f"Validation set {idx // 81}")


if __name__ == '__main__':
    # CharacterRenderer().prerender_all(mode='L')
    create_datasets()
    create_data_overview()
    train_cnn()
