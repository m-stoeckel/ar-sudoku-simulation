import h5py
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import *
from sklearn.metrics import classification_report

from simulation.digit import BalancedDataGenerator
from simulation.digit.data_generator import SimpleDataGenerator, ToBinaryGenerator, BaseDataGenerator
from simulation.digit.dataset import *
from simulation.transforms import *

tf.get_logger().setLevel('ERROR')
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


BASE_DATASET_NAMES = ["base_machine_dataset", "base_hand_dataset", "base_out_dataset",
                      "base_real_dataset", "validation_real_dataset"]
TRANSFORMED_DATASET_NAMES = ["train_machine_dataset", "train_hand_dataset", "train_out_dataset",
                             "train_real_dataset", "validation_real_dataset"]


def generate_base_datasets():
    digit_characters = "123456789"
    non_digit_characters = "0abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.!?"

    #########################
    # Machinewritten digits #
    #########################
    if not os.path.exists(f"datasets/base_machine_dataset.hdf5"):
        # "Digit" dataset
        digit_dataset = PrerenderedDigitDataset(digits_path="datasets/digits/")
        digit_dataset.resize(28)

        # Prerendered character dataset - digits
        prerendered_digit_dataset = PrerenderedCharactersDataset(
            digits_path="datasets/characters/",
            load_chars=digit_characters
        )
        prerendered_digit_dataset.resize(28)

        # Save base datasets for later
        concat_machine = ConcatDataset([digit_dataset, prerendered_digit_dataset])
        save_datsets([(concat_machine, "base_machine_dataset")])

    ######################
    # Handwritten digits #
    ######################
    if not os.path.exists(f"datasets/base_hand_dataset.hdf5"):
        # Curated digits
        curated_digits = ClassSeparateCuratedCharactersDataset(digits_path="datasets/curated/",
                                                               load_chars=digit_characters)
        curated_digits.resize(28)

        # Mnist digits
        mnist = ClassSeparateMNIST(data_home="datasets/")

        # Save base datasets for later
        concat_hand = ConcatDataset([mnist, curated_digits])
        save_datsets([(concat_hand, "base_hand_dataset")])

    ##############
    # Non-digits #
    ##############
    if not os.path.exists(f"datasets/base_out_dataset.hdf5"):
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
        empty_dataset = EmptyDataset(28, 12000)

        # Concatenate datasets
        # concat_out = ConcatDataset([curated_out, prerendered_nondigit_dataset, empty_dataset])
        # concat_out = ConcatDataset([curated_out, empty_dataset])
        concat_out = ConcatDataset([empty_dataset])

        # Save base datasets for later
        save_datsets([(concat_out, "base_out_dataset")])

    if not all(os.path.exists(f"datasets/{name}.hdf5") for name in BASE_DATASET_NAMES[3:]):
        #############
        # Real data #
        #############

        real_training = RealDataset("datasets/real")
        real_validation = RealValidationDataset("datasets/validation")

        # Save base datasets for later
        save_datsets([(real_training, "base_real_dataset"),
                      (real_validation, "validation_real_dataset")])


def generate_transformed_datasets():
    if not all(os.path.exists(f"datasets/{name}.hdf5") for name in BASE_DATASET_NAMES):
        generate_base_datasets()

    concat_machine, concat_hand, concat_out, real_dataset = load_datasets(BASE_DATASET_NAMES[:4])

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

    if not os.path.exists(f"datasets/train_machine_dataset.hdf5"):
        # Apply many transforms to machine digits
        print("Applying transforms to machine digits")
        for dataset in [concat_machine]:
            dataset.add_transforms(EmbedInRectangle())
            dataset.add_transforms(EmbedInGrid())
            dataset.apply_transforms(keep=False)  # -> 20086 images

            dataset.add_transforms(upscale_and_salt)
            dataset.add_transforms(GrainNoise())
            dataset.apply_transforms()  # -> 60258 images

            dataset.add_transforms(perspective_transform)
            dataset.add_transforms(perspective_transform, JPEGEncode())
            dataset.add_transforms(downscale_intermediate_transforms)
            dataset.add_transforms(PoissonNoise(), JPEGEncode())
            dataset.add_transforms(JPEGEncode())
            dataset.apply_transforms()  # -> 361548 images

        save_datsets([(concat_machine, "train_machine_dataset")])
        print(f"Created {concat_machine.test_y.size}/{concat_machine.train_y.size} machine images")

    if not os.path.exists(f"datasets/train_hand_dataset.hdf5"):
        # Apply some transforms to other digits
        print("Applying transforms to handwritten digits")
        for dataset in [concat_hand]:
            dataset.add_transforms(EmbedInRectangle())
            dataset.add_transforms(EmbedInGrid())
            dataset.apply_transforms(keep=False)  # -> 124748 images

            dataset.add_transforms(upscale_and_salt, perspective_transform, JPEGEncode())
            dataset.add_transforms(GrainNoise(), perspective_transform)
            dataset.apply_transforms()  # -> 374244 images

        save_datsets([(concat_hand, "train_hand_dataset")])
        print(f"Created {concat_hand.test_y.size}/{concat_hand.train_y.size} handwritten images")

    if not os.path.exists(f"datasets/train_out_dataset.hdf5"):
        print("Applying transforms to out images")
        for dataset in [concat_out]:
            dataset.add_transforms(EmbedInGrid(), upscale_and_salt)
            dataset.add_transforms(EmbedInGrid(), GrainNoise())
            dataset.add_transforms(EmbedInRectangle())
            dataset.apply_transforms(keep=False)  # -> 13500

            dataset.add_transforms(downscale_intermediate_transforms)
            dataset.add_transforms(perspective_transform, JPEGEncode())
            dataset.add_transforms(JPEGEncode())
            dataset.apply_transforms(keep=False)  # -> 40500 images

        save_datsets([(concat_out, "train_out_dataset")])
        print(f"Created {concat_out.test_y.size}/{concat_out.train_y.size} out images")

    if not os.path.exists(f"datasets/train_real_dataset.hdf5"):
        print("Applying transforms to real images")
        for dataset in [real_dataset]:
            dataset.add_transforms(JPEGEncode())
            dataset.add_transforms(perspective_transform, JPEGEncode())
            dataset.apply_transforms()  # -> 14433 images

        save_datsets([(real_dataset, "train_real_dataset")])
        print(f"Created {real_dataset.test_y.size}/{real_dataset.train_y.size} real images")


def load_datasets(file_names):
    datasets = []

    for name in tqdm(file_names, desc="Loading datasets from file", total=len(file_names)):
        dataset = CharacterDataset(28)
        with h5py.File(f"datasets/{name}.hdf5", "r") as f:
            dataset.train_x = f["train_x"][:]
            dataset.train_y = f["train_y"][:]
            dataset.test_x = f["test_x"][:]
            dataset.test_y = f["test_y"][:]
        datasets.append(dataset)

    return datasets


def save_datsets(datasets: Iterable[Tuple[CharacterDataset, str]]):
    for dataset, name in tqdm(datasets, desc="Writing datasets to file"):
        with h5py.File(f"datasets/{name}.hdf5", "w") as f:
            f.create_dataset("train_x", data=dataset.train_x)
            f.create_dataset("train_y", data=dataset.train_y)
            f.create_dataset("test_x", data=dataset.test_x)
            f.create_dataset("test_y", data=dataset.test_y)


def create_data_overview(samples=20):
    concat_machine, concat_hand, concat_out, train, _ = load_datasets(TRANSFORMED_DATASET_NAMES)

    dataset = ConcatDataset([concat_machine, concat_hand, concat_out], delete=False)
    render_overview(dataset.train_x, dataset.train_indices_by_number, samples, "train_samples.png")
    render_overview(dataset.test_x, dataset.test_indices_by_number, samples, "test_samples.png")

    dataset = ConcatDataset([train], delete=False)
    render_overview(dataset.train_x, dataset.train_indices_by_number, samples, "train_real.png")
    render_overview(dataset.test_x, dataset.test_indices_by_number, samples, "test_real.png")


def render_overview(data, indices_by_number, samples, filename):
    non_zero_classes = set()
    for i in indices_by_number.keys():
        if indices_by_number[i].size > 0:
            non_zero_classes.add(i)
    indices = np.array([np.random.choice(indices_by_number[i], samples) for i in non_zero_classes]).reshape(-1)

    class_images = np.empty((len(non_zero_classes) * 28, 28))
    for i, clazz in enumerate(non_zero_classes):
        img = np.full((128, 128), 255, dtype=np.uint8)
        img = cv2.putText(img, str(clazz), (0, 112), cv2.FONT_HERSHEY_SIMPLEX, 3, 0, 6, cv2.LINE_AA)
        class_images[i * 28:(i + 1) * 28, :] = cv2.resize(img, (28, 28), interpolation=cv2.INTER_LANCZOS4)

    image = data[indices] \
        .reshape(len(non_zero_classes), samples, 28, 28) \
        .swapaxes(1, 2) \
        .reshape(len(non_zero_classes) * 28, samples * 28)

    image = np.hstack((class_images, image))
    cv2.imwrite(filename, image)


def get_labels(y_true, y_pred):
    return [f"{y_pred[i]}" if y_pred[i] == y_true[i] else f"{y_pred[i]}/{y_true[i]}" for i in range(y_true.shape[0])]


def plot_9x9_grid(zipped, title: str):
    plt.tight_layout(0.1, rect=(0, 0, 0.8, 1))
    fig, axes = plt.subplots(9, 9, figsize=(9, 12))
    fig.suptitle(title, y=0.995)
    tuples = iter(zipped)
    for i in range(9):
        for j in range(9):
            img, label = tuples.__next__()
            axes[i][j].imshow(img.squeeze(), cmap="gray")
            axes[i][j].axis('off')
            axes[i][j].set_title(str(label))
    plt.savefig(f"{title.replace(' ', '_')}.png")
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
    concat_machine, concat_hand, concat_out, real_training, real_validation = load_datasets(TRANSFORMED_DATASET_NAMES)

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
def get_cnn_max_model(n_classes=18):
    model = Sequential()
    model.add(Conv2D(16, (3, 3),
                     input_shape=(28, 28, 1)))  # 26x26x16
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
    model.add(Dense(n_classes, activation='softmax'))
    return model


# Adadelta or Adagrad work well
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


def train_cnn(to_simple_digit=True):
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
        model = get_cnn_max_model(n_classes=train_generator.num_classes)

        # Hyperparameters
        epochs = 20
        ft_epochs = 20
        learning_rate = 0.01

        print("Compiling model..")
        model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adagrad(lr=learning_rate),
            metrics=[keras.metrics.categorical_accuracy],
        )
        print(model.summary())

        print("Training model")
        model.fit_generator(
            train_generator, validation_data=dev_generator,
            epochs=epochs,
            use_multiprocessing=True,
            workers=8,
            callbacks=[
                ModelCheckpoint("model/cnn_model.{epoch:02d}.hdf5", period=2),
                EarlyStopping(monitor='val_categorical_accuracy', restore_best_weights=True),
            ]
        )

        print("Finetuning model")
        model.fit_generator(
            ft_train_generator, validation_data=ft_train_generator,
            epochs=ft_epochs,
            use_multiprocessing=True,
            workers=8,
            callbacks=[
                ModelCheckpoint("model/cnn_model.ft.{epoch:02d}.hdf5", period=2),
                EarlyStopping(monitor='val_categorical_accuracy', restore_best_weights=True),
            ]
        )

        model.save("model/cnn_model.ft.final.hdf5")

        print("Evaluating")
        print(list(zip(model.metrics_names, model.evaluate_generator(dev_generator))))
        print(list(zip(model.metrics_names, model.evaluate_generator(ft_dev_generator))))
        evaluate_and_plot(model, test_generator)


def train_binary_model():
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
    with tf.device('/CPU:0'):
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
                ModelCheckpoint("model/binary_model.{epoch:02d}.hdf5", period=2),
                EarlyStopping(monitor='val_categorical_accuracy', restore_best_weights=True),
            ]
        )

        print("Finetuning model")
        model.fit_generator(
            ft_train_generator, validation_data=ft_train_generator,
            epochs=ft_epochs,
            use_multiprocessing=True,
            workers=8,
            callbacks=[
                ModelCheckpoint("model/binary_model.ft.{epoch:02d}.hdf5", period=2),
                EarlyStopping(monitor='val_categorical_accuracy', restore_best_weights=True),
            ]
        )

        model.save("model/binary_model.ft.final.hdf5")

        print("Evaluating")
        print(list(zip(model.metrics_names, model.evaluate_generator(dev_generator))))
        print(list(zip(model.metrics_names, model.evaluate_generator(ft_dev_generator))))
        print(list(zip(model.metrics_names, model.evaluate_generator(test_generator))))
        evaluate_and_plot(model, test_generator)


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
    model = keras.models.load_model("model/cnn_model.ft.final.hdf5")
    concat_machine, concat_hand, concat_out, real_training, real_validation = load_datasets(TRANSFORMED_DATASET_NAMES)

    train_generator = SimpleDataGenerator(
        concat_machine.train, concat_hand.train, concat_out.train,
        batch_size=64,
        shuffle=True,
        to_simple_digit=True
    )

    dev_generator = SimpleDataGenerator(
        concat_machine.test, concat_hand.test, concat_out.test,
        batch_size=64,
        shuffle=True,
        to_simple_digit=True
    )

    ft_dev_generator = SimpleDataGenerator(
        real_training.test,
        batch_size=64,
        shuffle=True,
        to_simple_digit=True
    )

    test_generator = SimpleDataGenerator(
        real_validation.test,
        batch_size=64,
        shuffle=False,
        to_simple_digit=True
    )

    evaluate(model, train_generator)
    evaluate(model, dev_generator)
    evaluate(model, ft_dev_generator)
    evaluate(model, test_generator)


if __name__ == '__main__':
    # CharacterRenderer().prerender_all(mode='L')
    # generate_base_datasets()
    # generate_transformed_datasets()
    # create_data_overview()
    # train_binary_model()
    train_cnn()
    # load_and_evaluate()
