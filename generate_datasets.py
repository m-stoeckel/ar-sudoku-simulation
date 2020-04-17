import os
from typing import Iterable, Tuple

import cv2
import h5py
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from simulation import PrerenderedDigitDataset, PrerenderedCharactersDataset, ConcatDataset, \
    ClassSeparateCuratedCharactersDataset, ClassSeparateMNIST, EmptyDataset, RealDataset, RealValidationDataset, \
    CharacterDataset, RandomPerspectiveTransform, RescaleIntermediateTransforms, JPEGEncode, \
    SaltAndPepperNoise, Dilate, EmbedInRectangle, EmbedInGrid, GrainNoise, PoissonNoise

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
        concat_machine = ConcatDataset(digit_dataset, prerendered_digit_dataset)
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
        concat_hand = ConcatDataset(mnist, curated_digits)
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
        concat_out = ConcatDataset(empty_dataset)

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
            dataset.apply_transforms(keep=False)  # -> 20086 images in train split

            dataset.add_transforms(upscale_and_salt)
            dataset.add_transforms(GrainNoise())
            dataset.apply_transforms()  # -> 60258 images in train split

            dataset.add_transforms(perspective_transform)
            dataset.add_transforms(perspective_transform, JPEGEncode())
            dataset.add_transforms(downscale_intermediate_transforms)
            dataset.add_transforms(PoissonNoise(), JPEGEncode())
            dataset.add_transforms(JPEGEncode())
            dataset.apply_transforms()  # -> 361548 images in train split

        save_datsets([(concat_machine, "train_machine_dataset")])
        print(f"Created {concat_machine.test_y.size}/{concat_machine.train_y.size} machine images")

    if not os.path.exists(f"datasets/train_hand_dataset.hdf5"):
        # Apply some transforms to other digits
        print("Applying transforms to handwritten digits")
        for dataset in [concat_hand]:
            dataset.add_transforms(EmbedInRectangle())
            dataset.add_transforms(EmbedInGrid())
            dataset.apply_transforms(keep=False)  # -> 124748 images in train split

            dataset.add_transforms(upscale_and_salt, perspective_transform, JPEGEncode())
            dataset.add_transforms(GrainNoise(), perspective_transform)
            dataset.apply_transforms()  # -> 374244 images in train split

        save_datsets([(concat_hand, "train_hand_dataset")])
        print(f"Created {concat_hand.test_y.size}/{concat_hand.train_y.size} handwritten images")

    if not os.path.exists(f"datasets/train_out_dataset.hdf5"):
        print("Applying transforms to out images")
        for dataset in [concat_out]:
            dataset.add_transforms(EmbedInGrid(), upscale_and_salt)
            dataset.add_transforms(EmbedInGrid(), GrainNoise())
            dataset.add_transforms(EmbedInRectangle())
            dataset.apply_transforms(keep=False)  # -> 32400 images in train split

            dataset.add_transforms(downscale_intermediate_transforms)
            dataset.add_transforms(perspective_transform, JPEGEncode())
            dataset.add_transforms(JPEGEncode())
            dataset.apply_transforms(keep=False)  # -> 97200 images in train split

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
    concat_machine, concat_hand, concat_out, real = load_datasets(TRANSFORMED_DATASET_NAMES[:3] + ["base_real_dataset"])

    dataset = ConcatDataset(concat_machine, concat_hand, concat_out, delete=False)
    render_overview(dataset.train_x, dataset.train_indices_by_number, samples, "docs/source/_static/train_samples.png")
    render_overview(dataset.test_x, dataset.test_indices_by_number, samples, "docs/source/_static/test_samples.png")

    dataset = ConcatDataset(real, delete=False)
    render_overview(dataset.train_x, dataset.train_indices_by_number, samples, "docs/source/_static/train_real.png")
    render_overview(dataset.test_x, dataset.test_indices_by_number, samples, "docs/source/_static/test_real.png")


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


if __name__ == '__main__':
    # CharacterRenderer().prerender_all(mode='L')
    # generate_base_datasets()
    # generate_transformed_datasets()
    create_data_overview()
