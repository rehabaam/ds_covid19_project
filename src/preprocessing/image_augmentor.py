# -*- coding: utf-8 -*-
import os

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    img_to_array,
    load_img,
)
from tensorflow.keras.utils import Sequence, to_categorical


def generate_augmented_images(path, image_size, batch_size) -> tuple:
    """
    generate_augmented_images_multiclass function generates
    augmented images using ImageDataGenerator.

    Input:
    path: str: Path to the images
    image_size: tuple: Size of the images
    batch_size: int: Number of augmented images to generate
    """
    # Define an ImageDataGenerator for augmentation
    datagen = ImageDataGenerator(
        dtype="uint8",  # Data type
        validation_split=0.2,
    )

    train_generator = datagen.flow_from_directory(
        path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        color_mode="grayscale",
        subset="training",
        seed=42,
        shuffle=True,
    )

    val_generator = datagen.flow_from_directory(
        path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        color_mode="grayscale",
        subset="validation",
        seed=42,
        shuffle=True,
    )

    class_labels = train_generator.classes
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(class_labels), y=class_labels
    )
    class_weight_dict = dict(enumerate(class_weights))

    print(f"Computed Class Weights:{class_weight_dict} labels: {train_generator.class_indices}")

    return train_generator, val_generator, class_weight_dict


class LungMaskGenerator(Sequence):
    """
    LungMaskGenerator is a custom data generator for loading and augmenting images and masks.
    It inherits from the Keras Sequence class to allow for easy integration with Keras models.
    """

    def __init__(
        self, image_paths, mask_paths, labels, batch_size=32, image_size=(256, 256), shuffle=True
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.labels = labels
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        idxs = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        batch_images = []
        batch_labels = []

        for i in idxs:
            img = load_img(self.image_paths[i], color_mode="grayscale", target_size=self.image_size)
            img = img_to_array(img) / 255.0

            mask = load_img(self.mask_paths[i], color_mode="grayscale", target_size=self.image_size)
            mask = img_to_array(mask) / 255.0

            # Concatenate image and mask: shape will be (H, W, 2)
            combined = np.concatenate([img, mask], axis=-1)
            batch_images.append(combined)
            batch_labels.append(self.labels[i])

        return np.array(batch_images), np.array(batch_labels)

    def get_class_labels(self):
        return np.argmax(self.labels, axis=1)


def get_image_mask_pairs(image_root, mask_root, classes):
    """
    get_image_mask_pairs function takes the root directories of images and masks
    and returns lists of image paths, mask paths, and their corresponding labels.
    Input:
    image_root: str: Root directory for images
    mask_root: str: Root directory for masks
    classes: list: List of class names

    Output:
    image_paths: list: List of image paths
    mask_paths: list: List of mask paths
    labels: list: List of labels corresponding to the images
    """
    # Ensure the root directories exist
    image_paths = []
    mask_paths = []
    labels = []

    for label_idx, class_name in enumerate(classes):
        image_class_dir = os.path.join(image_root, class_name)
        mask_class_dir = os.path.join(mask_root, class_name)

        # List all image files in the class directory
        for fname in os.listdir(image_class_dir):
            image_path = os.path.join(image_class_dir, fname)
            mask_path = os.path.join(mask_class_dir, fname)

            # Check that the corresponding mask exists
            if os.path.exists(mask_path):
                image_paths.append(image_path)
                mask_paths.append(mask_path)
                labels.append(label_idx)
            else:
                print(f"Warning: No mask found for {image_path}")

    return image_paths, mask_paths, labels


def generate_augmented_images_masks(
    train_img, val_img, train_mask, val_mask, train_lbl, val_lbl, classes
) -> tuple:
    """
    generate_augmented_images_masks function generates
    augmented images and masks using ImageDataGenerator.

    Input:
    path: str: Path to the images
    image_size: tuple: Size of the images

    batch_size: int: Number of augmented images to generate
    """

    train_lbl_one_hot = to_categorical(train_lbl, num_classes=4)
    val_lbl_one_hot = to_categorical(val_lbl, num_classes=4)

    # Define an ImageDataGenerator for augmentation
    train_gen = LungMaskGenerator(train_img, train_mask, train_lbl_one_hot)
    val_gen = LungMaskGenerator(val_img, val_mask, val_lbl_one_hot)

    class_weight_dict = compute_class_weight("balanced", classes=np.unique(classes), y=classes)
    class_weight_dict = dict(enumerate(class_weight_dict))
    print(f"Class weights: {class_weight_dict}")

    return train_gen, val_gen, class_weight_dict
