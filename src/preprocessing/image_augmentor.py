# -*- coding: utf-8 -*-
from random import randrange

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def generate_augmented_images_binaryclass(images, samples, random_seed) -> np.array:
    """
    generate_augmented_images_binaryclass function generates
    augmented images using ImageDataGenerator.

    Input:
    images: np.array: Images as a numpy array
    samples: int: Number of augmented images to generate
    random_seed: int: Random seed for reproducibility
    """

    if len(images) <= 0:
        raise TypeError(f"No images to augment, got : {len(images)} images")

    if samples <= 0 or not isinstance(samples, int):
        raise TypeError(f"Wrong input for samples: {samples}")

    # Define an ImageDataGenerator for augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,  # Rotate images by up to 15 degrees
        width_shift_range=0.1,  # Shift width by 10%
        height_shift_range=0.1,  # Shift height by 10%
        shear_range=0.1,  # Apply shearing
        zoom_range=0.1,  # Zoom in/out
        horizontal_flip=True,  # Flip images horizontally
        fill_mode="nearest",  # Fill missing pixels
        dtype="uint8",  # Data type
    )

    # Get the length of the images
    len_images = len(images)

    # Select a random image
    for i in range(samples):
        if random_seed == 42:
            image = images[random_seed + i]
        else:
            random_seed = randrange(0, len_images, 1)
            image = images[random_seed]

        image = np.expand_dims(image, axis=0)
        augmented_images = datagen.flow(image, batch_size=1)
        images.append(next(augmented_images)[0])

    return images


def generate_augmented_images_multiclass(path, image_size, batch_size) -> tuple:
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
    )

    val_generator = datagen.flow_from_directory(
        path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        color_mode="grayscale",
        subset="validation",
        seed=42,
    )

    class_labels = train_generator.classes
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(class_labels), y=class_labels
    )
    class_weight_dict = dict(enumerate(class_weights))

    print(f"Computed Class Weights:{class_weight_dict} labels: {train_generator.class_indices}")

    return train_generator, val_generator, class_weight_dict
