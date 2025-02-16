# -*- coding: utf-8 -*-
from random import randrange

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def generate_augmented_images(images, samples, random_seed) -> np.array:
    """
    generate_augmented_images function generates augmented images
    using ImageDataGenerator.

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

    # Select a random image
    for i in range(samples):
        if random_seed == 42:
            image = images[random_seed + i]
        else:
            random_seed = randrange(0, len(images), 1)
            image = images[random_seed]

        image = np.expand_dims(image, axis=0)
        augmented_images = datagen.flow(image, batch_size=1)
        images.append(next(augmented_images)[0])

    return images
