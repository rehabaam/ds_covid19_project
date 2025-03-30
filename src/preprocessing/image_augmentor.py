# -*- coding: utf-8 -*-

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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
