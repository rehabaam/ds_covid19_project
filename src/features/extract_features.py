# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
import builtins
import os

import cv2
import numpy as np
from scipy.stats import kurtosis, skew
from skimage.feature.texture import graycomatrix, graycoprops

from src.preprocessing.image_preprocessor import crop_image


# Function to extract basic statistical features from an image
def extract_features(image):
    """
    extract_features extracts statistical features from a chest X-ray image.

    Input:
    image: np.array: Image as a numpy array

    Output:
    features: np.array: Extracted features
    """

    # Compute descriptive statistics
    mean = np.mean(image)
    std_dev = np.std(image)
    var = np.var(image)
    skewness = skew(image.flatten())
    kurt = kurtosis(image.flatten())

    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    hist_mean = np.mean(hist)
    hist_std = np.std(hist)

    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    edge_mean = np.mean(sobelx) + np.mean(sobely)
    edge_var = np.var(sobelx) + np.var(sobely)

    glcm = graycomatrix(image, [1], [0], 256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, "contrast")[0, 0]
    dissimilarity = graycoprops(glcm, "dissimilarity")[0, 0]
    homogeneity = graycoprops(glcm, "homogeneity")[0, 0]
    energy = graycoprops(glcm, "energy")[0, 0]
    entropy = graycoprops(glcm, "entropy")[0, 0]

    # Combine features into a vector
    features = [
        mean,
        std_dev,
        var,
        skewness,
        kurt,
        hist_mean,
        hist_std,
        edge_mean,
        edge_var,
        contrast,
        dissimilarity,
        homogeneity,
        energy,
        entropy,
    ]

    return features


def get_extracted_features(images_dir, label, image_size, image_resized=False):
    """
    get_extracted_features Loads images from a folder and extracts features.

    Input:
    images_dir: str: Path to the folder containing images
    label: int: Label for the images
    image_size: int: Size of the image
    image_resized: bool: Resize the image

    Output:
    feature_list: np.array: List of extracted features
    """
    features = []
    labels = []
    image_list = []

    for filename in os.listdir(images_dir):
        if filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            image = crop_image(os.path.join(images_dir, filename), 0)
            if image_resized:
                img_resized = (
                    cv2.resize(image[1], (image_size, image_size)) / 255.0
                )
                image_list.append(img_resized)
            features.append(extract_features(image[1]))
            labels.append(label)

    return features, labels, image_list


def load_extracted_features(
    images_dir, category, dataset_label, image_size=128, image_resized=False
):
    """
    load_extracted_features Loads images from a folder and extracts features.

    Input:
    images_dir: str: Path to the folder containing images
    category: str or list: Category of the images
    label: int: Label for the images
    image_size: int: Size of the image (Default is 128)
    image_resized: bool: Resize the image

    Output:
    features: np.array: List of extracted features
    labels: np.array: List of labels
    """
    features = []
    labels = []
    image_list = []

    match type(category):
        case builtins.str:
            images_dir = images_dir.replace("{}", category)
            features, labels, image_list = get_extracted_features(
                images_dir, dataset_label, image_size, image_resized
            )
        case builtins.list:
            for cat in category:
                feature, label, image = get_extracted_features(
                    images_dir.replace("{}", cat),
                    dataset_label,
                    image_size,
                    image_resized,
                )
                features.extend(feature)
                labels.extend(label)
                image_list.extend(image)
        case _:
            raise TypeError("Wrong category used")

    print(
        "Loaded images for {}: {} resized images, {} features, and {} labels.".format(
            category, len(image_list), len(features), len(labels)
        )
    )
    return np.array(features), np.array(labels), np.array(image_list)
