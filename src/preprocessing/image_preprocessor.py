# -*- coding: utf-8 -*-
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def crop_image(image_path, margin_percentage=10):
    """
    crop_image function takes an image path as input
    and returns the cropped image.

    Input:
    image_path: str: Path to the image file
    margin_percentage: int: Percentage of the image
    to be cropped from all sides

    Output:
    cropped_image: np.array: Cropped image as a numpy array
    image: np.array: Original image as a numpy array
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    margin_x = int(width * margin_percentage / 100)
    margin_y = int(height * margin_percentage / 100)

    cropped_image = gray[margin_y : height - margin_y, margin_x : width - margin_x]

    return cropped_image, gray


def apply_image_mask(image_path, mask_path, target=""):
    """
    apply_image_mask function plots the image statistics.

    Input:
    image_path: str: Path to the image file
    mask_path: str: Path to the mask file
    target: str: Target to be masked

    Output:
    masked_image: np.array: Masked image as a numpy array
    """
    image = cv2.imread(image_path)

    mask = cv2.imread(mask_path)
    mask = cv2.resize(mask, image.shape[:2])
    mask = mask = (mask > 127).astype(np.uint8)

    return mask * image


def calulate_image_statistics(filename, image):
    """
    calulate_image_statistics function calculates
    the statistics of an image.

    Input:
    image: np.array: Image as a numpy array
    filename: str: Name of the image file
    """
    return [
        filename[:-4],
        np.min(image),
        np.max(image),
        np.mean(image),
        np.median(image),
        np.std(image),
    ]


def get_images_statistics(images_dir, margin_percentage=0):
    """
    get_images_statustics function returns the number
    of images in the directory.

    Input:
    images_dir: str: Path to the directory containing images
    margin_percentage: int: Percentage of the image to
    be cropped from all sides
    """
    data = []
    for filename in os.listdir(images_dir):
        if filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            image = crop_image(os.path.join(images_dir, filename), margin_percentage)
            data.append(calulate_image_statistics(filename, image[0]))
    return pd.DataFrame(data, columns=["image", "min", "max", "mean", "median", "std"])


def get_edges_images_statistics(images_dir, margin_percentage=0):
    """
    get_images_edges_statistics function returns the
    number of images in the directory.

    Input:
    images_dir: str: Path to the directory containing images
    margin_percentage: int: Percentage of the image
    to be cropped from all sides
    """
    data = []
    ddept = cv2.CV_8U
    for filename in os.listdir(images_dir):
        if filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            image = crop_image(os.path.join(images_dir, filename), margin_percentage)
            x = cv2.Sobel(image[0], ddept, 1, 0, ksize=3, scale=1)
            y = cv2.Sobel(image[0], ddept, 0, 1, ksize=3, scale=1)
            edge = cv2.addWeighted(cv2.convertScaleAbs(x), 0.5, cv2.convertScaleAbs(y), 0.5, 0)
            data.append(calulate_image_statistics(filename, edge))
    return pd.DataFrame(data, columns=["image", "min", "max", "mean", "median", "std"])


def get_masked_images_statistics(images_dir, mask_dir):
    """
    get_masked_images_statistics function returns the
    stastics of masked images in the directory.

    Input:
    images_dir: str: Path to the directory containing images
    mask_dir: str: Path to the directory containing masks
    margin_percentage: int: Percentage of the image
    to be cropped from all sides
    """
    data = []
    for filename in os.listdir(images_dir):
        if filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            image = apply_image_mask(
                os.path.join(images_dir, filename),
                os.path.join(mask_dir, filename),
            )
            data.append(calulate_image_statistics(filename, image[0]))
    return pd.DataFrame(data, columns=["image", "min", "max", "mean", "median", "std"])


def store_images_statistics(images_data, csv_filename):
    """
    store_images_statistics function stores the image
    statistics in a CSV file.

    Input:
    images_data: pd.DataFrame: DataFrame containing
    image statistics
    csv_filename: str: Name of the CSV file
    """
    images_data.to_csv(csv_filename)


def plot_images_statistics(dataset, stats, no_of_cols=2):
    """
    plot_images_statistics function plots the image statistics.

    Input:
    dataset: str: Name of the dataset
    stats: pd.DataFrame: DataFrame containing image statistics
    no_of_cols: int: Number of columns in the plot
    """

    # Select only numerical columns
    stats = stats.select_dtypes(exclude=["object"])

    # Calculate the number of rows
    no_of_rows = int(len(stats.columns) // 2 + 1)

    # Plot the image statistics
    _, axs = plt.subplots(no_of_rows, no_of_cols, figsize=(10, 10))
    plt.suptitle(f"{dataset} Image Statistics")

    for i in range(len(stats.columns)):
        fig_index = axs[i // no_of_cols, i % no_of_cols] if no_of_rows > 1 else axs
        sns.histplot(data=stats.iloc[:, i], bins=50, kde=True, ax=fig_index)
        fig_index.set_title(stats.columns[i])
        fig_index.set_xlabel("")

    # Remove empty subplots
    if no_of_rows > 1:
        [plt.delaxes(ax) for ax in axs.flatten() if not ax.has_data()]
    # Display the subplots
    plt.tight_layout()
    plt.show()


def normalize_image(image):
    """
    normalize_image function normalizes the image.

    Input:
    image: np.array: Image as a numpy array

    Output:
    normalized_image: np.array: Normalized image as a numpy array
    """
    return (image - np.min(image)) / (np.max(image) - np.min(image))


def standardize_image(image):
    """
    standardize_image function standardizes the image.

    Input:
    image: np.array: Image as a numpy array

    Output:
    standardized_image: np.array: Standardized image as a numpy array
    """
    return (image - np.mean(image)) / np.std(image)


def get_images_statistics_by_scales(images_dir, margin_percentage=0):
    """
    get_images_statustics function returns the number
    of images in the directory.

    Input:
    images_dir: str: Path to the directory containing images
    margin_percentage: int: Percentage of the image to
    be cropped from all sides
    """
    normalized = []
    standardized = []
    for filename in os.listdir(images_dir):
        if filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            image = crop_image(os.path.join(images_dir, filename), margin_percentage)
            normalized_image = normalize_image(image[0])
            normalized.append(calulate_image_statistics(filename, normalized_image))

            standardized_image = standardize_image(image[0])
            standardized.append(calulate_image_statistics(filename, standardized_image))
    return pd.DataFrame(
        normalized, columns=["image", "min", "max", "mean", "median", "std"]
    ), pd.DataFrame(standardized, columns=["image", "min", "max", "mean", "median", "std"])
