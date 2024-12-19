# -*- coding: utf-8 -*-
import os

import cv2
import numpy as np
import pandas as pd
import pytest

from src.preprocessing.image_preprocessor import (
    calulate_image_statistics,
    crop_image,
    get_images_statistics,
    store_images_statistics,
)


@pytest.fixture
def setup_test_images(tmpdir):
    # Create a temporary directory with sample images
    image_dir = tmpdir.mkdir("images")
    for i in range(3):
        image_path = os.path.join(image_dir, f"test_image_{i}.jpg")
        # Create a dummy image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(image_path, image)
    return image_dir


def test_get_images_statistics(setup_test_images):
    image_dir = setup_test_images
    stats_df = get_images_statistics(str(image_dir))
    stats_df.sort_values(by="image", inplace=True)
    # Verify the statistics DataFrame
    assert len(stats_df) == 3
    assert list(stats_df.columns) == [
        "image",
        "min",
        "max",
        "mean",
        "median",
        "std",
    ]
    for i in range(3):
        assert stats_df.iloc[i]["image"] == f"test_image_{i}"
        assert stats_df.iloc[i]["min"] == 0
        assert stats_df.iloc[i]["max"] == 0
        assert stats_df.iloc[i]["mean"] == 0
        assert stats_df.iloc[i]["median"] == 0
        assert stats_df.iloc[i]["std"] == 0


def test_store_images_statistics(tmpdir):
    # Create a sample DataFrame
    data = {
        "image": ["test_image_0", "test_image_1", "test_image_2"],
        "min": [0, 0, 0],
        "max": [0, 0, 0],
        "mean": [0, 0, 0],
        "median": [0, 0, 0],
        "std": [0, 0, 0],
    }
    images_data = pd.DataFrame(data)

    # Define the CSV filename
    csv_filename = os.path.join(tmpdir, "test_statistics.csv")

    # Store the statistics in a CSV file
    store_images_statistics(images_data, csv_filename)

    # Verify the CSV file
    assert os.path.exists(csv_filename)

    # Read the CSV file and verify its content
    stored_data = pd.read_csv(csv_filename, index_col=0)
    pd.testing.assert_frame_equal(images_data, stored_data)


def test_calulate_image_statistics():
    # Create a dummy image
    image = np.zeros((100, 100), dtype=np.uint8)
    filename = "test_image.jpg"

    # Calculate statistics
    stats = calulate_image_statistics(filename, image)

    # Verify the statistics
    assert stats[0] == "test_image"
    assert stats[1] == 0
    assert stats[2] == 0
    assert stats[3] == 0
    assert stats[4] == 0
    assert stats[5] == 0

    # Create a non-zero dummy image
    image = np.ones((100, 100), dtype=np.uint8) * 255
    stats = calulate_image_statistics(filename, image)

    # Verify the statistics
    assert stats[0] == "test_image"
    assert stats[1] == 255
    assert stats[2] == 255
    assert stats[3] == 255
    assert stats[4] == 255
    assert stats[5] == 0


def test_crop_image(tmpdir):
    # Create a temporary directory with a sample image
    image_dir = tmpdir.mkdir("images")
    image_path = os.path.join(image_dir, "test_image.jpg")

    # Create a dummy image
    image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    cv2.imwrite(image_path, image)

    # Test cropping with default margin_percentage
    cropped_image, original_image = crop_image(image_path)
    assert cropped_image.shape == (80, 80, 3)
    assert original_image.shape == (100, 100, 3)

    # Test cropping with a different margin_percentage
    cropped_image, original_image = crop_image(
        image_path, margin_percentage=20
    )
    assert cropped_image.shape == (60, 60, 3)
    assert original_image.shape == (100, 100, 3)

    # Test cropping with a margin_percentage of 0
    cropped_image, original_image = crop_image(image_path, margin_percentage=0)
    assert cropped_image.shape == (100, 100, 3)
    assert original_image.shape == (100, 100, 3)

    # Test cropping with a margin_percentage of 50
    cropped_image, original_image = crop_image(
        image_path, margin_percentage=50
    )
    assert cropped_image.shape == (0, 0, 3)
    assert original_image.shape == (100, 100, 3)
