# -*- coding: utf-8 -*-
import numpy as np
import pytest

from .build_features import get_features


@pytest.fixture
def dummy_image():
    # Create a dummy image
    image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    return image


def test_get_features_good(dummy_image):
    original_image, image_with_features = get_features(dummy_image, method="Good")
    assert original_image.shape == (100, 100, 3)
    assert image_with_features.shape == (100, 100, 3)
    assert not np.array_equal(original_image, image_with_features)


def test_get_features_fast(dummy_image):
    original_image, image_with_features = get_features(dummy_image, method="Fast")
    assert original_image.shape == (100, 100, 3)
    assert image_with_features.shape == (100, 100, 3)
    assert not np.array_equal(original_image, image_with_features)


def test_get_features_orb(dummy_image):
    original_image, image_with_features = get_features(dummy_image, method="ORB")
    assert original_image.shape == (100, 100, 3)
    assert image_with_features.shape == (100, 100, 3)
    assert not np.array_equal(original_image, image_with_features)


def test_get_features_sift(dummy_image):
    original_image, image_with_features = get_features(dummy_image, method="SIFT")
    assert original_image.shape == (100, 100, 3)
    assert image_with_features.shape == (100, 100, 3)
    assert not np.array_equal(original_image, image_with_features)


def test_get_features_empty_method(dummy_image):
    with pytest.raises(TypeError, match="Empty method"):
        get_features(dummy_image, method="")


def test_get_features_unsupported_method(dummy_image):
    with pytest.raises(TypeError, match="unsupported"):
        get_features(dummy_image, method="Unsupported")
