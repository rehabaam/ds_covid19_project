import pytest
import os
import cv2
import numpy as np
from image_preprocessor import Image_PreProcessor

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

def test_process_all_images_in_directory(setup_test_images):
    image_dir = setup_test_images
    processor = Image_PreProcessor(images_dir=str(image_dir), display_image=False)
    
    # Mock the draw_image_countours method to avoid displaying images
    processor.draw_image_countours = lambda x: None
    processor.process_all_images_in_directory()
    
    # Verify that all images in the directory were processed
    processed_images = [f"test_image_{i}.jpg" for i in range(3)]
    for image_name in processed_images:
        image_path = os.path.join(image_dir, image_name)
        assert os.path.exists(image_path), f"{image_path} does not exist"