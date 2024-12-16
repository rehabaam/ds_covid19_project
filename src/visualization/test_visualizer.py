import pytest
import os
import pandas as pd
from src.preprocessing.image_preprocessor import *
import matplotlib.pyplot as plt

@pytest.fixture
def test_plot_images_statistics(tmpdir):
    # Create a sample DataFrame
    data = {
        'image': ['test_image_0', 'test_image_1', 'test_image_2'],
        'min': [0, 0, 0],
        'max': [0, 0, 0],
        'mean': [0, 0, 0],
        'median': [0, 0, 0],
        'std': [0, 0, 0]
    }
    images_data = pd.DataFrame(data)
    
    # Define the CSV filename
    csv_filename = os.path.join(tmpdir, "test_statistics.csv")
    
    # Store the statistics in a CSV file
    store_images_statistics(images_data, csv_filename)
    
    # Plot the statistics
    plt.figure()
    plot_images_statistics("Test Dataset", csv_filename)
    
    # Verify that the plot was created
    assert plt.gcf().number == 2