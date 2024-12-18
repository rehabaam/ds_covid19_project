import cv2
import os
import pandas as pd
import numpy as np

def crop_image(image_path, margin_percentage=10):
    """
    crop_image function takes an image path as input and returns the cropped image.

    Input:
    image_path: str: Path to the image file
    margin_percentage: int: Percentage of the image to be cropped from all sides

    Output:
    cropped_image: np.array: Cropped image as a numpy array
    image: np.array: Original image as a numpy array
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    margin_x = int(width * margin_percentage / 100)
    margin_y = int(height * margin_percentage / 100)

    cropped_image = gray[margin_y:height-margin_y, margin_x:width-margin_x]

    return cropped_image, gray

def calulate_image_statistics(filename, image):
    """
    calulate_image_statistics function calculates the statistics of an image.
    
    Input:
    image: np.array: Image as a numpy array
    filename: str: Name of the image file
    """
    return [filename[:-4],
            np.min(image),
            np.max(image),
            np.mean(image),
            np.median(image),
            np.std(image)]

def get_images_statistics(images_dir, margin_percentage=0):
    """
    get_images_statustics function returns the number of images in the directory.
    
    Input:
    images_dir: str: Path to the directory containing images
    margin_percentage: int: Percentage of the image to be cropped from all sides
    """
    data = []
    for filename in os.listdir(images_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image = crop_image(os.path.join(images_dir, filename), margin_percentage)
            data.append(calulate_image_statistics(filename, image[0]))
    return pd.DataFrame(data,columns = ['image','min','max','mean','median','std'])

def get_images_edges_statistics(images_dir, margin_percentage=0):
    """
    get_images_edges_statistics function returns the number of images in the directory.
    
    Input:
    images_dir: str: Path to the directory containing images
    margin_percentage: int: Percentage of the image to be cropped from all sides
    """
    data = []
    ddept=cv2.CV_8U
    for filename in os.listdir(images_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image = crop_image(os.path.join(images_dir, filename), margin_percentage)
            x = cv2.Sobel(image[0], ddept, 1,0, ksize=3, scale=1)
            y = cv2.Sobel(image[0], ddept, 0,1, ksize=3, scale=1)
            edge = cv2.addWeighted(cv2.convertScaleAbs(x), 0.5, cv2.convertScaleAbs(y), 0.5,0)
            data.append(calulate_image_statistics(filename,edge))
    return pd.DataFrame(data,columns = ['image','min','max','mean','median','std'])

def store_images_statistics(images_data,csv_filename):
    """
    store_images_statistics function stores the image statistics in a CSV file.
    
    Input:
    images_data: pd.DataFrame: DataFrame containing image statistics
    csv_filename: str: Name of the CSV file
    """
    images_data.to_csv(csv_filename)

def plot_images_statistics(dataset,stats):
    """
    plot_images_statistics function plots the image statistics.
    
    Input:
    dataset: str: Name of the dataset
    stats: pd.DataFrame: DataFrame containing image statistics
    """
    stats.plot.hist(subplots=True, layout=(3,3), figsize=(15, 15), bins=20, title=dataset)