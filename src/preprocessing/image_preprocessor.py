import cv2
import os
import pandas as pd
import numpy as np

def draw_image_countours(image_path,display_image = False) :
    """
    draw_image_countours function takes an image path as input and displays the image with contours.
    
    Input:
    image_path: str: Path to the image file
    display_image: bool: If True, the image with contours will be displayed
    """
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edges = cv2.Canny(blurred_image, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)        
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    
    if display_image:
        cv2.imshow('Contours', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def get_images_statistics(images_dir):
    """
    get_images_statustics function returns the number of images in the directory.
    
    Input:
    images_dir: str: Path to the directory containing images
    """
    data = []
    for filename in os.listdir(images_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image = cv2.imread(os.path.join(images_dir, filename))
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_statistics = [filename[:-4],np.min(gray_image),np.max(gray_image),np.mean(gray_image),np.median(gray_image),np.std(gray_image)]
            data.append(image_statistics)
    return pd.DataFrame(data,columns = ['image','min','max','mean','median','std'])


def store_images_statistics(images_data,csv_filename):
    """
    store_images_statistics function stores the image statistics in a CSV file.
    
    Input:
    images_data: pd.DataFrame: DataFrame containing image statistics
    csv_filename: str: Name of the CSV file
    """
    images_data.to_csv(csv_filename)
            