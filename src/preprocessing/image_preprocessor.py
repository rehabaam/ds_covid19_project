import cv2
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
        cv2.waitKey(1)
        
def plot_image(original_image, modified_image):
    """
    plot_image function takes an image path as input and displays the image.
    
    Input:
    image_path: str: Path to the image file
    """
    # Read the image
    _, axs = plt.subplots(1, 2, figsize=(7, 4))

    # Plot the original image
    axs[0].imshow(original_image)
    axs[0].set_title('Original Image')

    # Plot the modified image
    axs[1].imshow(modified_image)
    axs[1].set_title('Modified image')

    # Remove ticks from the subplots
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    # Display the subplots
    plt.tight_layout()
    plt.show()

def calulate_image_statistics(filename,image):
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
            data.append(calulate_image_statistics(filename,gray_image))
    return pd.DataFrame(data,columns = ['image','min','max','mean','median','std'])

def get_images_edges_statistics(images_dir):
    """
    get_images_edges_statistics function returns the number of images in the directory.
    
    Input:
    images_dir: str: Path to the directory containing images
    """
    data = []
    ddept=cv2.CV_8U
    for filename in os.listdir(images_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image = cv2.imread(os.path.join(images_dir, filename))
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            x = cv2.Sobel(gray_image, ddept, 1,0, ksize=3, scale=1)
            y = cv2.Sobel(gray_image, ddept, 0,1, ksize=3, scale=1)
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

def plot_images_statistics(dataset,filename):
    """
    plot_images_statistics function plots the image statistics.
    
    Input:
    images_data: pd.DataFrame: DataFrame containing image statistics
    """
    stats = pd.read_csv(filename, index_col=0)
    stats.plot.hist(subplots=True, layout=(3,3), figsize=(15, 15), bins=20, title=dataset)