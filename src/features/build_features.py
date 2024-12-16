import cv2
import os
import pandas as pd
from skimage.feature import hog

def get_descriptor(descriptor):
    """
    get_descriptor function returns the descriptor method.

    Input:
    descriptor: str: Name of the descriptor

    Output:
    descriptor_method: cv2.Feature2D: Descriptor method
    """
    match descriptor:
        case 'ORB':
           descriptor_method = cv2.ORB_create() 
        case 'SIFT':
            descriptor_method = cv2.SIFT_create()
        case 'BRIEF':
            descriptor_method = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        case '':
            TypeError('Descriptor not found')
    return descriptor_method

def get_image_features(images_dir, method = 'ORB'):
    """
    get_image_SIFT_features function plots the image statistics.
    
    Input:
    images_dir: str: Path to the directory containing images
    method: str: Name of the descriptor to be used

    Output:
    keyPoints: list: List of keypoints
    descriptors: list: List of descriptors
    data: pd.DataFrame: Dataframe containing image names and number of descriptors
    """
    data = []
    keyPoints, descriptors = None, None

    descriptor_method = get_descriptor(method)

    for filename in os.listdir(images_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image = cv2.imread(os.path.join(images_dir, filename), cv2.COLOR_BGR2GRAY)
            keyPoints, descriptors = descriptor_method.detectAndCompute(image,None)
            get_descriptors = lambda x: x.shape[0] if x is not None else 0
            data.append([filename[:-4],get_descriptors(descriptors)]) 
    return keyPoints, descriptors, pd.DataFrame(data,columns = ['image','descriptors'])

def get_hog_features(image):
    """
    get_hog_features function returns the HOG feature descriptor.

    Output:
    hog_features: np.array: HOG feature descriptor
    """
    return hog(image, orientations=9, pixels_per_cell=(4, 4),
                                cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)