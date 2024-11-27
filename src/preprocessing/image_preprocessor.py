import cv2
import os

class Image_PreProcessor:
    """
    Image_PreProcessor class processes images in a directory by drawing contours around objects in the images.
    """
    def __init__(self, images_dir, display_image):
        self.images_dir = images_dir
        self.display_image = display_image

    def draw_image_countours(self,image_path,display_image = False) :
        """
        draw_image_countours function takes an image path as input and displays the image with contours.
        
        Input:
        image_path: str: Path to the image file
        display_image: bool: If True, the image with contours will be displayed
        """
        print(type(image_path))
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


    def process_all_images_in_directory(self):
        """
        process_all_images_in_directory function processes all images in the directory by calling draw_image_countours function.
        """
        for filename in os.listdir(self.images_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_path = os.path.join(self.images_dir, filename)
                print(image_path)
                self.draw_image_countours(image_path)