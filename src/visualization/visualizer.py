import cv2
import src.preprocessing.image_preprocessor as preprocessor
import matplotlib.pyplot as plt

def draw_image_histogram(image):
    """
    draw_image_histogram function takes an image as input and displays the histogram of the image.

    Input:
    image: np.array: Image as a numpy array
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist_gray = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    # Plot the histogram
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.plot(hist_gray)
    plt.xlim([0, 256])
    plt.show()

def draw_image_countours(image_path,display_image = False) :
    """
    draw_image_countours function takes an image path as input and displays the image with contours.
    
    Input:
    image_path: str: Path to the image file
    display_image: bool: If True, the image with contours will be displayed
    """
    image = preprocessor.crop_image(image_path, 10)
    blurred_image = cv2.GaussianBlur(image[0], (5, 5), 0)
    edges = cv2.Canny(blurred_image, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)        
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    
    if display_image:
        cv2.imshow('Contours', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        
def plot_images(images, labels):
    """
    plot_image function takes an image path as input and displays the image.
    
    Input:
    images : list : List of images
    labels : list : List of labels
    """
    # Read the image
    _, axs = plt.subplots(1, len(images), figsize=(7, 4))

    for i in range(len(images)):
        axs[i].imshow(images[i], cmap='gray')
        axs[i].set_title(labels[i])

    # Remove ticks from the subplots
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    # Display the subplots
    plt.tight_layout()
    plt.show()