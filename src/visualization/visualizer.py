# -*- coding: utf-8 -*-
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model

import src.preprocessing.image_preprocessor as preprocessor


def draw_image_histogram(image):
    """
    draw_image_histogram function takes an image as input
    and displays the histogram of the image.

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


def draw_image_countours(image_path, display_image=False):
    """
    draw_image_countours function takes an image path as input
    and displays the image with contours.

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
        cv2.imshow("Contours", image)
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
        axs[i].imshow(images[i], cmap="gray")
        axs[i].set_title(labels[i])

    # Remove ticks from the subplots
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    # Display the subplots
    plt.tight_layout()
    plt.show()


def grad_cam(image, model, layer_name) -> tuple:
    """
    "grad_cam function takes an image, a model, and a layer name as input
    and returns the Grad-CAM heatmap and the predicted class.

    Input:
    image: np.array: Image as a numpy array
    model: tf.keras.Model: Trained Keras model
    layer_name: str: Name of the convolutional layer to visualize

    Output:
    heatmap: np.array: Grad-CAM heatmap
    predicted_class: int: Predicted class index
    """
    # Retrieve the convolutional layer
    layer = model.get_layer(layer_name)

    # Create a model that generates the outputs of the convolutional layer and the predictions
    grad_model = Model(inputs=model.input, outputs=[layer.output, model.output])

    # Add a batch dimension
    image = tf.expand_dims(image, axis=0)

    # Compute the gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        predicted_class = tf.argmax(predictions[0])  # Predicted class
        loss = predictions[:, predicted_class]  # Loss for the predicted class

    # Gradients of the scores with respect to the outputs of the convolutional layer
    grads = tape.gradient(loss, conv_outputs)

    # Weighted average of the gradients for each channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the activations by the calculated gradients
    conv_outputs = conv_outputs[0]  # Remove the batch dimension
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0)  # Focus only on positive values
    heatmap /= tf.math.reduce_max(heatmap)  # Normalize between 0 and 1
    heatmap = heatmap.numpy()  # Convert to numpy array for visualization

    # Resize the heatmap to match the original image size
    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis], (image.shape[1], image.shape[2])
    ).numpy()
    heatmap_resized = np.squeeze(
        heatmap_resized, axis=-1
    )  # Remove the singleton dimension at the end of the heatmap_resized array

    # Color the heatmap with a palette (e.g., "jet")
    heatmap_colored = plt.cm.jet(heatmap_resized)[..., :3]  # Get the R, G, B channels
    superimposed_image = heatmap_colored * 0.7 + image[0].numpy() / 255.0

    return np.clip(superimposed_image, 0, 1), predicted_class


def show_grad_cam_cnn(images, model, class_names):
    """
    show_grad_cam_cnn function takes a list of images, a model, and class names as input
    and displays the Grad-CAM heatmaps for each image.

    Input:
    images: list: List of images
    model: tf.keras.Model: Trained Keras model
    class_names: list: List of class names

    """
    number_of_images = images.shape[0]
    conv_layers = [layer.name for layer in model.layers if isinstance(layer, Conv2D)]

    plt.figure(figsize=(16, 16))

    for j, layer in enumerate(conv_layers):

        for i in range(number_of_images):

            subplot_index = i + 1 + j * number_of_images
            plt.subplot(len(conv_layers), number_of_images, subplot_index)

            # Get the image with the overlaid heatmap
            grad_cam_image, predicted_class = grad_cam(images[i], model, layer)

            # Display the image with Grad-CAM
            plt.title(f"Grad-CAM {layer} {class_names[predicted_class]}")
            plt.imshow(grad_cam_image)
            plt.axis("off")

    plt.show()
