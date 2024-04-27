import numpy as np
from google.colab.patches import cv2_imshow

# Define the pooling function
def average_pooling(image, pool_size):
    # Get the dimensions of the input image
    height, width = image.shape

    # Calculate the dimensions of the output feature map
    output_height = height // pool_size
    output_width = width // pool_size

    # Initialize the output feature map
    output_image = np.zeros((output_height, output_width), dtype=np.uint8)

    # Perform average pooling
    for i in range(0, output_height):
        for j in range(0, output_width):
            # Calculate the pooling region
            start_i = i * pool_size
            start_j = j * pool_size
            end_i = start_i + pool_size
            end_j = start_j + pool_size

            # Extract the region from the input image
            region = image[start_i:end_i, start_j:end_j]

            # Compute the average value of the region
            average_value = np.mean(region)

            # Assign the average value to the corresponding location in the output feature map
            output_image[i, j] = average_value

    return output_image

# Load the input image
input_image = cv2.imread('images.jpeg', cv2.IMREAD_GRAYSCALE)

# Define the size of the pooling filter (e.g., 2x2)
pool_size = 2

# Perform average pooling on the input image
output_image = average_pooling(input_image, pool_size)


# Display the input and output images
cv2_imshow(input_image)
cv2_imshow(output_image)
