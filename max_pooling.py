from google.colab.patches import cv2_imshow
import cv2
import numpy as np

def max_pooling(image, pool_size=(2, 2)):
    # Get image dimensions
    height, width = image.shape[:2]

    # Calculate output dimensions after pooling
    new_height = height // pool_size[0]
    new_width = width // pool_size[1]

    # Initialize an empty array for the pooled image
    pooled_image = np.zeros((new_height, new_width))

    # Perform max pooling
    for i in range(0, height, pool_size[0]):
        for j in range(0, width, pool_size[1]):
            # Extract the region of interest (ROI) from the input image
            roi = image[i:i+pool_size[0], j:j+pool_size[1]]

            # Take the maximum value from the ROI and assign it to the corresponding pixel in the pooled image
            pooled_image[i//pool_size[0], j//pool_size[1]] = np.max(roi)

    return pooled_image.astype(np.uint8)

# Load an input image
input_image = cv2.imread("images.jpeg", cv2.IMREAD_GRAYSCALE)

# Perform max pooling with a pool size of (2, 2)
output_image = max_pooling(input_image)

# Display input and output images
cv2_imshow(input_image)
cv2_imshow(output_image)

# Save the output image
cv2.imwrite("output_image.jpg", output_image)
