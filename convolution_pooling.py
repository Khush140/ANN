# 7. Perform convolution and pooling operation on an input image and display both input, output images.
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Upload your input image to Google Colab
# Make sure to upload an image file named 'input.jpeg'

# Step 2: Read the input image
input_image = cv2.imread('images.jpeg', cv2.IMREAD_GRAYSCALE)

# Step 3: Define the convolution kernel/filter
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])

# Step 4: Perform convolution operation
convolution_output = cv2.filter2D(input_image, -1, kernel)

# Step 5: Perform pooling operation
pooling_output = cv2.resize(input_image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

# Step 6: Display input and output images in a single row with equal sizes and axis labels
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(input_image, cmap='gray')
plt.title('Input Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(convolution_output, cmap='gray')
plt.title('Convolution Output')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(pooling_output, cmap='gray')
plt.title('Pooling Output')
plt.axis('off')

plt.show()
