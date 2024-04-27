import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Read the input image
input_image = cv2.imread('images.jpeg', cv2.IMREAD_GRAYSCALE)

# Step 2: Define the convolution kernel/filter
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])

# Step 3: Perform convolution operation
convolution_output = cv2.filter2D(input_image, -1, kernel)

# Step 4: Perform pooling operation
pooling_output = cv2.resize(input_image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

# Step 5: Flatten the pooling output
flattened_output = pooling_output.flatten()

# Step 6: Reshape the flattened output to original dimensions
reshaped_output = flattened_output.reshape(pooling_output.shape)

# Step 7: Display input, output images, and flattened output
plt.figure(figsize=(12, 4))

plt.subplot(1, 4, 1)
plt.imshow(input_image, cmap='gray')
plt.title('Input Image')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(convolution_output, cmap='gray')
plt.title('Convolution Output')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(pooling_output, cmap='gray')
plt.title('Pooling Output')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(reshaped_output, cmap='gray')
plt.title('Flattened Output')
plt.axis('off')

plt.show()
