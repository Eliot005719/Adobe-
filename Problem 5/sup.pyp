import numpy as np
from matplotlib import pyplot as plt

# Load an image using a library like Matplotlib
# This will convert the image into a NumPy array
image = plt.imread('input_image.jpg')

# Define the coordinates of the region you want to crop
x1, y1, x2, y2 = 100, 150, 300, 400

# Crop the image using array slicing
cropped_image = image[y1:y2, x1:x2, :]

# Save the cropped image using a library like Matplotlib
plt.imsave('cropped_image.jpg', cropped_image)

# Display the cropped image using Matplotlib (optional)
plt.imshow(cropped_image)
plt.show()
