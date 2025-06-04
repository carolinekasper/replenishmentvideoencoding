import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the two frames
frame1 = cv2.imread("frame1.jpg")
frame2 = cv2.imread("frame2.jpg")

# Resize if needed
frame1 = cv2.resize(frame1, (640, 480))
frame2 = cv2.resize(frame2, (640, 480))

# Convert to grayscale
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

# Parameters
block_size = 16
threshold = 20  # change sensitivity

# Prepare output
reconstructed = np.zeros_like(gray1)
mask = np.zeros_like(gray1)

# Compare block-wise
for i in range(0, gray1.shape[0], block_size):
    for j in range(0, gray1.shape[1], block_size):
        block1 = gray1[i:i+block_size, j:j+block_size]
        block2 = gray2[i:i+block_size, j:j+block_size]

        diff = np.mean(np.abs(block1.astype(int) - block2.astype(int)))

        if diff > threshold:
            reconstructed[i:i+block_size, j:j+block_size] = block2
            mask[i:i+block_size, j:j+block_size] = 255  # white block = updated
        else:
            reconstructed[i:i+block_size, j:j+block_size] = block1

# Display results
plt.figure(figsize=(12,4))
plt.subplot(1,3,1), plt.title("Original Frame"), plt.imshow(gray1, cmap='gray')
plt.subplot(1,3,2), plt.title("Mask (Changed Blocks)"), plt.imshow(mask, cmap='gray')
plt.subplot(1,3,3), plt.title("Reconstructed Frame"), plt.imshow(reconstructed, cmap='gray')
plt.tight_layout()
plt.show()