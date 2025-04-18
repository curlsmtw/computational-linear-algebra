import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# loads image
img = Image.open('roses.jpg')

# grayscale
gray_img = img.convert('L')

# resizes the image to 50x50 pixels
gray_img_resized = gray_img.resize((50, 50))

# converts to a numpy array
img_array = np.array(gray_img_resized)

# displays the original grayscale image
plt.imshow(img_array, cmap='gray')
plt.title("Grayscale Image")
plt.show()

# performs Singular Value Decomposition
U, S, VT = np.linalg.svd(img_array, full_matrices=False)

# plots the screen plot of the singular values
plt.plot(S)
plt.title('Plot of Singular Values')
plt.xlabel('Index')
plt.ylabel('Singular Value Magnitude')
plt.show()

def reconstruct_image(U, S, VT, k):
    # reconstructs using the first k singular values
    S_k = np.diag(S[:k])
    U_k = U[:, :k]
    VT_k = VT[:k, :]
    return np.dot(U_k, np.dot(S_k, VT_k))

# different values of k
k_values = [1, 5, 10, 20, 50]
for k in k_values:
    reconstructed_img = reconstruct_image(U, S, VT, k)
    plt.imshow(reconstructed_img, cmap='gray')
    plt.title(f'Reconstructed Image with k={k}')
    plt.show()

# dimensions of the OG image
m, n = img_array.shape

# compression ratio for each rank k
def calculate_compression_ratio(k, m, n):
    return (k * (m + n + 1)) / (m * n)

# displays compression ratios for different k values
compression_ratios = [calculate_compression_ratio(k, m, n) for k in k_values]
for k, ratio in zip(k_values, compression_ratios):
    print(f"Compression Ratio for k={k}: {ratio:.2f}")