import streamlit as st
import numpy as np
from PIL import Image

def gaussian_blur(image, kernel_size=5, sigma=1):
    kernel = create_gaussian_kernel(kernel_size, sigma)
    return convolve2d(image, kernel)

def create_gaussian_kernel(size, sigma):
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    return kernel / np.sum(kernel)

def median_blur(image, kernel_size=3):
    pad_size = kernel_size // 2
    padded_image = np.pad(image, pad_size, mode='reflect')
    output = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            kernel = padded_image[i:i+kernel_size, j:j+kernel_size]
            output[i, j] = np.median(kernel)
    return output / 255.0  # Normalize image data to [0.0, 1.0]

def average_blur(image, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    return convolve2d(image, kernel) / 255.0  # Normalize image data to [0.0, 1.0]

def bilateral_filter(image, d=5, sigma_color=75, sigma_space=75):
    # Simplified bilateral filter
    def gaussian(x, sigma):
        return np.exp(-(x ** 2) / (2 * (sigma ** 2)))

    output = np.zeros_like(image)
    padded_image = np.pad(image, d // 2, mode='reflect')
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            wp = 0
            filtered_pixel = 0
            for k in range(-d // 2, d // 2 + 1):
                for l in range(-d // 2, d // 2 + 1):
                    gi = gaussian(np.linalg.norm(padded_image[i + d // 2 + k, j + d // 2 + l] - padded_image[i + d // 2, j + d // 2]), sigma_color)
                    gs = gaussian(np.linalg.norm(np.array([k, l])), sigma_space)
                    weight = gi * gs
                    filtered_pixel += weight * padded_image[i + d // 2 + k, j + d // 2 + l]
                    wp += weight
            output[i, j] = filtered_pixel / wp
    return output / 255.0  # Normalize image data to [0.0, 1.0]

def convolve2d(image, kernel):
    m, n = kernel.shape
    y, x = image.shape
    y = y - m + 1
    x = x - n + 1
    new_image = np.zeros((y, x))
    for i in range(y):
        for j in range(x):
            new_image[i][j] = np.sum(image[i:i+m, j:j+n] * kernel)
    return new_image / 255.0  # Normalize image data to [0.0, 1.0]

def main():
    st.title("Image Processing Operations")

    # Load and display image
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        image = np.array(Image.open(uploaded_file).convert('L')) / 255.0  # Normalize image data to [0.0, 1.0]
        st.image(image, caption='Uploaded Image', use_column_width=True)

        st.subheader("Image Processing Operations")

        if st.button("Apply Gaussian Blur"):
            blurred_image = gaussian_blur(image)
            st.image(blurred_image, caption='Gaussian Blur', use_column_width=True)

        if st.button("Apply Median Blur"):
            blurred_image = median_blur(image)
            st.image(blurred_image, caption='Median Blur', use_column_width=True)

        if st.button("Apply Average Blur"):
            blurred_image = average_blur(image)
            st.image(blurred_image, caption='Average Blur', use_column_width=True)

        if st.button("Apply Bilateral Filter"):
            blurred_image = bilateral_filter(image)
            st.image(blurred_image, caption='Bilateral Filter', use_column_width=True)

if __name__ == "__main__":
    main()
