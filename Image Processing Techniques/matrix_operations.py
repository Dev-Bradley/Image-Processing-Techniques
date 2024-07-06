import numpy as np

def convolve2d(image, kernel):
    m, n = kernel.shape
    y, x = image.shape
    y = y - m + 1
    x = x - n + 1
    new_image = np.zeros((y, x))
    for i in range(y):
        for j in range(x):
            new_image[i][j] = np.sum(image[i:i+m, j:j+n] * kernel)
    return new_image

# Define other matrix operations
def convolution(image, kernel):
    return convolve2d(image, kernel)

def correlation(image, kernel):
    kernel = np.flipud(np.fliplr(kernel))
    return convolve2d(image, kernel)

def fourier_transform(image):
    return np.fft.fft2(image)

def inverse_fourier_transform(image):
    return np.fft.ifft2(image)

def matrix_transformations(image, transformation_matrix):
    output = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            x, y = np.dot(transformation_matrix, [i, j, 1])[:2]
            if 0 <= x < image.shape[0] and 0 <= y < image.shape[1]:
                output[int(x), int(y)] = image[i, j]
    return output

# Normalization function to scale image data to [0.0, 1.0]
def normalize_image(image):
    image_min = np.min(image)
    image_max = np.max(image)
    return (image - image_min) / (image_max - image_min)