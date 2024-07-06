import numpy as np
import matplotlib.pyplot as plt

def active_contour_segmentation(rgb_image):
    def rgb_to_gray(rgb_image):
        # Convert RGB to grayscale using the luminance method
        gray_image = np.dot(rgb_image[..., :3], [0.2989, 0.5870, 0.1140])
        return gray_image

    def gaussian_filter(image, sigma):
        # Apply a Gaussian filter to the image
        def gaussian_kernel(size, sigma):
            kernel = np.fromfunction(
                lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x - (size//2))**2 + (y - (size//2))**2) / (2*sigma**2)),
                (size, size)
            )
            return kernel / np.sum(kernel)

        kernel_size = int(6*sigma + 1)
        kernel = gaussian_kernel(kernel_size, sigma)
        
        padded_image = np.pad(image, pad_width=((kernel_size//2, kernel_size//2), (kernel_size//2, kernel_size//2)), mode='reflect')
        filtered_image = np.zeros_like(image)
        
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                filtered_image[i, j] = np.sum(padded_image[i:i+kernel_size, j:j+kernel_size] * kernel)
        
        return filtered_image

    def active_contour(image, snake, alpha, beta, gamma, max_iterations=2500, convergence=0.1):
        # Define finite difference operators
        def finite_diff(arr):
            forward_diff = np.roll(arr, -1, axis=0) - arr
            backward_diff = arr - np.roll(arr, 1, axis=0)
            return forward_diff, backward_diff

        for i in range(max_iterations):
            fx, bx = finite_diff(snake[:, 1])
            fy, by = finite_diff(snake[:, 0])

            # Gradient magnitudes
            grad_x = np.gradient(image, axis=1)
            grad_y = np.gradient(image, axis=0)
            
            force_x = grad_x[snake[:, 0].astype(int), snake[:, 1].astype(int)]
            force_y = grad_y[snake[:, 0].astype(int), snake[:, 1].astype(int)]

            # Update snake positions
            snake[:, 1] += gamma * (alpha * (bx + fx) - beta * force_x)
            snake[:, 0] += gamma * (alpha * (by + fy) - beta * force_y)

            # Check for convergence
            if i % 100 == 0:
                if np.linalg.norm(gamma * (alpha * (bx + fx) - beta * force_x)) < convergence and \
                   np.linalg.norm(gamma * (alpha * (by + fy) - beta * force_y)) < convergence:
                    break

        return snake

    # Convert RGB image to grayscale
    gray_img = rgb_to_gray(rgb_image)

    # Apply Gaussian filter
    smoothed_image = gaussian_filter(gray_img, sigma=3)

    # Initialize a circular contour around the image
    s = np.linspace(0, 2 * np.pi, 400)
    r = 100 + 100 * np.sin(s)
    c = 220 + 100 * np.cos(s)
    init = np.array([r, c]).T

    # Apply active contour model
    snake = active_contour(smoothed_image, init, alpha=0.015, beta=10, gamma=0.001)

    return gray_img, snake
