import matplotlib.pyplot as plt
import numpy as np
from skimage import img_as_float, exposure

def plot_img_and_hist(image):
    # Plot image and its histogram side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ax_img, ax_hist = axes

    ax_img.imshow(image, cmap='gray')
    ax_img.axis('off')
    ax_img.set_title('Image')

    ax_hist.hist(image.ravel(), bins=256, histtype='step', color='black')
    ax_hist.set_title('Histogram')
    ax_hist.set_xlabel('Pixel value')
    ax_hist.set_ylabel('Frequency')

    return fig

def histogram_equalization(image):
    # Calculate histogram
    hist, bins = np.histogram(image.flatten(), 256, [0,256])
    # Calculate cumulative distribution function
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    # Perform histogram equalization
    equalized_image = np.interp(image.flatten(), bins[:-1], cdf_normalized)
    equalized_image = equalized_image.reshape(image.shape)
    return equalized_image