import numpy as np
from skimage import img_as_float, exposure, color, filters

def simple_thresholding(image, threshold):
    # Perform simple thresholding
    segmented_image = image > threshold
    return segmented_image

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
