# image_details.py

import numpy as np

def get_image_details(image):
    """
    Get details of the input image.

    Parameters:
        image (numpy.ndarray): Input image as a numpy array.

    Returns:
        dict: Dictionary containing image details.
    """
    # Get image shape
    height, width = image.shape
    channels = 1 if len(image.shape) == 2 else image.shape[2]

    # Get image datatype
    datatype = image.dtype

    # Calculate image size
    size = image.nbytes / (1024 * 1024)  # Convert bytes to MB

    return {
        "Height": height,
        "Width": width,
        "Channels": channels,
        "Datatype": datatype,
        "Size (MB)": size
    }
