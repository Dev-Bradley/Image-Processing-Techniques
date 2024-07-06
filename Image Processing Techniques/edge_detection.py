import numpy as np
from scipy.ndimage import gaussian_filter

def sobel_operator(image, low_threshold, high_threshold):
    # Define Sobel kernels
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
    # Convolution with the Sobel kernels
    Gx = convolve2d(image, Kx)
    Gy = convolve2d(image, Ky)
    
    # Calculate gradient magnitude
    G = np.sqrt(Gx**2 + Gy**2)
    G = G / G.max() * 255
    return G.astype(np.uint8)

def sobel_operator_with_angle(image):
    # Define Sobel kernels
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
    # Convolution with the Sobel kernels
    Gx = convolve2d(image, Kx)
    Gy = convolve2d(image, Ky)
    
    # Calculate gradient magnitude and direction
    G = np.sqrt(Gx**2 + Gy**2)
    theta = np.arctan2(Gy, Gx)
    
    return G, theta

def canny_edge_detector(image, low_threshold, high_threshold):
    # Step 1: Noise Reduction (Gaussian Blur)
    image_blur = gaussian_blur(image, kernel_size=5, sigma=1.4)
    
    # Step 2: Gradient Calculation (Sobel Operator)
    gradient, theta = sobel_operator_with_angle(image_blur)
    
    # Step 3: Non-Maximum Suppression
    suppressed = non_maximum_suppression(gradient, theta)
    
    # Step 4: Double Threshold and Edge Tracking by Hysteresis
    strong, weak = 255, 75
    edges = hysteresis(suppressed, low_threshold, high_threshold, strong, weak)
    return edges

def convolve2d(image, kernel):
    m, n = kernel.shape
    y, x = image.shape
    y, x = y - m + 1, x - n + 1
    new_image = np.zeros((y, x))
    for i in range(y):
        for j in range(x):
            new_image[i][j] = np.sum(image[i:i+m, j:j+n] * kernel)
    return new_image

def gaussian_blur(image, kernel_size=5, sigma=1.4):
    return gaussian_filter(image, sigma=sigma)

def non_maximum_suppression(gradient, theta):
    M, N = gradient.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M-1):
        for j in range(1, N-1):
            try:
                q = 255
                r = 255
                # angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = gradient[i, j+1]
                    r = gradient[i, j-1]
                # angle 45
                elif 22.5 <= angle[i,j] < 67.5:
                    q = gradient[i+1, j-1]
                    r = gradient[i-1, j+1]
                # angle 90
                elif 67.5 <= angle[i,j] < 112.5:
                    q = gradient[i+1, j]
                    r = gradient[i-1, j]
                # angle 135
                elif 112.5 <= angle[i,j] < 157.5:
                    q = gradient[i-1, j-1]
                    r = gradient[i+1, j+1]

                if (gradient[i,j] >= q) and (gradient[i,j] >= r):
                    Z[i,j] = gradient[i,j]
                else:
                    Z[i,j] = 0

            except IndexError as e:
                pass
    
    return Z

def hysteresis(image, low_threshold, high_threshold, strong, weak):
    M, N = image.shape
    res = np.zeros((M, N), dtype=np.int32)
    
    strong_i, strong_j = np.where(image >= high_threshold)
    weak_i, weak_j = np.where((image <= high_threshold) & (image >= low_threshold))
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (res[i,j] == weak):
                try:
                    if ((res[i+1, j-1] == strong) or (res[i+1, j] == strong) or (res[i+1, j+1] == strong)
                        or (res[i, j-1] == strong) or (res[i, j+1] == strong)
                        or (res[i-1, j-1] == strong) or (res[i-1, j] == strong) or (res[i-1, j+1] == strong)):
                        res[i, j] = strong
                    else:
                        res[i, j] = 0
                except IndexError as e:
                    pass
    
    return res
