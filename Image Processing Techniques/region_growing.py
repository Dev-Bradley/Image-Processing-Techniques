import numpy as np

def region_growing(image, seed_point, threshold=5):
    x, y = seed_point
    region_mean = image[x, y]
    region_size = 1
    region_pixels = [(x, y)]
    visited = np.zeros_like(image, dtype=bool)
    visited[x, y] = True
    stack = [(x, y)]
    
    while stack:
        cx, cy = stack.pop()
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < image.shape[0] and 0 <= ny < image.shape[1] and not visited[nx, ny]:
                visited[nx, ny] = True
                if abs(image[nx, ny] - region_mean) < threshold:
                    stack.append((nx, ny))
                    region_pixels.append((nx, ny))
                    region_mean = (region_mean * region_size + image[nx, ny]) / (region_size + 1)
                    region_size += 1
    
    region_image = np.zeros_like(image)
    for px, py in region_pixels:
        region_image[px, py] = 255
    
    return region_image

def region_merging_criteria(region1, region2, threshold):
    mean1 = np.mean(region1)
    mean2 = np.mean(region2)
    return abs(mean1 - mean2) < threshold

def stopping_criteria(region_size, max_size):
    return region_size >= max_size
