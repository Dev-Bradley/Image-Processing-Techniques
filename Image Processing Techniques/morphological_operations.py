import streamlit as st
import numpy as np
from PIL import Image

def erosion(image, kernel):
    kernel_size = kernel.shape[0]
    pad_size = kernel_size // 2
    padded_image = np.pad(image, pad_size, mode='constant', constant_values=0)
    output = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i+kernel_size, j:j+kernel_size]
            output[i, j] = np.min(region[kernel == 1])
    return output

def dilation(image, kernel):
    kernel_size = kernel.shape[0]
    pad_size = kernel_size // 2
    padded_image = np.pad(image, pad_size, mode='constant', constant_values=0)
    output = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i+kernel_size, j:j+kernel_size]
            output[i, j] = np.max(region[kernel == 1])
    return output

def opening(image, kernel):
    return dilation(erosion(image, kernel), kernel)

def closing(image, kernel):
    return erosion(dilation(image, kernel), kernel)

def morphological_gradient(image, kernel):
    dilated = dilation(image, kernel)
    eroded = erosion(image, kernel)
    return dilated - eroded

def main():
    st.title("Morphological Operations")

    # Load and display image
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        image = np.array(Image.open(uploaded_file).convert('L')) / 255.0  # Normalize image data to [0.0, 1.0]
        st.image(image, caption='Uploaded Image', use_column_width=True)

        st.subheader("Morphological Operations")

        kernel_size = st.slider("Kernel Size", min_value=3, max_value=15, step=2, value=3)
        kernel = np.ones((kernel_size, kernel_size), dtype=int)

        if st.button("Apply Erosion"):
            eroded_image = erosion(image, kernel)  # Pass kernel as argument
            st.image(eroded_image, caption='Erosion', use_column_width=True)

        if st.button("Apply Dilation"):
            dilated_image = dilation(image, kernel)  # Pass kernel as argument
            st.image(dilated_image, caption='Dilation', use_column_width=True)

        if st.button("Apply Opening"):
            opened_image = opening(image, kernel)  # Pass kernel as argument
            st.image(opened_image, caption='Opening', use_column_width=True)

        if st.button("Apply Closing"):
            closed_image = closing(image, kernel)  # Pass kernel as argument
            st.image(closed_image, caption='Closing', use_column_width=True)

        if st.button("Apply Morphological Gradient"):
            gradient_image = morphological_gradient(image, kernel)  # Pass kernel as argument
            st.image(gradient_image, caption='Morphological Gradient', use_column_width=True)

if __name__ == "__main__":
    main()
