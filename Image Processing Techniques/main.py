import streamlit as st
import numpy as np
from PIL import Image
from edge_detection import sobel_operator, canny_edge_detector
from blurring import gaussian_blur, median_blur, average_blur, bilateral_filter
from histogram import plot_img_and_hist,histogram_equalization
from morphological_operations import erosion, dilation, opening, closing, morphological_gradient
from active_contour import active_contour_segmentation
from region_growing import region_growing
from matrix_operations import convolution, correlation, fourier_transform, inverse_fourier_transform, matrix_transformations
from segmentation import simple_thresholding
from image_details import get_image_details
import matplotlib.pyplot as plt

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background-image: url("https://img.freepik.com/free-photo/digital-art-beautiful-mountains_23-2151123688.jpg");
        background-size: cover;
        background-blend-mode: darken;
        background-color: rgba(0, 0, 0, 0.6); /* Darkens the background image by blending with black */
    }
    </style>
    """,
    unsafe_allow_html=True
)

def normalize_image(image):
                    image_min = np.min(image)
                    image_max = np.max(image)
                    return (image - image_min) / (image_max - image_min)
def main():
    st.title("Görüntü İşleme")

    # Create sidebar navigation menu
    selected_operation = st.sidebar.selectbox("İşlemi seçin", 
                                              ("Edge Detection", "Blurring", "Histogram", 
                                               "Morphological Operations", "Active Contour", 
                                               "Region Growing", "Matrix Operations", "Segmentation", "Görüntü Ayrıntıları"))

    # Load and display image
    uploaded_file = st.file_uploader("Choose an image...", type="jpg", key="file_uploader_1")
    if uploaded_file is not None:
        
        
        
        image_rgb = np.array(Image.open(uploaded_file).convert('RGB'))
        st.image(image_rgb, caption='Uploaded Image', use_column_width=True)

        image_gray = np.dot(image_rgb[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

        if selected_operation == "Edge Detection":
            st.subheader("Edge Detection")
            st.sidebar.write("Select edge detection method:")
            edge_method = st.sidebar.radio("", ("Sobel Operator", "Canny Edge Detector"))

            if edge_method == "Sobel Operator":
                low_threshold = st.sidebar.slider("Low Threshold", 0, 255, 50)
                high_threshold = st.sidebar.slider("High Threshold", 0, 255, 150)
                sobel_image = sobel_operator(image_gray,low_threshold, high_threshold)
                st.image(sobel_image, caption='Sobel Operator', use_column_width=True)
            elif edge_method == "Canny Edge Detector":
                low_threshold = st.sidebar.slider("Low Threshold", 0, 255, 50)
                high_threshold = st.sidebar.slider("High Threshold", 0, 255, 150)
                canny_image = canny_edge_detector(image_gray, low_threshold, high_threshold)
                st.image(canny_image, caption='Canny Edge Detector', use_column_width=True)

        elif selected_operation == "Blurring":
            st.subheader("Blurring")
            st.sidebar.write("Select blurring method:")
            blur_method = st.sidebar.radio("", ("Gaussian Blur", "Median Blur", "Average Blur", "Bilateral Filter"))

            if blur_method == "Gaussian Blur":
                blurred_image = gaussian_blur(image_gray)
                st.image(blurred_image, caption='Gaussian Blur', use_column_width=True)
            elif blur_method == "Median Blur":
                blurred_image = median_blur(image_gray)
                st.image(blurred_image, caption='Median Blur', use_column_width=True)
            elif blur_method == "Average Blur":
                blurred_image = average_blur(image_gray)
                st.image(blurred_image, caption='Average Blur', use_column_width=True)
            elif blur_method == "Bilateral Filter":
                blurred_image = bilateral_filter(image_gray)
                st.image(blurred_image, caption='Bilateral Filter', use_column_width=True)
        
        elif selected_operation == "Histogram":
            st.subheader("Histogram")
            st.sidebar.write("Select histogram operation:")
            hist_method = st.sidebar.radio("", ("Calculate Histogram", "Histogram Equalization"))

            if hist_method == "Calculate Histogram":
                hist = plot_img_and_hist(image_gray)
                st.pyplot(hist)

            elif hist_method == "Histogram Equalization":
                equalized_image = histogram_equalization(image_gray)
                slider_value = st.sidebar.slider("Adjust slider", 0, 100, 50)
                hist_eq = plot_img_and_hist(equalized_image)
                st.pyplot(hist_eq)

                # Add a slider
                
                
        elif selected_operation == "Morphological Operations":
            st.subheader("Morphological Operations")
            st.sidebar.write("Select morphological operation:")
            morph_op = st.sidebar.radio("", ("Erosion", "Dilation", "Opening", "Closing", "Morphological Gradient"))

            if morph_op == "Erosion":
                kernel_size = st.sidebar.slider("Kernel Size", min_value=3, max_value=15, step=2, value=3)
                kernel = np.ones((kernel_size, kernel_size), dtype=int)
                eroded_image = erosion(image_gray, kernel)
                st.image(eroded_image, caption='Erosion', use_column_width=True)

            elif morph_op == "Dilation":
                kernel_size = st.sidebar.slider("Kernel Size", min_value=3, max_value=15, step=2, value=3)
                kernel = np.ones((kernel_size, kernel_size), dtype=int)
                dilated_image = dilation(image_gray, kernel)
                st.image(dilated_image, caption='Dilation', use_column_width=True)

            elif morph_op == "Opening":
                kernel_size = st.sidebar.slider("Kernel Size", min_value=3, max_value=15, step=2, value=3)
                kernel = np.ones((kernel_size, kernel_size), dtype=int)
                opened_image = opening(image_gray, kernel)
                st.image(opened_image, caption='Opening', use_column_width=True)

            elif morph_op == "Closing":
                kernel_size = st.sidebar.slider("Kernel Size", min_value=3, max_value=15, step=2, value=3)
                kernel = np.ones((kernel_size, kernel_size), dtype=int)
                closed_image = closing(image_gray, kernel)
                st.image(closed_image, caption='Closing', use_column_width=True)

            elif morph_op == "Morphological Gradient":
                kernel_size = st.sidebar.slider("Kernel Size", min_value=3, max_value=15, step=2, value=3)
                kernel = np.ones((kernel_size, kernel_size), dtype=int)
                gradient_image = morphological_gradient(image_gray, kernel)
                st.image(gradient_image, caption='Morphological Gradient', use_column_width=True)

        elif selected_operation == "Region Growing":
            st.subheader("Region Growing")
            st.sidebar.write("Select seed point and threshold:")
            seed_x = st.sidebar.slider("Seed X", min_value=0, max_value=image_gray.shape[0]-1, step=1, value=image_gray.shape[0] // 2)
            seed_y = st.sidebar.slider("Seed Y", min_value=0, max_value=image_gray.shape[1]-1, step=1, value=image_gray.shape[1] // 2)
            threshold = st.sidebar.slider("Threshold", min_value=1, max_value=20, step=1, value=5)
            seed_point = (seed_x, seed_y)

            region_grown_image = region_growing(image_gray, seed_point, threshold)
            st.image(region_grown_image, caption='Region Growing', use_column_width=True)

        elif selected_operation == "Active Contour":
            st.subheader("Active Contour Segmentation")
            if uploaded_file is not None:
                gray_img, snake = active_contour_segmentation(image_rgb)

                # Display the results
                st.image(image_rgb, caption='Original Image', use_column_width=True)
                fig, ax = plt.subplots(figsize=(7, 7))
                ax.imshow(gray_img, cmap=plt.cm.gray)
                ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
                ax.set_xticks([]), ax.set_yticks([])
                ax.axis([0, gray_img.shape[1], gray_img.shape[0], 0])
                st.pyplot(fig)

        elif selected_operation == "Matrix Operations":
            st.subheader("Matrix Operations")
            st.sidebar.write("Select matrix operation:")
            matrix_op = st.sidebar.radio("", ("Convolution", "Correlation", "Fourier Transform", 
                                            "Inverse Fourier Transform", "Matrix Transformations"))

            if matrix_op == "Convolution":
                st.write("Performing Convolution...")
                kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])  # Example kernel for edge detection
                convolved_image = convolution(image_gray, kernel)
                normalized_image = normalize_image(convolved_image)
                st.image(normalized_image, caption='Convolution', use_column_width=True)

            elif matrix_op == "Correlation":
                st.write("Performing Correlation...")
                kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])  # Example kernel for edge detection
                correlated_image = correlation(image_gray, kernel)
                normalized_image = normalize_image(correlated_image)
                st.image(normalized_image, caption='Correlation', use_column_width=True)

            elif matrix_op == "Fourier Transform":
                st.write("Performing Fourier Transform...")
                fourier_image = np.log(np.abs(fourier_transform(image_gray)))
                normalized_image = normalize_image(fourier_image)
                st.image(normalized_image, caption='Fourier Transform', use_column_width=True)

            elif matrix_op == "Inverse Fourier Transform":
                st.write("Performing Inverse Fourier Transform...")
                transformed_image = fourier_transform(image_gray)
                inverse_fourier_image = np.abs(inverse_fourier_transform(transformed_image))
                normalized_image = normalize_image(inverse_fourier_image)
                st.image(normalized_image, caption='Inverse Fourier Transform', use_column_width=True)

            elif matrix_op == "Matrix Transformations":
                st.write("Performing Matrix Transformations...")
                # Example transformation matrix: rotation by 45 degrees
                angle = np.radians(45)
                transformation_matrix = np.array([
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1]
                ])
                transformed_image = matrix_transformations(image_gray, transformation_matrix)
                normalized_image = normalize_image(transformed_image)
                st.image(normalized_image, caption='Matrix Transformations', use_column_width=True)
      
        elif selected_operation == "Segmentation":
            st.subheader("Segmentation")
            st.sidebar.write("Select segmentation method:")
            seg_method = st.sidebar.radio("", ("Simple Thresholding",))

            if seg_method == "Simple Thresholding":
              
                threshold = st.sidebar.slider("Threshold", 0, 255, 127)
                segmented_image = simple_thresholding(image_gray, threshold)

                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                ax_img, ax_seg = axes

                ax_img.imshow(image_gray, cmap='gray')
                ax_img.axis('off')
                ax_img.set_title('Original Image')

                ax_seg.imshow(segmented_image, cmap='gray')
                ax_seg.axis('off')
                ax_seg.set_title('Segmented Image')

                st.pyplot(fig)
        
        elif selected_operation == "Görüntü Ayrıntıları":
            st.subheader("Görüntü Ayrıntıları")
            # Get image details
            image_details = get_image_details(image_gray)

           
           
            for key, value in image_details.items():
                st.write(f"{key}: {value}")

            # Display the original image
            st.image(image_gray, caption='Original Image', use_column_width=True)


           
if __name__ == "__main__":
    main()
