import os
import cv2
import numpy as np

# Input and output directories
input_directory = '/Users/anishravuri/Desktop/Junior Year/Semester 1/DATS 4001/captcha_dataset/samples'  # Replace with the path to your dataset
output_directory = '/Users/anishravuri/Desktop/Junior Year/Semester 1/DATS 4001/processed images'  # Output directory to save processed images

# create kernel for morphological operation

kernel = cv2.getStructuringElement(cv2.MORPH_OPEN, (3,3))


# Create output directory
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Load images and preprocess them
for filename in os.listdir(input_directory):
    if filename.endswith('.png') or filename.endswith('.jpg'):  # Filter image file extensions
        # Load the image
        image_path = os.path.join(input_directory, filename)
        captcha_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_image = clahe.apply(captcha_image)

        # Apply binary thresholding and otsus thresholding
        _, binary_image = cv2.threshold(enhanced_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        

        # Invert image to black background and white foreground
        inverted_image = cv2.bitwise_not(binary_image)

        # finding contours of letters
        contours, _ = cv2.findContours(inverted_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # create a black background image to draw countours on
        result_image = np.zeros_like(inverted_image)

        # Draw contours in white on the black background
        cv2.drawContours(result_image, contours, -1, (255, 255, 255), thickness= cv2.FILLED)

        # implement a morpholocial openeing operation to remove stray blobs and fill in chipped edges

        opened_image = cv2.morphologyEx(result_image, cv2.MORPH_OPEN, kernel)

        # Maybe add an inpainting step to replace parts of segments removed by opening to preserve characters


        # Save the processed image with the same filename in the output directory
        output_path = os.path.join(output_directory, filename)
        cv2.imwrite(output_path, opened_image)

        print(f"Processed and saved: {output_path}")


