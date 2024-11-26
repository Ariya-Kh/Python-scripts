import os
import cv2
import numpy as np

def add_padding(image, bottom_padding, right_padding):
    # Get the height and width of the original image
    h, w = image.shape[:2]
    
    # Create a new blank (black) image with the new dimensions
    padded_image = np.zeros((h + bottom_padding, w + right_padding, 3), dtype=np.uint8)
    
    # Copy the original image into the padded image
    padded_image[:h, :w] = image
    
    return padded_image

def process_images(input_folder, output_folder, bottom_padding, right_padding):
    # Check if output folder exists, create it if not
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img_path = os.path.join(input_folder, filename)
            image = cv2.imread(img_path)

            if image is not None:
                # Add padding to the image
                padded_image = add_padding(image, bottom_padding, right_padding)

                # Save the new image to the output folder
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, padded_image)
                print(f"Processed {filename}")

if __name__ == "__main__":
    input_folder = "/media/perticon/Backup/datasets/Kalleh/1403-06-17-Kalleh/2/count"  # Replace with your input folder path
    output_folder = "/media/perticon/Backup/datasets/Kalleh/1403-06-17-Kalleh/2/count-pad"  # Replace with your output folder path
    bottom_padding = 1440-350  # Change to your desired bottom padding
    right_padding = 0  # Change to your desired right padding

    process_images(input_folder, output_folder, bottom_padding, right_padding)
