import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from datetime import datetime

def overlay_images(image_path1, image_path2, output_path):
    # Read the images
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)

    # Check if images have the same size
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions for overlay")

    # Create an overlay by averaging the pixel values
    overlay = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

    # Save the overlay image
    cv2.imwrite(output_path, overlay)

    # Display the images
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title('Overlay')
    plt.show()

# Example usage
# Get today's date
today = datetime.today().strftime('%Y%m%d')

# Define a function to generate a unique filename
def generate_unique_filename(base_path, base_name, extension):
    counter = 1
    while True:
        filename = f"{base_name}_{today}_{counter}.{extension}"
        if not os.path.exists(os.path.join(base_path, filename)):
            return filename
        counter += 1

# Define the base path and base name for the output file
base_path = 'stereo_model/calibration'
base_name = 'output_overlay_testing'
extension = 'jpg'

# Generate a unique filename
output_filename = generate_unique_filename(base_path, base_name, extension)
output_path = os.path.join(base_path, output_filename)
left_img_path = r'scene3d/stereo_utils/stereo_analysis_samples/cholest_at_1860sec_left.jpg'
right_img_path = r'scene3d/stereo_utils/stereo_analysis_samples/cholest_at_1860sec_right.jpg'
overlay_images(left_img_path, right_img_path, output_path)