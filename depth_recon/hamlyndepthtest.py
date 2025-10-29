import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_mask(depth_image01, mask, depth_image02=None): 

    # Normalize the first depth image to the range 0 to 255
    normalized_image01 = depth_image01 * 255
    # normalized_image01 = cv2.normalize(depth_image01, None, 0, 255, cv2.NORM_MINMAX)
    normalized_image01 = normalized_image01.astype(np.uint8)

    if depth_image02 is not None:
        # Normalize the second depth image to the range 0 to 255
        normalized_image02 = cv2.normalize(depth_image02, None, 0, 255, cv2.NORM_MINMAX)
        normalized_image02 = normalized_image02.astype(np.uint8)

        # Compute the difference map
        difference_map = cv2.absdiff(normalized_image01, normalized_image02)
        # Compute the non-normalized difference map
        non_normalized_difference = cv2.absdiff(depth_image01, depth_image02)
    else:
        difference_map = None
        non_normalized_difference = None

    # Apply the mask to the first depth image
    masked_image01 = np.where(mask, depth_image01, 0)

    # Normalize the masked image for visualization
    normalized_masked_image01 = cv2.normalize(masked_image01, None, 0, 255, cv2.NORM_MINMAX)
    normalized_masked_image01 = normalized_masked_image01.astype(np.uint8)

    # Add the masked image to the plot
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3 if depth_image02 is None else 4, 1)
    plt.imshow(normalized_image01, cmap='gray')
    plt.colorbar(label='Pixel Intensity')
    plt.title('Normalized Depth Image 01')
    plt.axis('off')

    if depth_image02 is not None:
        plt.subplot(1, 4, 2)
        plt.imshow(normalized_image02, cmap='gray')
        plt.colorbar(label='Pixel Intensity')
        plt.title('Normalized Depth Image 02')
        plt.axis('off')

        plt.subplot(1, 4, 3)
        plt.imshow(difference_map, cmap='hot')
        plt.colorbar(label='Pixel Intensity Difference')
        plt.title('Difference Map')
        plt.axis('off')

    # Combine the red mask with the normalized masked image
    colored_masked_image01 = cv2.cvtColor(normalized_masked_image01, cv2.COLOR_GRAY2BGR)
    colored_masked_image01[~mask] = [0, 0, 255]  # BGR red

    plt.subplot(1, 3 if depth_image02 is None else 4, 3 if depth_image02 is None else 4)
    plt.imshow(colored_masked_image01)
    plt.title('Masked Depth Image 01 with Red Zone')
    plt.axis('off')

    plt.tight_layout()

    if depth_image02 is not None:
        # Print the range of the non-normalized difference map
        min_val = np.min(non_normalized_difference)
        max_val = np.max(non_normalized_difference)
        print(f"Non-normalized difference range: Min = {min_val}, Max = {max_val}")

    plt.show()

def create_mask(depth_image01, depth_image02=None):
    if depth_image02 is not None:
        # Create mask for valid depth values
        mask = np.logical_and(depth_image01 > 0, depth_image01 < 1000)  # Assuming max depth is 1000mm
        difference = np.abs(depth_image01 - depth_image02)
        threshold = 300  # 15mm threshold for depth difference
        mask = np.logical_and(mask, depth_image02 > 0, depth_image02 < 1000)
        mask = np.logical_and(mask, difference <= threshold)
    else:
        # Create mask for valid depth values for a single depth image
        mask = np.logical_and(depth_image01 > 3, depth_image01 < 500)  # Assuming max depth is 1000mm
    return mask

if __name__ == "__main__":
    # Example paths for depth images
    image01_path = 'hamlyn_data/rectified08/depth01/0000000100.png'
    image02_path = None

    # Read the depth images
    depth_image01 = cv2.imread(image01_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    depth_image02 = None  # Set to None if not available

    if image02_path:
        depth_image02 = cv2.imread(image02_path, cv2.IMREAD_UNCHANGED)
    else:
        depth_image02 = None

    mask = create_mask(depth_image01)
    # Visualize the mask and depth images
    visualize_mask(depth_image01, mask)