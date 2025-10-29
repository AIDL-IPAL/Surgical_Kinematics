import cv2
import numpy as np

def disparity_SGBM(left_img, right_img):
  # Set up StereoSGBM to compute disparity with standard parameters.
  window_size = 5
  min_disp = 2
  num_disp = 16 * 9  # Must be divisible by 16. Adjust as needed.
  stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=window_size,
    P1=8 * 3 * window_size**2,
    P2=32 * 3 * window_size**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM
  )
  
  disparity = stereo.compute(left_img, right_img).astype(np.float32) / 16.0  # Divide by 16 to get floating point disparity values.
  return disparity


def disparity_pyelas(frame_left, frame_right, **kwargs):
    """
    Compute disparity using the ELAS algorithm (PyElas).
    Current system requires linux bash commands to install & run the ELAS module.
    Parameters:
      left_gray: Grayscale left image.
      right_gray: Grayscale right image.
    
    Returns:
      disparity: A float32 disparity map.
    """
    import elas as pyelas # PyElas module; if installed with name pyelas, use import pyelas
    # Convert frames to grayscale for ELAS.
    left_gray = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
        
    height, width = left_gray.shape
    disparity = np.zeros((height, width), dtype=np.float32)
    # Create the ELAS object with default parameters.
    elas = pyelas.Elas(**kwargs)
    # Process the pair of images to compute disparity.
    disparity_l, disp_r = elas.process(left_gray, right_gray)
    disp_min = np.min(disparity_l)
    disp_max = np.max(disparity_l)
    # Avoid division by zero if no disparity is found.
    if disp_max - disp_min <= 0:
        print("No disparity")
    
    # Extract the disparity map for the first camera. (left)
    # disparity_l = disparity_l.astype(np.float32)
    return disp_r / 255.0 


