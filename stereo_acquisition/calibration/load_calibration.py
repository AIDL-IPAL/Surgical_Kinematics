import numpy as np


def load_calibration(stereo_params, img_size):
    """
    Loads the calibration analysis and reconstructs intrinsic and extrinsic parameters.
    
    Parameters:
      stereo_params: str
          Path to the .npy file containing the analysis dictionary.
      img_size: tuple(int, int)
          The (width, height) of the images (needed to set the principal point).
          
    Returns:
      calib_params: dictionary with keys:
         - 'left_camera_matrix'
         - 'left_dist'
         - 'right_camera_matrix'
         - 'right_dist'
         - 'R'
         - 'T'
         - 'E'
         - 'F'
         - (Optionally, other keys if you wish to reconstruct further parameters)
    """
    # Load the analysis dictionary (assumed saved with np.save(..., analysis))
    analysis = np.load(stereo_params, allow_pickle=True).item()
    
    # # Extract the mean translation vector (in mm; square_size calc in mm)
    # # T = analysis['T_mean']
    # T = np.array([5.5, 0.0, 0.1], dtype=np.float32)
    # T_dist_mean = analysis['T_dist_mean']
    
    # # Extract focal lengths (assumed in pixels)
    # focal_left = analysis['focal_left_mean']
    # focal_right = analysis['focal_right_mean']
    
    # # Set distortion values
    # dist_left = analysis['left_dist_mean']
    # dist_right = analysis['right_dist_mean']
    
    # # Assume principal point is at the center of the image.
    # width, height = img_size
    # cx, cy = width / 2.0, height / 2.0
    
    # mtxL = np.array([[focal_left, 0, cx],
    #                  [0, focal_left, cy],
    #                  [0, 0, 1]], dtype=np.float32)
    # mtxR = np.array([[focal_right, 0, cx],
    #                  [0, focal_right, cy],
    #                  [0, 0, 1]], dtype=np.float32)
    
    # # Extract rotation, essential, and fundamental matrices
    # R = analysis['R_mean']
    # E = analysis['E_mean']
    # F = analysis['F_mean']
    
    # calib_params = {
    #     'left_camera_matrix': mtxL,
    #     'left_camera_matrix_avg': analysis['left_camera_matrix_avg'],
    #     'left_dist': dist_left,
    #     'right_camera_matrix': mtxR,
    #     'right_dist': dist_right,
    #     'R': R,
    #     'T': T,
    #     'T_dist_mean': T_dist_mean,
    #     'E': E,
    #     'F': F
    # }
    return analysis