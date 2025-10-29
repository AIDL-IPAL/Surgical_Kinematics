import cv2
import numpy as np
import glob
import os
from datetime import datetime
import sys

def calibrate_test_set(test_set_dir, checkerboard_size, square_size, verbose=True):
    """
    Calibrates a stereo rig for one test set.
    
    Parameters:
      test_set_dir: str
          The base directory for the test set; expects subdirectories "left" and "right"
      checkerboard_size: tuple(int, int)
          The number of inner corners per row and column (e.g., (7, 7))
      square_size: float
          The physical size of one square on your checkerboard (e.g., in millimeters)
    
    Returns:
      A dictionary with calibration parameters:
         { 'left_camera_matrix': ..., 'left_dist': ...,
           'right_camera_matrix': ..., 'right_dist': ...,
           'R': ..., 'T': ..., 'E': ..., 'F': ...,
           'reprojection_error': ... }
    """
    
    # Prepare object points based on your physical checkerboard dimensions.
    objp = np.zeros((checkerboard_size[0]*checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp = objp * square_size  # scale by the square size

    # Arrays to store object points and image points from all images.
    objpoints = []  # 3d points in real-world space
    imgpoints_left = []  # 2d points in left image
    imgpoints_right = []  # 2d points in right image

    # Assume test_set_dir contains subdirectories "left" and "right" with images.
    print("Current working directory:", os.getcwd()) if verbose else None

    right_dir = os.path.join(test_set_dir, "camera0_*.jpg")
    left_dir = os.path.join(test_set_dir, "camera1_*.jpg")
    left_imgs = sorted(glob.glob(left_dir))
    right_imgs = sorted(glob.glob(right_dir))
    print("Left images dir:", left_dir) if verbose else None
    print(f"Found {len(left_imgs)} left images and {len(right_imgs)} right images.") if verbose else None

    if len(left_imgs) != len(right_imgs):
        raise ValueError("The number of left and right images does not match.")
    
    # Termination criteria for corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

    for left_img_path, right_img_path in zip(left_imgs, right_imgs):
        left_img = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
        print("Left image shape:", left_img.shape)
        right_img = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)

        # Find the chessboard corners for both images.
        ret_left, corners_left = cv2.findChessboardCornersSB(left_img, (checkerboard_size[0], checkerboard_size[1]), flags=cv2.CALIB_CB_EXHAUSTIVE)
        ret_right, corners_right = cv2.findChessboardCornersSB(right_img, (checkerboard_size[0], checkerboard_size[1]), flags=cv2.CALIB_CB_EXHAUSTIVE)

        if ret_left and ret_right:
            # Refine corner locations
            corners_left = cv2.cornerSubPix(left_img, corners_left, (checkerboard_size[0], checkerboard_size[1]), (-1, -1), criteria)
            corners_right = cv2.cornerSubPix(right_img, corners_right, (checkerboard_size[0], checkerboard_size[1]), (-1, -1), criteria)
            imgpoints_left.append(corners_left)
            imgpoints_right.append(corners_right)
            objpoints.append(objp)
        else:
            print(f"Corner detection failed for images:\n {left_img_path}\n {right_img_path}")

    # Get image size from first left image
    img_shape = left_img.shape[::-1]  # (width, height)

    # Calibrate each camera separately
    ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
        objpoints, imgpoints_left, img_shape, None, None)
    ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
        objpoints, imgpoints_right, img_shape, None, None)

    # Stereo calibration
    flags = cv2.CALIB_FIX_INTRINSIC
    stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    ret_stereo, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        mtx_left, dist_left,
        mtx_right, dist_right,
        img_shape,
        criteria=stereocalib_criteria,
        flags=flags
    )

    # Compute the reprojection error (currently using ret_stereo)
    reproj_error = ret_stereo

    # Return all calibration parameters
    calib_results = {
        'left_camera_matrix': mtx_left,
        'left_dist': dist_left,
        'right_camera_matrix': mtx_right,
        'right_dist': dist_right,
        'R': R,
        'T': T,
        'E': E,
        'F': F,
        'reprojection_error': reproj_error
    }
    return calib_results

def analyze_calibration_results(results_list):
    """
    Given a list of calibration result dictionaries from multiple test sets,
    compute averages and variance for select parameters.
    
    Parameters:
      results_list: list of dict
          Each dictionary is the output of calibrate_test_set().
          
    Returns:
      A dictionary with averaged values and variances for, for example:
        - reprojection_error
        - T (translation vector)
        - Focal lengths (extracted from the camera matrices)
    """
    reproj_errors = []
    T_vectors = []
    focal_left = []
    focal_right = []
    
    for res in results_list:
        reproj_errors.append(res['reprojection_error'])
        T_vectors.append(res['T'].flatten())
        focal_left.append(res['left_camera_matrix'][0,0])   # assuming fx is at [0,0]
        focal_right.append(res['right_camera_matrix'][0,0])
    
    reproj_errors = np.array(reproj_errors)
    T_vectors = np.array(T_vectors)
    T_dist = np.sqrt(T_vectors[:,0]**2 + T_vectors[:,1]**2 + T_vectors[:,2]**2)
    print("Translation vector distances:", T_dist) if len(T_dist) > 0 else None
    T_dist_mean = np.mean(T_dist)
    focal_left = np.array(focal_left)
    focal_right = np.array(focal_right)
    
    left_cam_list = [res['left_camera_matrix'] for res in results_list]
    right_cam_list = [res['right_camera_matrix'] for res in results_list]

    # Calculate per-element average for left and right camera matrices
    left_camera_matrix_avg = np.zeros_like(left_cam_list[0])
    right_camera_matrix_avg = np.zeros_like(right_cam_list[0])
    for i in range(left_camera_matrix_avg.shape[0]):
        for j in range(left_camera_matrix_avg.shape[1]):
            left_camera_matrix_avg[i, j] = np.mean([m[i, j] for m in left_cam_list])
            right_camera_matrix_avg[i, j] = np.mean([m[i, j] for m in right_cam_list])
    
    print("Left camera matrix average:\n", left_camera_matrix_avg)
    print("Left camera matrix list:\n", left_cam_list)
    print("Right camera matrix average:\n", right_camera_matrix_avg)

    analysis = {
        'reprojection_error_mean': np.mean(reproj_errors),
        'reprojection_error_var': np.var(reproj_errors),
        'left_camera_matrix': left_cam_list,
        'right_camera_matrix': right_cam_list,
        'left_camera_matrix_avg': left_camera_matrix_avg,
        'right_camera_matrix_avg': right_camera_matrix_avg,
        'R_mean': np.mean([res['R'] for res in results_list], axis=0),
        'R_var': np.var([res['R'] for res in results_list], axis=0),
        'T_mean': np.mean(T_vectors, axis=0),
        'T_dist_mean': T_dist_mean,
        'T_var': np.var(T_vectors, axis=0),
        'E_mean': np.mean([res['E'] for res in results_list], axis=0),
        'E_var': np.var([res['E'] for res in results_list], axis=0),
        'F_mean': np.mean([res['F'] for res in results_list], axis=0),
        'F_var': np.var([res['F'] for res in results_list], axis=0),
        'focal_left_mean': np.mean(focal_left),
        'focal_left_var': np.var(focal_left),
        'focal_right_mean': np.mean(focal_right),
        'focal_right_var': np.var(focal_right),
        'left_dist_mean': np.mean([res['left_dist'] for res in results_list], axis=0),
        'left_dist_var': np.var([res['left_dist'] for res in results_list], axis=0),
        'right_dist_mean': np.mean([res['right_dist'] for res in results_list], axis=0),
        'right_dist_var': np.var([res['right_dist'] for res in results_list], axis=0)
    }
    return analysis

if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))
    sys.path.append(os.path.abspath('scene3d'))

    # print("Current working directory:", os.getcwd())
    # print(f"File directory: {os.path.dirname(__file__)}")
    test_set_dirs = [
        # test directories containing left (camera0_*) and right (camera1_*) camera img files
        "scene3d/stereo_utils/calibration/acq_data/03_08_2025/small1",
        "scene3d/stereo_utils/calibration/acq_data/03_08_2025/supermicro1",
        # "stereo_model/calibr_data/8x6/",
        # "stereo_model/calibr_data/small/",
        # "stereo_model/calibr_data/supermicro/",
        # "stereo_model/calibr_data/supermicro_lowlight/",
        # "stereo_model/calibr_data/supermicro_robo/"
    ]
    
    test_set_configs = [
        {"dir": test_set_dirs[0], "checkerboard_size": (8, 6), "square_size": 4.74}, # NxM, square in mm
        {"dir": test_set_dirs[1], "checkerboard_size": (8, 6), "square_size": 2.39},
        # {"dir": test_set_dirs[2], "checkerboard_size": (8, 6), "square_size": 4.74},
        # {"dir": test_set_dirs[3], "checkerboard_size": (8, 6), "square_size": 2.39},
        # {"dir": test_set_dirs[4], "checkerboard_size": (8, 6), "square_size": 2.39},
        # {"dir": test_set_dirs[5], "checkerboard_size": (8, 6), "square_size": 2.39}
    ]
    
    all_results = []
    for config in test_set_configs:
        print(f"Calibrating test set in {config['dir']} with pattern {config['checkerboard_size']}")
        try:
            res = calibrate_test_set(config["dir"], config["checkerboard_size"], config["square_size"])
            all_results.append(res)
        except Exception as e:
            print(f"Error calibrating test set {config['dir']}: {e}")
            continue
    
    if all_results:
        analysis = analyze_calibration_results(all_results)
        print("Calibration Analysis Across Test Sets:")
        print("Mean reprojection error:", analysis['reprojection_error_mean'])
        print("Reprojection error variance:", analysis['reprojection_error_var'])
        print("Mean translation vector T:", analysis['T_mean'])
        print("Translation variance:", analysis['T_var'])
        print("Mean left focal length:", analysis['focal_left_mean'])
        print("Left focal length variance:", analysis['focal_left_var'])
        print("Mean right focal length:", analysis['focal_right_mean'])
        print("Right focal length variance:", analysis['focal_right_var'])
        print("Mean translation vector distance:", analysis['T_dist_mean'])
        print("Mean left camera matrix:")
        print(analysis['left_camera_matrix_avg'])
        # Save the results to a file
        date_from_path = os.path.basename(os.path.dirname(test_set_configs[0]["dir"]))  # Extract date from the path's parent directory
        timestamp = date_from_path  # Format date to remove underscores
        savepath = os.path.join("scene3d/stereo_utils/calibration/calibr_results", timestamp)
        os.makedirs(savepath, exist_ok=True)
        np.save(os.path.join(savepath, "calibration_analysis.npy"), analysis)
 