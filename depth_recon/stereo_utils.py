import cv2
import numpy as np
import os
import sys
from tqdm import tqdm



def load_intrinsics(intrinsic_file, scale=1.0):
    # Load intrinsics
    try:
        with open(intrinsic_file, 'r') as f:  # Open intrinsic file, units in mm
            lines = f.readlines()
            if len(lines) < 3:
                raise ValueError(f"Intrinsic file {intrinsic_file} must contain at least 3 lines.")
        intrinsic_matrix = np.array([list(map(float, line.strip().split()[:3])) for line in lines], dtype=np.float32)
        # Construct K (3x3)
        K = np.zeros((3, 3), dtype=np.float32)
        # self.K in pixel units
        K[0, 0] = intrinsic_matrix[0, 0]  # fx
        K[1, 1] = intrinsic_matrix[1, 1]  # fy
        K[0, 2] = intrinsic_matrix[0, 2]  # cx
        K[1, 2] = intrinsic_matrix[1, 2]  # cy
        K[2, 2] = 1.0

    except Exception as e:
        raise IOError(f"Error reading intrinsic or extrinsic files: {e}")

    # Intrinsic scaling if needed
    K_scaled = K.copy()
    K_scaled[:2] *= scale  # Scale intrinsics if you downscale images

    # print(f"Loaded intrinsic matrix:\n{K_scaled}")
    return K_scaled

def load_extrinsics(extrinsic_file, metric_unit='mm'):

    # Load extrinsics
    with open(extrinsic_file, 'r') as f:
        lines = f.readlines()
    lines = [line for line in lines if len(line.strip().split()) >= 4]
    if len(lines) < 3:
        raise ValueError(f"Extrinsic file {extrinsic_file} must contain at least 3 valid lines.")
    # Parse Rotation (R) and Translation (t)
    t = np.array([float(line.strip().split()[3]) for line in lines[:3]], dtype=np.float32)
    R = np.array([list(map(float, line.strip().split()[:3])) for line in lines[:3]], dtype=np.float32)  # (3x3)

    if metric_unit == 'm': t = t / 1000.0  # Convert mm → meters
    baseline = np.linalg.norm(t)  # Baseline in meters

    return baseline, R, t


def rectify_images(left_img, right_img, calib_path, visualize=False):
    """
    Given a stereo pair and reconstructed calibration parameters from analysis,
    compute a colorized depth map.
    """
    print("Calibration Parameters:")
    # if not calib_params:
    #     print("No calibration parameters provided. Using default values.")
    #     # return None, None, None, None
    # else:
    #     for key, value in calib_params.items():
    #         print(f"{key}: {value}")

    # Load intrinsic and extrinsic parameters from files
    try:
        intrinsic_file = os.path.join(calib_path, '01', 'intrinsics.txt')
        intrinsic_kr_file = os.path.join(calib_path, '01',  'intrinsics_kr.txt')
        extrinsic_file = os.path.join(calib_path, '01',  'extrinsics.txt')
    except Exception as e:
        print(f"Error loading calibration files: {e}")
        intrinsic_file = os.path.join(calib_path, 'intrinsics.txt')
        intrinsic_kr_file = os.path.join(calib_path, 'intrinsics_kr.txt')
        extrinsic_file = os.path.join(calib_path, 'extrinsics.txt')
        print("loaded now")    
    
    # Load intrinsic parameters
    Kl = load_intrinsics(intrinsic_file, scale=1.0)  # Left camera intrinsics
    Kr = load_intrinsics(intrinsic_kr_file, scale=1.0)  # Right camera intrinsics

    # Load distortion coefficients (direclty defined from Northwell Calibration on 05/16/2025)
    Dl = np.array([-3.33823958e-02, 4.69750137e-01, -2.35926364e-04, 1.15611262e-04, -1.54273657e+00], dtype=np.float64)  # Left camera distortion coefficients
    Dr = np.array([1.97474796e-02, -2.19392875e-01, -1.12231727e-03, -8.18694203e-04, 9.28765797e-01], dtype=np.float64)  # Right camera distortion coefficients

    # Load extrinsic parameters
    baseline, R, t = load_extrinsics(extrinsic_file, metric_unit='mm')  # Translation vector (baseline)
    T = np.array([baseline, 0.0, 0.0])  # Translation vector (baseline) in mm
    print("Translation vector (T):", T)

    # print('Left camera:')
    # print(Kl)
    # print('Left camera distortion:')
    # print(Dl)
    # print('Right camera:')
    # print(Kr)
    # print('Right camera distortion:')
    # print(Dr)
    # print('Rotation matrix:')
    # print(R)
    # print('Translation:')
    # print(T)

    # Critical: Check magnitude and unit (should be a few mm)
    # print("T magnitude (mm?):", np.linalg.norm(T))
    # Adjust T if necessary (usually baseline should be around a few mm)
    if np.linalg.norm(T) > 100:
        print("Scaling down T by 1000 (assume mm to meters)")
        T = T / 1000.0  # convert mm to meters

    img_size = (720, 480)  # explicitly defined correctly: width x height
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        Kl, Dl, Kr, Dr, img_size, R, T, alpha=0
    )

    # Rectification maps
    mapLx, mapLy = cv2.initUndistortRectifyMap(Kl, Dl, R1, P1, img_size, cv2.CV_32FC1)
    mapRx, mapRy = cv2.initUndistortRectifyMap(Kr, Dr, R2, P2, img_size, cv2.CV_32FC1)

    # Perform rectification
    rect_left = cv2.remap(left_img, mapLx, mapLy, cv2.INTER_LINEAR)
    rect_right = cv2.remap(right_img, mapRx, mapRy, cv2.INTER_LINEAR)
    if visualize:
        import matplotlib.pyplot as plt
        # Visualize results
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0, 0].imshow(left_img)
        axes[0, 0].set_title('Original Left')
        axes[0, 1].imshow(right_img)
        axes[0, 1].set_title('Original Right')
        axes[1, 0].imshow(rect_left)
        axes[1, 0].set_title('Rectified Left')
        axes[1, 1].imshow(rect_right)
        axes[1, 1].set_title('Rectified Right')
        plt.show()

    return rect_left, rect_right, Q, T
