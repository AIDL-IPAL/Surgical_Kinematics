import cv2
import numpy as np
import os

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
    """
    analysis = np.load(stereo_params, allow_pickle=True).item()
    
    # T = analysis['T_mean']
    T = np.array([5.5, 0.0, 0.0], dtype=np.float32) # manually setting the translation vector for testing
    focal_left = analysis['focal_left_mean']
    focal_right = analysis['focal_right_mean']
    dist_left = analysis['left_dist_mean']
    dist_right = analysis['right_dist_mean']
    
    width, height = img_size
    cx, cy = width / 2.0, height / 2.0
    
    mtxL = np.array([[focal_left, 0, cx],
                     [0, focal_left, cy],
                     [0, 0, 1]], dtype=np.float32)
    mtxR = np.array([[focal_right, 0, cx],
                     [0, focal_right, cy],
                     [0, 0, 1]], dtype=np.float32)
    
    R = analysis['R_mean']
    E = analysis['E_mean']
    F = analysis['F_mean']
    
    calib_params = {
        'left_camera_matrix': mtxL,
        'left_dist': dist_left,
        'right_camera_matrix': mtxR,
        'right_dist': dist_right,
        'R': R,
        'T': T,
        'E': E,
        'F': F
    }
    return calib_params

def compute_disparity_pyelas(frame_left, frame_right):
    """
    Compute disparity using the ELAS algorithm (PyElas).
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
    elas = pyelas.Elas()
    # Process the pair of images to compute disparity.
    disparity_left, disparity_right = elas.process(left_gray, right_gray)
    return disparity_right

def compute_depth_map(left_img, right_img, calib_params):
    """
    Given a stereo pair and calibration parameters, compute a colorized depth map.
    """
    mtxL    = calib_params['left_camera_matrix']
    distL   = calib_params['left_dist']
    mtxR    = calib_params['right_camera_matrix']
    distR   = calib_params['right_dist']
    T       = calib_params['T']
    
    img_size = (left_img.shape[1], left_img.shape[0])
    
    # For rectification we assume an identity rotation matrix (simplification)
    R = np.eye(3, dtype=np.float32)
    
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        mtxL, distL, mtxR, distR, img_size, R, T, alpha=0
    )
    
    map1x, map1y = cv2.initUndistortRectifyMap(mtxL, distL, R1, P1, img_size, cv2.CV_16SC2)
    map2x, map2y = cv2.initUndistortRectifyMap(mtxR, distR, R2, P2, img_size, cv2.CV_16SC2)
    
    rect_left  = cv2.remap(left_img, map1x, map1y, cv2.INTER_LINEAR)
    rect_right = cv2.remap(right_img, map2x, map2y, cv2.INTER_LINEAR)
    
    # Set up StereoSGBM parameters.
    window_size = 1
    min_disp = 10
    num_disp = 16 * 5
    stereo = cv2.StereoSGBM_create(
        minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = window_size,
        P1 = 8 * 3 * window_size**2,
        P2 = 32 * 3 * window_size**2,
        disp12MaxDiff = 1,
        uniquenessRatio = 5,
        speckleWindowSize = 50,
        speckleRange = 2,
        preFilterCap = 63,
        mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    
    # disparity = stereo.compute(rect_left, rect_right).astype(np.float32) / 16.0
    # disparity[disparity <= 0] = 0.1
    disparity = compute_disparity_pyelas(rect_left, rect_right).astype(np.float32)
    points_3d = cv2.reprojectImageTo3D(disparity, Q)
    depth_map = points_3d[:, :, 2]
    
    # Normalize depth for visualization.
    depth_map[np.isnan(depth_map)] = 0
    depth_map[np.isinf(depth_map)] = 0
    depth_norm = cv2.normalize(depth_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    depth_norm = np.uint8(depth_norm)
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    
    return depth_color, disparity, depth_map

def main():
    # Paths to the two video files and calibration parameters.
    video_path_left  = r"stereo_model/stereo_analysis_samples/camera0_20250221_175113_chunk.avi"
    video_path_right = r"stereo_model/stereo_analysis_samples/camera1_20250221_175113_chunk.avi"
    params_path = r'stereo_model/calibr_data/calibr_results/calibration_analysis_20250224-170141.npy'
    
    # Open the video files.
    cap_left = cv2.VideoCapture(video_path_left)
    cap_right = cv2.VideoCapture(video_path_right)
    if not cap_left.isOpened():
        print(f"Error opening left video file: {video_path_left}")
        return
    if not cap_right.isOpened():
        print(f"Error opening right video file: {video_path_right}")
        return
    
    # Get video properties.
    fps = cap_left.get(cv2.CAP_PROP_FPS)
    total_frames_left = cap_left.get(cv2.CAP_PROP_FRAME_COUNT)
    total_frames_right = cap_right.get(cv2.CAP_PROP_FRAME_COUNT)
    
    # Both videos should have the same frame dimensions.
    width  = int(cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Load calibration parameters using the single camera image size.
    calib_params = load_calibration(params_path, (width, height))
    
    # Define output video writer for the depth video.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video_path = r'stereo_model/depth_results_tmp/PyElas/depth_video.mp4'
    depth_writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))
    
    # Process only the first N seconds.
    max_playtime = 10.0
    max_frames = int(fps * max_playtime)
    frame_count = 0
    
    while frame_count < max_frames:
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()
        
        # If either capture fails, break out of the loop.
        if not ret_left or not ret_right:
            break
        
        # Compute the depth map using the corresponding frames.
        depth_color, disparity, depth_map = compute_depth_map(frame_left, frame_right, calib_params)
        
        # Display the depth map.
        # cv2.imshow("Depth Map (Color)", depth_color)
        
        # Write the depth frame to the output video.
        depth_writer.write(depth_color)
        
        # Allow early exit on 'q' key.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
    cap_left.release()
    cap_right.release()
    depth_writer.release()
    cv2.destroyAllWindows()
    print(f"Processed {frame_count} frames. Depth video saved to {out_video_path}")

if __name__ == "__main__":
    main()
