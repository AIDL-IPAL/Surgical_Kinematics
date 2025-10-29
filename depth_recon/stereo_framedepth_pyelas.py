import cv2
import numpy as np
import argparse
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def normalize_frame(frame):
    """
    Normalize the image to the range [0, 255].
    """
    return cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

def compute_depth_map(disp, left_img, right_img, Q):
    overlay = cv2.addWeighted(left_img, 0.5, right_img, 0.5, 0)
    import matplotlib.pyplot as plt
    # Display the images
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title('Rectified Overlay')
    plt.show()
    print("Computing disparity...")
    # from disparity import disparity_SGBM
    # disparity = disparity_SGBM(rect_left, rect_right)
    if disp == 'SGBM':
        # Compute disparity using SGBM.
        from disparity import disparity_SGBM
        disparity = disparity_SGBM(left_img, right_img)
    elif disp == 'PyElas':
        from disparity import disparity_pyelas
    elif disp == 'NVFS':
        try:
            # !python3 depth_NVFS.py
            import FoundationStereo.Utils
        except ImportError:
            print("NVFS depth estimation not available here. Please set up system.")
            return

        # # Adjusting pyelas hyperparameters for better disparity results
        pyelas_params = {
            "disp_min": 0,
            "disp_max": 128,  # reasonable upper bound for 4–5mm baseline at 720p
            "support_threshold": 0.85,  # lower slightly to allow more support points
            "support_texture": 15,  # increase to reject low-texture points
            "candidate_stepsize": 2,  # finer steps for dense candidate search
            "incon_window_size": 5,
            "incon_threshold": 2,
            "incon_min_support": 5,  # loosen support requirement
            "add_corners": True,
            "grid_size": 5,  # more support points overall
            "beta": 0.1,      # slightly higher penalty
            "gamma": 7,       # higher smoothness
            "sigma": 2.0,     # stronger Gaussian smoothing
            "sradius": 15,     # small search window
        }

        disparity = disparity_pyelas(left_img, right_img)
        # disparity = cv2.bilateralFilter(disparity.astype(np.float32), 7, 75, 75)

    # disparity[disparity <= 0] = 0.1  # avoid zeros
    
    # Reproject disparity to 3D
    points_3d = cv2.reprojectImageTo3D(disparity, Q)
    depth_map = points_3d[:, :, 2]  # Z axis

    # Normalize & threshold the disparity map to 0-255 for visualization.
    mean_val = depth_map.mean()
    print(f"Mean depth: {mean_val}") # in mm
    depth_min = np.min(depth_map)
    depth_max = np.max(depth_map)

    # thresholding
    depth_map[depth_map > 300] = 300
    depth_map[depth_map < 0] = 0

    return disparity, depth_map, mean_val, depth_min, depth_max

def main(disp='SGBM'):
    """
    Main function to compute and visualize depth map from stereo images.
    """
    # Specify the paths to the images and calibration parameters.
    import sys
    from pathlib import Path
    THIS_FILE = Path(__file__).resolve()             # …/your_script.py
    DIR_ROOT = THIS_FILE.parents[0]              # adjust depth as needed
    # left_img_path = 'stereo_model/calibration/03_08_2025/jigsawcut/frame_at_3sec_left.jpg'
    # right_img_path = 'stereo_model/calibration/03_08_2025/jigsawcut/frame_at_3sec_right.jpg'
    # left_img_path = 'stereo_model/stereo_analysis_samples/JIGSAWS_frame_at_3sec_left.jpg'
    # right_img_path = 'stereo_model/stereo_analysis_samples/JIGSAWS_frame_at_3sec_right.jpg' remember to change calibrtion
    left_img_path = DIR_ROOT / 'stereo_analysis_samples/cholest_at_1860sec_left.jpg'
    right_img_path = DIR_ROOT / 'stereo_analysis_samples/cholest_at_1860sec_right.jpg'
    params_path = DIR_ROOT / 'calibration/calibr_results/03_08_2025/calibration_analysis.npy'

    # Check if files exist
    if not os.path.exists(left_img_path):
        print(f"Left image file does not exist: {left_img_path}")
        return
    if not os.path.exists(right_img_path):
        print(f"Right image file does not exist: {right_img_path}")
        return
    if not os.path.exists(params_path):
        print(f"Calibration parameters file does not exist: {params_path}")
        return
    
    # Load stereo images.
    left_img  = cv2.imread(left_img_path)
    right_img = cv2.imread(right_img_path)
    if left_img is None or right_img is None:
        print("Error loading images.")
        return
    
    # Load calibration parameters (assumed to be a dictionary saved with np.save)
    img_size = (left_img.shape[1], left_img.shape[0])
    from calibration.load_calibration import load_calibration
    calib_params = load_calibration(params_path, img_size)
    rect_left, rect_right, Q, T = rectify_images(left_img, right_img, calib_params)
    # Tile images: top row with original images, bottom row with rectified images.
    # top_row = np.hstack((left_img, right_img))
    # bottom_row = np.hstack((rect_left, rect_right))
    # tiled_output = np.vstack((top_row, bottom_row))
    
    # cv2.imshow("Tiled Stereo Images", tiled_output)
    # print("Press any key in the image window to continue...")
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # Compute the depth map.
    disparity, depth_map, mean_val, depth_min, depth_max = compute_depth_map(disp, rect_left, rect_right, Q)
    
    
    # Display the results.
    # combined_display = np.hstack((left_img, depth_color))
    # cv2.imshow("Stereo Depth Visualization", combined_display)
    
    import matplotlib
    import matplotlib.pyplot as plt

    # plt.imshow(
    #     depth_map, 
    #     cmap='jet',
    #     vmin=depth_min,
    #     vmax=depth_max
    # )
    # plt.gca().set_aspect(float(depth_map.shape[0]) / float(depth_map.shape[1]))
    # plt.colorbar(label='Distance (mm)')
    # plt.title('Depth Map (Scaled)')
    # plt.show()

    # while True:
    #     if cv2.getWindowProperty("Depth Map (Color)", cv2.WND_PROP_VISIBLE) < 1:
    #         break
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # cv2.destroyAllWindows()

    # Optionally, you can save the depth image:
    s = input("Save depth map? (y/enter or n): ")
    if s.lower() == 'y' or s == '':
        analysis_dir = DIR_ROOT / "depth_results_tmp" / "framedepth"
        app = str(disp).lower()
        analysis_dir = os.path.join(analysis_dir, app)
        if not os.path.exists(analysis_dir):
            os.makedirs(analysis_dir)

        norm_depth_map = normalize_frame(depth_map)
        norm_disparity = normalize_frame(disparity)
        depth_color = cv2.applyColorMap((norm_depth_map * 255).astype(np.uint8), cv2.COLORMAP_JET)

        cv2.imwrite(os.path.join(analysis_dir, "depth_normalized.png"), (norm_depth_map * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(analysis_dir, "disparity_normalized.png"), (norm_disparity * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(analysis_dir, "left_img.png"), left_img)
        cv2.imwrite(os.path.join(analysis_dir, "rect_left.png"), left_img)
        cv2.imwrite(os.path.join(analysis_dir, "depth_color.png"), depth_color)
        plt.savefig(os.path.join(analysis_dir, "depth_map_legend.png"))
        disp_normalized = normalize_frame(disparity)
        cv2.imwrite(os.path.join(analysis_dir, "disparity.png"), (disp_normalized * 255).astype(np.uint8))
        np.save(os.path.join(analysis_dir, "depth_map.npy"), depth_map)
        print(f"Saved results to {analysis_dir}")

if __name__ == "__main__":
    main()