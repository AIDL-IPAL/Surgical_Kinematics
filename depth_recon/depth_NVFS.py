
# Utilizing FoundationStereo 2025, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0

import os
import sys
import argparse
import logging
import cv2
import numpy as np
import torch
import open3d as o3d
from omegaconf import OmegaConf
import imageio.v3 as iio # Use imageio v3 for imread

# --- Dependency Handling ---
# Issues: Check if core and Utils are in the expected locations relative to this script
try:
    # Adjust the path modification based on your project structure
    # This assumes depth_NVFS.py is in scene3d/stereo_utils/
    code_dir = os.path.dirname(os.path.realpath(__file__))
    NV_root = os.path.abspath(os.path.join(code_dir, '..', '..', 'FoundationStereo'))
    print(f"Adding {NV_root} to sys.path")
    sys.path.append(NV_root) # Add FoundationStereo root to sys.path
    # core_path = os.path.join(project_root_guess, 'core')
    # utils_path = os.path.join(project_root_guess) # Assuming Utils.py is in the root

    from core.utils.utils import InputPadder, bilinear_sampler
    from core.foundation_stereo import FoundationStereo
    from Utils import *
    import traceback    # Assuming Utils.py contains these helpers

except ImportError as e:
    print(f"Warning: Could not import necessary modules ({e}).")
    print("Please ensure 'core' and 'Utils.py' are in the correct locations relative to this script,")
    print("or adjust the sys.path modifications above.")
    print("Functionality may be limited.")

try:
    autocast = torch.cuda.amp.autocast
except:
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


# --- Main Utility Class ---
class DepthProcessorNVFS:
    """
    A utility class for framewise depth estimation using NVIDIA's Foundation Stereo model.

    Encapsulates model loading, preprocessing, inference, and postprocessing steps.
    """
    def __init__(self, ckpt_dir, intrinsic_file, extrinsic_file, metric_unit='mm', scale=1.0, hiera=False, valid_iters=32,
                 remove_invisible=True, z_far=10.0, denoise_cloud=True,
                 denoise_nb_points=30, denoise_radius=0.03, device='cuda'):
        """
        Initializes the NVidia Foundation Stereo depth processor.

        Args:
            ckpt_dir (str): Path to the pretrained model checkpoint (.pth file).
            intrinsic_file (str): Path to the camera intrinsic matrix and baseline file.
            extrinsic_file (str): Path to the camera extrinsic matrix file.
            scale (float): Downsize factor for input images (<=1). Default: 1.0.
            hiera (bool): Use hierarchical inference (for high-res images). Default: False.
            valid_iters (int): Number of flow-field updates during inference. Default: 32.
            remove_invisible (bool): Remove non-overlapping points from point cloud. Default: True.
            z_far (float): Maximum depth value for point cloud clipping. Default: 10.0.
            denoise_cloud (bool): Whether to denoise the output point cloud. Default: True.
            denoise_nb_points (int): Number of neighbors for radius outlier removal. Default: 30.
            denoise_radius (float): Radius for outlier removal. Default: 0.03.
            device (str): Device to run inference on ('cuda' or 'cpu'). Default: 'cuda'.
        """
        self.metric_unit = metric_unit
        self.scale = scale
        self.hiera = hiera
        self.valid_iters = valid_iters
        self.remove_invisible = remove_invisible
        self.z_far = z_far
        self.denoise_cloud = denoise_cloud
        self.denoise_nb_points = denoise_nb_points
        self.denoise_radius = denoise_radius
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.mixed_precision = torch.cuda.is_available() and device == 'cuda' # Enable mixed precision if on CUDA

        if not os.path.isfile(ckpt_dir):
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_dir}")
        
        # Load configuration from cfg.yaml located with the checkpoint
        cfg_path = os.path.join(os.path.dirname(ckpt_dir), 'cfg.yaml')
        if not os.path.exists(cfg_path):
             # Fallback: Create a minimal default config if cfg.yaml is missing
             logging.warning(f"cfg.yaml not found at {cfg_path}. Using default vit_size 'vitl'.")
             cfg = OmegaConf.create({'vit_size': 'vitl'}) # Default or common value
        else:
             cfg = OmegaConf.load(cfg_path)

        # Ensure essential keys exist, potentially overriding with init args
        cfg.setdefault('vit_size', 'vitl') # Default if not in cfg.yaml
        # Update cfg with parameters passed to the constructor for consistency
        cfg.scale = self.scale
        cfg.hiera = self.hiera
        cfg.valid_iters = self.valid_iters
        cfg.remove_invisible = self.remove_invisible
        cfg.z_far = self.z_far
        cfg.denoise_cloud = self.denoise_cloud
        cfg.denoise_nb_points = self.denoise_nb_points
        cfg.denoise_radius = self.denoise_radius
        self.args = cfg # Store merged config

        # load intrinsic and extrinsic into K and baseline
        if not os.path.isfile(intrinsic_file):
            raise FileNotFoundError(f"Intrinsic file not found: {intrinsic_file}")
        self.K_scaled, self.baseline = self.load_intrinsics(intrinsic_file, extrinsic_file)

        logging.info(f"Initializing FoundationStereo model with args:\n{OmegaConf.to_yaml(self.args)}")
        logging.info(f"Using pretrained model from {ckpt_dir}")

        self.model = FoundationStereo(self.args)
        ckpt = torch.load(ckpt_dir, map_location=self.device, weights_only=False) # Load to target device
        logging.info(f"Loaded checkpoint global_step: {ckpt.get('global_step', 'N/A')}, epoch: {ckpt.get('epoch', 'N/A')}")

        # Adjust for potential DataParallel wrapping or different state_dict structure
        model_state_dict = ckpt.get('model', ckpt.get('state_dict', ckpt))
        if model_state_dict:
            # Remove 'module.' prefix if present (from DataParallel)
            if list(model_state_dict.keys())[0].startswith('module.'):
                model_state_dict = {k.replace('module.', '', 1): v for k, v in model_state_dict.items()}
            self.model.load_state_dict(model_state_dict)
        else:
            logging.warning("Could not find model state_dict in checkpoint.")

        self.model.to(self.device)
        self.model.eval()
        logging.info(f"DepthProcessorNVFS initialized on device: {self.device}")

    def load_intrinsics(self, intrinsic_file, extrinsic_file):
        # Load intrinsics
        try:
            with open(intrinsic_file, 'r') as f:  # Open intrinsic file, units in mm
                lines = f.readlines()
                if len(lines) < 3:
                    raise ValueError(f"Intrinsic file {intrinsic_file} must contain at least 3 lines.")
            intrinsic_matrix = np.array([list(map(float, line.strip().split()[:3])) for line in lines], dtype=np.float32)
            # Construct K (3x3)
            self.K = np.zeros((3, 3), dtype=np.float32)
            # self.K in pixel units
            self.K[0, 0] = intrinsic_matrix[0, 0]  # fx
            self.K[1, 1] = intrinsic_matrix[1, 1]  # fy
            self.K[0, 2] = intrinsic_matrix[0, 2]  # cx
            self.K[1, 2] = intrinsic_matrix[1, 2]  # cy
            self.K[2, 2] = 1.0

            # Load extrinsics
            with open(extrinsic_file, 'r') as f:
                lines = f.readlines()
            lines = [line for line in lines if len(line.strip().split()) >= 4]
            if len(lines) < 3:
                raise ValueError(f"Extrinsic file {extrinsic_file} must contain at least 3 valid lines.")
            # Parse Rotation (R) and Translation (t)
            R = np.array([list(map(float, line.strip().split()[:3])) for line in lines[:3]], dtype=np.float32)  # (3x3)
            t = np.array([float(line.strip().split()[3]) for line in lines[:3]], dtype=np.float32)

            if self.metric_unit == 'm': t = t / 1000.0  # Convert mm → meters

            baseline = np.linalg.norm(t)  # Baseline in meters

        except Exception as e:
            raise IOError(f"Error reading intrinsic or extrinsic files: {e}")

        # Intrinsic scaling if needed
        self.K_scaled = self.K.copy()
        self.K_scaled[:2] *= self.scale  # Scale intrinsics if you downscale images

        self.baseline = baseline  # Save baseline too if needed later

        return self.K_scaled, baseline
    
    def global_crop_black_borders(self, left_img, right_img, threshold=5):
        def get_bounds(img):
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
            _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            coords = cv2.findNonZero(thresh)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                return x, y, x + w, y + h
            else:
                return 0, 0, img.shape[1], img.shape[0]

        x0_left, y0_left, x1_left, y1_left = get_bounds(left_img)
        x0_right, y0_right, x1_right, y1_right = get_bounds(right_img)

        # Global crop: take the worst-case crop for each side
        crop_margin = 15  # pixels to pad away from detected bounds
        x0_global = max(x0_left, x0_right) + crop_margin
        x1_global = min(x1_left, x1_right) - crop_margin
        x0_global = np.clip(x0_global, 0, left_img.shape[1]-1)
        x1_global = np.clip(x1_global, 0, left_img.shape[1]-1)

        y0_global = max(y0_left, y0_right)
        y1_global = min(y1_left, y1_right)
        
        # import matplotlib.pyplot as plt
        # plt.imshow(left_img[y0_global:y1_global, x0_global:x1_global])
        # plt.title("Cropped Left")
        # plt.show()
        left_cropped = left_img[y0_global:y1_global, x0_global:x1_global]
        right_cropped = right_img[y0_global:y1_global, x0_global:x1_global]

        return left_cropped, right_cropped, (x0_global, y0_global, x1_global, y1_global)

    @torch.no_grad() # Ensure no gradients are computed during inference
    def process_frame(self, 
                      left_img, right_img, fname, 
                      re_crop=True, cropvals=None, savecrop=False, img_dir=None, 
                      intrinsic_file=None, 
                      return_point_cloud=False):
        """
        Processes a pair of stereo images to compute depth and optionally a point cloud.

        Args:
            left_img (np.ndarray): Left image (H, W, C) in RGB format (uint8 or float).
            right_img (np.ndarray): Right image (H, W, C) in RGB format (uint8 or float).
            return_point_cloud (bool): Whether to compute and return the point cloud. Default: False.

        Returns:
            tuple: (depth_map, point_cloud)
                - depth_map (np.ndarray): Computed depth map (H_scaled, W_scaled) in meters(TODO CHECK MINMAX PRBLM).
                - point_cloud (o3d.geometry.PointCloud or None): Computed and optionally
                  denoised point cloud, or None if return_point_cloud is False.
        """
        torch.cuda.empty_cache() # Clear cache before processing

        if left_img.shape[:2] != right_img.shape[:2] or left_img.shape[2] != 3:
            raise ValueError("Left and right images must have the same H, W and 3 channels (RGB).")

        H_orig, W_orig = left_img.shape[:2]

        # --- Preprocessing ---
        # crop any black borders
        # Assume left_img and right_img loaded, intrinsic loaded as K

        # Crop black borders jointl
        if re_crop:
            if cropvals is None:
                left_img, right_img, cropvals = self.global_crop_black_borders(left_img, right_img)
                x_crop, y_crop, x1_crop, y1_crop = cropvals
            else:
                x_crop, y_crop, x1_crop, y1_crop = cropvals 
                left_img = left_img[y_crop:y1_crop, x_crop:x1_crop]
                right_img = right_img[y_crop:y1_crop, x_crop:x1_crop]

            # # Display cropped images (debugging)         
            import matplotlib.pyplot as plt
            import time
            if logging.getLogger().level <= logging.INFO:
                fig, axs = plt.subplots(2, 2, figsize=(12, 8))
                axs[0, 0].imshow(left_img)
                axs[0, 0].set_title("Original Left Image")
                axs[0, 1].imshow(right_img)
                axs[0, 1].set_title("Original Right Image")
                axs[1, 0].imshow(left_img)
                axs[1, 0].set_title(f"Cropped Left Image (x={x_crop}, y={y_crop})")
                axs[1, 1].imshow(right_img)
                axs[1, 1].set_title("Cropped Right Image")
                plt.tight_layout()
                # plt.savefig(os.path.join(os.path.dirname(os.path.dirname(self.args.out_dir)), 
                                        #  f"crop_comparison_{time.time()}.png"))
                plt.show()
                plt.close()
                logging.info(f"Cropped images: left={left_img.shape}, right={right_img.shape}")

            self.K_scaled[0, 2] -= x_crop
            self.K_scaled[1, 2] -= y_crop

            # Save cropped images & intrinsics if requested
            if savecrop:

                # Save the updated intrinsic matrix to a new file
                new_intrinsic_file = os.path.join(os.path.dirname(intrinsic_file), 'intrinsics_crop.txt')
                if not os.path.exists(new_intrinsic_file):
                    with open(new_intrinsic_file, 'w') as f:
                        # Write the intrinsic matrix in the same format as the original file
                        # First row: fx 0 cx
                        f.write(f"{self.K_scaled[0, 0]} 0 {self.K_scaled[0, 2]}\n")
                        # Second row: 0 fy cy
                        f.write(f"0 {self.K_scaled[1, 1]} {self.K_scaled[1, 2]}\n")
                        # Third row: 0 0 1
                        f.write("0 0 1\n")
                    logging.info(f"Saved updated intrinsic matrix to {new_intrinsic_file}")
                else:
                    logging.info(f"Intrinsic file {new_intrinsic_file} already exists, skipping txt save.")

                # Create output directories for cropped images
                crop_dir_left = os.path.join(img_dir, 
                                            'image01_crop')
                crop_dir_right = os.path.join(img_dir, 
                                            'image02_crop')
                
                # Create directories if they don't exist
                os.makedirs(crop_dir_left, exist_ok=True)
                os.makedirs(crop_dir_right, exist_ok=True)
                
                # Save cropped images with proper naming
                
                left_crop_file = os.path.join(crop_dir_left, fname.split('.')[0] + '.png')
                right_crop_file = os.path.join(crop_dir_right, fname.split('.')[0] + '.png')

                # Save as PNG files
                cv2.imwrite(left_crop_file, cv2.cvtColor(left_img, cv2.COLOR_RGB2BGR))
                cv2.imwrite(right_crop_file, cv2.cvtColor(right_img, cv2.COLOR_RGB2BGR))
                logging.info(f"Saved cropped images to {left_crop_file} and {right_crop_file}")

        # Convert to float32 if uint8
        img0_proc = left_img.astype(np.float32) if left_img.dtype == np.uint8 else left_img.copy()
        img1_proc = right_img.astype(np.float32) if right_img.dtype == np.uint8 else right_img.copy()

        if self.scale < 1.0: # resize images 
            new_H, new_W = int(H_orig * self.scale), int(W_orig * self.scale)
            # Use cv2.INTER_LINEAR for resizing float images
            img0_proc = cv2.resize(img0_proc, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
            img1_proc = cv2.resize(img1_proc, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
        # else: img0_proc, img1_proc remain the original float versions

        H, W = img0_proc.shape[:2]
        # Convert to torch tensor, move to device, add batch dim, set channel first (B, C, H, W)
        img0_torch = torch.from_numpy(img0_proc).to(self.device).permute(2, 0, 1).unsqueeze(0)
        img1_torch = torch.from_numpy(img1_proc).to(self.device).permute(2, 0, 1).unsqueeze(0)

        padder = InputPadder(img0_torch.shape, divis_by=32, force_square=False)
        img0_padded, img1_padded = padder.pad(img0_torch, img1_torch)

        # --- Inference ---
        try:
            with autocast(enabled=(self.mixed_precision)):  
                if not self.hiera:
                    disp_padded = self.model.forward(img0_padded, img1_padded, iters=self.valid_iters, test_mode=True)
                else:
                    # Ensure run_hierachical exists and handles device placement correctly
                    disp_padded = self.model.run_hierachical(img0_padded, img1_padded, iters=self.valid_iters, test_mode=True, small_ratio=0.5)
        except Exception as e:
            logging.error(f"Model inference failed: {e}")
            raise

        disp = padder.unpad(disp_padded.float())
        # Squeeze removes batch and channel dims (if they are 1), move to CPU, convert to numpy
        disp = disp.squeeze().cpu().numpy()

        # --- Post-processing ---
        # Ensure disparity is positive before calculating depth
        disp = np.maximum(disp, 1e-6) # Set minimum disparity to avoid division by zero

        if self.remove_invisible:
            # Calculate corresponding pixel coordinates in the right image
            yy, xx = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing='ij')
            us_right = xx - disp
            # Mark points where the corresponding pixel is outside the right image bounds
            invalid_mask = (us_right < 0) | (us_right >= W)
            # Set disparity for invalid points to a value that results in zero depth later
            disp[invalid_mask] = 1e-6 # Effectively infinite depth, will be set to 0

        # Ensure K_scaled and baseline are valid
        if self.K_scaled[0, 0] <= 0 or self.baseline <= 0:
            raise ValueError("Invalid camera intrinsics (fx <= 0) or baseline (<= 0).")
        # print(f'K_scaled: {self.K_scaled}, baseline: {self.baseline}')

        # Calculate depth: depth = baseline * fx / disparity
        depth = (self.K_scaled[0, 0] * self.baseline) / disp
        # Clamp depth values based on z_far and handle invalid disparities
        depth[disp <= 1e-6] = 0 # Set depth to 0 for invalid/zero disparity
        depth[depth > self.z_far] = 0 # Clip depth at z_far, setting far points to 0 depth
        depth[depth < 0] = 0 # Ensure depth is non-negative
        # Print minimum and maximum depth in given unit [mm]
        try:
            min_depth = np.min(depth[depth > 0])
            max_depth = np.max(depth)
            logging.info(f"Depth range: min = {min_depth:.2f} {self.metric_unit}, max = {max_depth:.2f} {self.metric_unit}")
        except ValueError:
            logging.warning("No valid depth values found after processing. Depth map may be empty.")
            return None, None, None
        # --- Point Cloud Generation (Optional) ---
        pcd = None
        if return_point_cloud: # generate a filtered point cloud
            # Use the scaled color image (img0_proc) for coloring the point cloud
            # Ensure depth map and color image have the same H, W
            if depth.shape != (H, W):
                 # This shouldn't happen if padding/unpadding is correct
                 logging.warning(f"Depth map shape {depth.shape} differs from scaled image shape {(H,W)}. Skipping point cloud.")
            else:
                xyz_map = depth2xyzmap(depth, self.K_scaled)
                # Filter points based on valid depth (depth > 0) before creating cloud
                valid_depth_mask = (depth > 1e-6).reshape(-1) # Flatten mask
                points = xyz_map.reshape(-1, 3)[valid_depth_mask]
                colors = img0_proc.reshape(-1, 3)[valid_depth_mask]

                if points.shape[0] > 0: # Check if there are any valid points
                    pcd = toOpen3dCloud(points, colors / 255.0 if colors.max() > 1.0 else colors) # Normalize color if needed

                    # Denoise (Optional)
                    if self.denoise_cloud and len(pcd.points) > self.denoise_nb_points : # Check if enough points for denoising
                        try:
                            # Use remove_radius_outlier which returns the cleaned cloud and indices
                            denoised_pcd, ind = pcd.remove_radius_outlier(nb_points=self.denoise_nb_points,
                                                                          radius=self.denoise_radius)
                            if len(denoised_pcd.points) > 0:
                                pcd = denoised_pcd
                                logging.debug(f"Point cloud denoised. Kept {len(ind)/len(pcd.points)*100}% points.")
                            else:
                                logging.warning("Denoising removed all points. Returning original filtered cloud.")
                        except Exception as e:
                            logging.warning(f"Point cloud denoising failed: {e}. Returning original filtered cloud.")
                else:
                     logging.warning("No valid points found for point cloud generation after depth filtering.")

        if re_crop:
            return depth, pcd, cropvals
        else:
            return depth, pcd, None

# --- Example Usage ---
if __name__ == "__main__":
    # --- Configuration ---
    # Guess project root assuming script is in scene3d/stereo_utils
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    FS_root = os.path.join(project_root, 'FoundationStereo')

    # Define default paths relative to the project root
    default_ckpt_path = os.path.join(FS_root, 'pretrained_models', '23-51-11', 'model_best_bp2.pth') #large model
    default_intrinsic_file = os.path.join('HAMLYN/hamlyn_data/calibration/01/intrinsics.txt')
    default_extrinsic_file = os.path.join('HAMLYN/hamlyn_data/calibration/01/extrinsics.txt')
    default_left_file = os.path.join('HAMLYN/hamlyn_data/rectified01/image01/0000000000.jpg')
    default_right_file = os.path.join('HAMLYN/hamlyn_data/rectified01/image02/0000000000.jpg')
    default_out_dir = os.path.join(script_dir, 'depth_results_tmp', 'depth_NVFS')

    parser = argparse.ArgumentParser(description="Framewise Depth Processing Utility using NVFS")
    parser.add_argument('--left_file', default=default_left_file, type=str, help='Path to the left image file.')
    parser.add_argument('--right_file', default=default_right_file, type=str, help='Path to the right image file.')
    parser.add_argument('--intrinsic_file', default=default_intrinsic_file, type=str, help='Path to the camera intrinsic matrix and baseline file.')
    parser.add_argument('--extrinsic_file', default=default_extrinsic_file, type=str, help='Path to the camera extrinsic matrix file.')
    parser.add_argument('--ckpt_path', default=default_ckpt_path, type=str, help='Path to the pretrained model checkpoint (.pth file).')
    parser.add_argument('--data_dir', default='HAMLYN/hamlyn_data', type=str, help='Base directory for input data.')
    parser.add_argument('--calibr_dir', default='HAMLYN/hamlyn_data/calibration', type=str, help='Base directory for calibration data.')
    parser.add_argument('--out_dir', default=default_out_dir, type=str, help='Directory to save output results.')
    parser.add_argument('--metric_unit', default='mm', choices=['m', 'mm'], help='Output depth unit: meters (m) or millimeters (mm).')
    parser.add_argument('--z_far', default=300.0, type=float, help='Maximum depth for point cloud clipping (metric unit).')
    parser.add_argument('--scale', default=1.0, type=float, help='Downsize factor for input images (<=1).')
    parser.add_argument('--hiera', action='store_true', help='Use hierarchical inference.')
    parser.add_argument('--valid_iters', type=int, default=32, help='Number of flow-field updates during inference.')
    parser.add_argument('--no_remove_invisible', action='store_true', help='Do not remove non-overlapping points.')
    parser.add_argument('--no_denoise_cloud', action='store_true', help='Disable point cloud denoising.')
    parser.add_argument('--denoise_nb_points', type=int, default=250, help='Number of neighbors for radius outlier removal.')
    parser.add_argument('--denoise_radius', type=float, default=3, help='Radius for outlier removal (metric units).')
    parser.add_argument('--no_save_output', action='store_true', help='Do not save output files (depth, cloud, vis).')
    parser.add_argument('--no_visualize', action='store_true', help='Do not visualize the point cloud.')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device to use for inference.')
    args = parser.parse_args()

    # Setup logging and seed
    set_logging_format()
    set_seed(0)
    os.makedirs(args.out_dir, exist_ok=True)
    logging.info(f"Default Output directory: {args.out_dir}")

    # Define the base directories from arguments
    # dataset_ids = ['08', '01', '04', '05'] # List of dataset IDs to process
    # dataset_ids = ['24', '26', '27'] # List of dataset IDs to process
    # dataset_ids = ['01', '04', '05', '06', '08', '09', '11', '12', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27'] # List of dataset IDs to process
    dataset_ids = ['01'] # List of dataset IDs to process
    # base_data_dir = args.data_dir
    # calibration_dir = args.calibr_dir

    base_data_dir = 'scene3d/acq_data/04_08_2025/bypass' # Base directory for input data
    calibration_dir = 'scene3d/stereo_utils/calibration/calibr_results/05_16_2025/25mmx60pct' # Base directory for calibration data
    rectify = True # if rectification is needed on images

    start_frame = 0 # Start frame for processing
    max_frames = 5000 # Limit to N frames for testing

    logging.info(f"Processing dataset IDs: {dataset_ids}")
    logging.info(f"Using base data directory: {base_data_dir}")
    logging.info(f"Using calibration directory: {calibration_dir}")

    # --------- Process Multiple Datasets ---------
    for id in dataset_ids:
        logging.info(f"--- Starting processing for dataset ID: {id} ---")

        # Construct paths dynamically based on the current ID
        if os.path.isdir(os.path.join(base_data_dir, f'rectified{id}')):
            rectified_data_path = os.path.join(base_data_dir, f'rectified{id}')
        else:
            rectified_data_path = base_data_dir
        left_dir = os.path.join(rectified_data_path, 'image01')
        right_dir = os.path.join(rectified_data_path, 'image02')
        current_out_dir = os.path.join(rectified_data_path, 'NVFS_crop') # Use a specific output dir per dataset
        current_intrinsic_file = os.path.join(calibration_dir, id, 'intrinsics.txt')
        current_extrinsic_file = os.path.join(calibration_dir, id, 'extrinsics.txt')

        # Update args for the current dataset (optional, but can be useful if processor reads from args)
        args.left_dir = left_dir # Store for potential later use, though the loop uses left_dir directly
        args.right_dir = right_dir
        args.out_dir = current_out_dir # Set the specific output dir for this dataset
        args.intrinsic_file = current_intrinsic_file
        args.extrinsic_file = current_extrinsic_file

        # Check if required directories and files exist for this ID
        if not os.path.isdir(left_dir):
            logging.warning(f"Left image directory not found for ID {id}: {left_dir}. Skipping.")
            continue
        if not os.path.isdir(right_dir):
            logging.warning(f"Right image directory not found for ID {id}: {right_dir}. Skipping.")
            continue
        if not os.path.isfile(current_intrinsic_file):
            logging.warning(f"Intrinsic file not found for ID {id}: {current_intrinsic_file}. Skipping.")
            continue
        if not os.path.isfile(current_extrinsic_file):
            logging.warning(f"Extrinsic file not found for ID {id}: {current_extrinsic_file}. Skipping.")
            continue

        os.makedirs(args.out_dir, exist_ok=True) # Create output directory for the current dataset
        logging.info(f"Dataset {id}: Input Left Dir: {left_dir}")
        logging.info(f"Dataset {id}: Input Right Dir: {right_dir}")
        logging.info(f"Dataset {id}: Output Dir: {args.out_dir}")
        logging.info(f"Dataset {id}: Intrinsics: {args.intrinsic_file}")
        logging.info(f"Dataset {id}: Extrinsics: {args.extrinsic_file}")


        # --- Initialize Processor ---
        try:
            depth_processor = DepthProcessorNVFS(
                ckpt_dir=args.ckpt_path,
                intrinsic_file=args.intrinsic_file,    # processor is specific to intrinsics files
                extrinsic_file=args.extrinsic_file,
                metric_unit=args.metric_unit,
                scale=args.scale,
                hiera=args.hiera,
                valid_iters=args.valid_iters,
                remove_invisible=not args.no_remove_invisible,
                z_far=args.z_far,
                denoise_cloud=not args.no_denoise_cloud,
                denoise_nb_points=args.denoise_nb_points,
                denoise_radius=args.denoise_radius,
                device=args.device
            )
        except FileNotFoundError as e:
            logging.error(f"Initialization failed: {e}")
            sys.exit(1)
        except ImportError as e:
            logging.error(f"Import error during initialization: {e}. Check dependencies and paths.")
            sys.exit(1)
        except Exception as e:
            logging.error(f"An unexpected error occurred during initialization: {e}")
            traceback.print_exc()
            sys.exit(1)

        # --- Process Multiple Images in Dataset Folder ---
        left_files = sorted([
            f for f in os.listdir(left_dir)
            if os.path.isfile(os.path.join(left_dir, f))
        ])
        right_files = sorted([
            f for f in os.listdir(right_dir)
            if os.path.isfile(os.path.join(right_dir, f))
        ])

        # max_frames = len(left_files) # Limit to N frames for testing
        # max_frames = len(left_files) # Limit to N frames

        cropvals = None # Initialize cropvals for the first frame
        for i in range(min(max_frames, len(left_files))):
            if i < start_frame:
                logging.info(f"Skipping frame {i} (start_frame={start_frame})")
                continue
            args.left_file = os.path.join(left_dir, left_files[i])
            args.right_file = os.path.join(right_dir, right_files[i])
            depth_path_png = os.path.join(args.out_dir, left_files[i].split('.')[0] + '.png') # Save as #######.png
            # Check if depth PNG already exists
            if os.path.exists(depth_path_png):
                logging.info(f"Depth PNG already exists at {depth_path_png}. Skipping this frame.")
                continue

            # --- Load & Process Images ---
            try:
                img_left = iio.imread(args.left_file)
                img_right = iio.imread(args.right_file)
                logging.info(f"Loaded images: {args.left_file}, {args.right_file} with shape {img_left.shape}")


                # Rectification
                if rectify:
                    from stereo_utils import rectify_images
                    img_right, img_left, Q, T = rectify_images(img_left, img_right, calibration_dir, visualize=False)
                    # TODO need to adjust folder names in 04_08 run to correct imgleft & right
                
                
                # Ensure images are RGB (3 channels)
                if img_left.ndim == 2: # Grayscale
                    img_left = cv2.cvtColor(img_left, cv2.COLOR_GRAY2RGB)
                    img_right = cv2.cvtColor(img_right, cv2.COLOR_GRAY2RGB)
                    logging.info("Converted grayscale images to RGB.")
                elif img_left.shape[2] == 4: # RGBA
                    img_left = cv2.cvtColor(img_left, cv2.COLOR_RGBA2RGB)
                    img_right = cv2.cvtColor(img_right, cv2.COLOR_RGBA2RGB)
                    logging.info("Converted RGBA images to RGB.")
                elif img_left.shape[2] != 3:
                    raise ValueError(f"Input images must have 3 channels (RGB), but got {img_left.shape[2]}")

            except FileNotFoundError:
                logging.error(f"Could not load input images: {args.left_file}, {args.right_file}")
                sys.exit(1)
            except Exception as e:
                logging.error(f"Error reading or processing images: {e}")
                sys.exit(1)

            # --- Process Frame ---
            logging.info("Processing frame...")
            try:
                fname = left_files[i]
                depth_map, point_cloud, cropvals = depth_processor.process_frame(
                    img_left, img_right, fname,
                    intrinsic_file=current_intrinsic_file, cropvals=cropvals, 
                    savecrop=True, img_dir=rectified_data_path,
                    return_point_cloud=False
                )
                logging.info("Frame processing complete.")
                if depth_map is not None:
                    logging.info(f"Generated depth map with shape: {depth_map.shape}")
                if point_cloud is not None:
                    logging.info(f"Generated point cloud with {len(point_cloud.points)} points.")
                else:
                    logging.info("Point cloud generation skipped or resulted in no points.")

            except Exception as e:
                logging.error(f"Error during frame processing: {e}")
                traceback.print_exc()
                sys.exit(1)

            # --- Save and Visualize Results (Optional) ---
            if not args.no_save_output:
                # Save depth map (e.g., as numpy array)
                if depth_map is not None:
                    # depth_path_npy = os.path.join(args.out_dir, left_files[i].split('.')[0] + '.npy')
                    # np.save(depth_path_npy, depth_map)
                    # logging.info(f"Depth map saved to {depth_path_npy}")

                    # Save depth map as a grayscale image (here for training use - no scale adj)
                    depth_vis = depth_map.copy()
                    # valid depth range (0 < depth <= z_far) to [0, 255]
                    valid_mask = (depth_vis > 0) & (depth_vis <= 255) # will cap at 255mm
                    if np.any(valid_mask):
                        min_d, max_d = depth_vis[valid_mask].min(), depth_vis[valid_mask].max()
                        print(f"Depth visualization range: min={min_d}, max={max_d}")
                        # depth_vis[valid_mask] = (depth_vis[valid_mask] - min_d) / (max_d - min_d + 1e-6) * 255.0
                    depth_vis[~valid_mask] = 0 # Set invalid depth (0) to black
                    depth_vis_img = depth_vis.astype(np.uint8)
                    iio.imwrite(depth_path_png, depth_vis_img)
                    logging.info(f"Depth map visualization saved to {depth_path_png}")

                # Save point cloud
                if point_cloud is not None and len(point_cloud.points) > 0:
                    cloud_path = os.path.join(args.out_dir, left_files[i].split('.')[0], 'point_cloud.ply')
                    o3d.io.write_point_cloud(cloud_path, point_cloud, write_ascii=False)
                    logging.info(f"Point cloud saved to {cloud_path}")
                else:
                    logging.info("Point cloud is empty or not generated, skipping save.")

            # Visualize point cloud
            if point_cloud is not None and len(point_cloud.points) > 0 and not args.no_visualize:
                logging.info("Visualizing point cloud. Press 'Q' or 'ESC' to exit.")
                try:
                    o3d.visualization.draw_geometries([point_cloud],
                                                    window_name="NVFS Point Cloud",
                                                    point_show_normal=False,
                                                    width=800, height=600)

                    # Alternative manual visualizer if more control is needed:
                    # vis = o3d.visualization.Visualizer()
                    # vis.create_window(window_name="NVFS Point Cloud", width=800, height=600)
                    # vis.add_geometry(point_cloud)
                    # opt = vis.get_render_option()
                    # opt.point_size = 1.0
                    # opt.background_color = np.asarray([0.1, 0.1, 0.1])
                    # vis.run()
                    # vis.destroy_window()
                except Exception as e:
                    logging.error(f"Open3D visualization failed: {e}")
        
        logging.info(f"Processed {len(left_files)} frames in dataset {id}.")

    logging.info("Processing finished.")