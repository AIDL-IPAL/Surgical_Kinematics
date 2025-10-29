import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
import sys
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.cuda import amp
from torch.utils.data import Dataset, DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import argparse
from pathlib import Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.depth_visualization import comparison_vis, make_movie, tile_output, visualize_diff, visualize_depth
from utils.dataprocess import DepthDataset, postprocess_depth, metric3d_prep_img, metric3d_unpad_and_scale
from model_training.utils.hamlyn_intrinsics import read_intrinsics
from utils.err_metrics import get_error_metrics

def load_model(framework:str, encoder:str, repo:str, ckpt:str):
    """Return (model, preprocess_flag) where preprocess_flag==True
       if input must be run through metric3d_prep_img / unpad."""
    if framework.startswith('depth_anything'):
        sys.path.append(os.path.abspath('Depth-Anything-V2'))
        from depth_anything_v2.dpt import DepthAnythingV2
        cfgs = {'vits':{'encoder':'vits','features':64,'out_channels':[48,96,192,384]},
                'vitb':{'encoder':'vitb','features':128,'out_channels':[96,192,384,768]},
                'vitl':{'encoder':'vitl','features':256,'out_channels':[256,512,1024,1024]}}
        model = DepthAnythingV2(**cfgs[encoder])
        model.load_state_dict(torch.load(f'Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder}.pth',
                         map_location=device, weights_only=True))
        return model.to(device).eval(), False
    # ---------------- Metric3D ----------------
    model = torch.hub.load(repo, framework, pretrain=(ckpt is None))
    if ckpt:
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True), strict=False)
        return model.to(device).eval(), True

def metric3d_inference(model, image_t, og_shape, pad_info, intrinsic, scale=None):
    image_t = image_t.to(device)
    model.eval()

    with torch.no_grad():
        # with amp.autocast():
        pred_depth, confidence, output_dict = model({'input': image_t})

    if torch.isnan(pred_depth).any():
        print(f"NaN values in predicted depth.")
        return None

    pred_depth = metric3d_unpad_and_scale(pred_depth, intrinsic, pad_info, og_shape)

    if scale is not None:
        pred_depth = postprocess_depth(pred_depth, scale=scale)
    else:
        pred_depth = postprocess_depth(pred_depth, scale=1.0)

    pred_depth = pred_depth.squeeze(1) # squeeze color channel dim
    return pred_depth

def hamlyn_inference_main():
    parser = argparse.ArgumentParser(description='Depth Inference Script with Visualization')
    parser.add_argument('--start_frame', type=int, default=0, help='Start frame for inference')
    parser.add_argument('--end_frame', type=int, default=250, help='End frame for inference')
    parser.add_argument('--device', type=str, default=device, help='Device to run inference on (e.g., cuda, cpu)')
    parser.add_argument('--repo', type=str, default='yvanyin/metric3d', help='Repository for the Metric3D model')
    parser.add_argument('--model_name', type=str, default='metric3d', help='Model name for Metric3D')
    parser.add_argument('--encoder', type=str, default='vitb', help='Encoder type (e.g., vitb, vits, vitl)')
    parser.add_argument('--model_ckpt', type=str, default=None, help='Path to the fine-tuned Metric3D checkpoint')
    parser.add_argument('--global_scale', type=float, default=None, help='Global scale factor for depth map')
    parser.add_argument('--data_dir', type=str, default='HAMLYN/hamlyn_data', help='Path to the data directory')
    parser.add_argument('--img_dir', type=str, default='data/images', help='Path to RGB images directory')
    parser.add_argument('--depth_dir', type=str, default='data/depth', help='Path to main depth directory')
    parser.add_argument('--intrinsics_file', type=str, default='data/intrinsics.txt', help='Path to intrinsic parameters file')
    parser.add_argument('--depth_dir02', type=str, default=None, help='Optional path to second depth directory')
    parser.add_argument('--output_dir', type=str, default='inference_output', help='Where to save result images')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference') # Currently only 1 supported
    args = parser.parse_args()

    if args.global_scale is not None:
        if args.global_scale <= 0:
            args.global_scale = None
    
    intrinsic = read_intrinsics(args.intrinsics_file) # [fx​,fy​,cx​,cy​]

    # 1) Load Model
    model, needs_metric3d_post = load_model(args.model_name, args.encoder, args.repo, args.model_ckpt)
    model.to(device).eval()

    # 2) Create Dataset & Dataloader
    dataset = DepthDataset(args.img_dir, intrinsic, args.depth_dir, args.start_frame, args.end_frame)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # 3) Run Inference & Visualization
    # Set up report for metrics as a DataFrame
    report = pd.DataFrame(columns=['frame', 'MSE', 'MAE', 'MAPE', 'R2'])
    os.makedirs(args.output_dir, exist_ok=True)

    # scale factor convergence for depth map
    num_accum  = torch.zeros(1, device=device)   # ∑ g·p
    den_accum  = torch.zeros(1, device=device)   # ∑ p²
    compare = None

    for i, batch in enumerate(tqdm(dataloader, desc='Progress', total=args.end_frame - args.start_frame)):
        rgb, depth_gt, mask, rgb_origin, img_name, pad_info, intrinsic, original_shapes = batch
        # print(f"Processing frame {img_name[0]}")
        og_shape = depth_gt.shape[1:]
        rgb = rgb.to(device)
        depth_gt = depth_gt.to(device)
        mask = mask.to(device)

        pad_info = [p for p in pad_info]  # Convert to list of 4-element tensors
        intrinsic = [k for k in intrinsic]

        # image_og = image_rgb.clone()
        if rgb.numel() == 0 or torch.all(mask):
            print(f"Skipping frame due to empty image / mask covering the entire image.")
            continue

        if args.model_name == 'depth_anything':
            # image_rgb = image_rgb.to(device)
            image_rgb = rgb_origin.squeeze(0).squeeze(0)  # Remove batch dimension
            image_rgb = image_rgb.cpu().numpy()  # Convert from torch tensor to numpy array (H, W, C)
            # print(f"shape of image_rgb: {image_rgb.shape}")
            pred_depth = model.infer_image(image_rgb)  # Returns HxW raw depth map
            pred_depth = torch.from_numpy(pred_depth).to(device)  # Convert back to tensor for processing
            pred_depth = pred_depth.unsqueeze(0)#.unsqueeze(0)  # Add batch and channel dimensions
            if torch.isnan(pred_depth).any():
                print(f"Skipping frame {img_name[0]} due to NaN values in predicted depth.")
                continue
            if args.global_scale is not None:
                pred_depth = postprocess_depth(pred_depth, scale=args.global_scale)
            else:
                pred_depth = postprocess_depth(pred_depth, scale=1.0)

        else: # Metric3d
            pred_depth = metric3d_inference(model, rgb, og_shape, pad_info, intrinsic, scale=args.global_scale)
            rgb_s = rgb[0].detach().cpu().numpy()
            from utils.dataprocess import denormalize_rgb
            rgb_sample, rgb_raw = denormalize_rgb(rgb_s)
        
        # print(f"shapes - pred_depth: {pred_depth.shape}, depth_gt: {depth_gt.shape}, mask: {mask.shape}, rgb_origin: {rgb_origin.shape}")
        diff = pred_depth[mask] - depth_gt[mask]
        m          = mask.bool()
        num_accum += torch.sum(depth_gt[m] * pred_depth[m])
        den_accum += torch.sum(pred_depth[m] ** 2)
        pred_depth[~m] = 0.0

        # Get error metrics and visualize
        # mean_depth = torch.mean(depth_gt[mask])
        # mean_pred_depth = torch.mean(pred_depth[mask])
        errors = get_error_metrics(depth_gt, pred_depth, diff, mask)
        # print(f"Errors for frame {img_name[0]}: {{'MSE': {errors['MSE']:.5g}, 'MAE': {errors['MAE']:.5g}, 'MAPE': {errors['MAPE']:.5g}, 'R2': {errors['R2']:.5g}}}")
        # img_name = os.path.basename(dataset.img_list[i])
        compare, _, _, _, _, _ = comparison_vis(
            compare, rgb_sample, 
            depth_gt, pred_depth, diff, mask, 
            errors, img_name, output_dir=args.output_dir, show=False
            )
        
        # Save the metrics to the report DataFrame
        new_row = pd.DataFrame({'frame': [img_name[0]], 'MSE': [errors['MSE']], 'MAE': [errors['MAE']], 'MAPE': [errors['MAPE']], 'R2': [errors['R2']]})
        if not new_row.isnull().all().all() and not new_row.empty:  # Exclude rows that are all-NA or empty
            report = pd.concat([report, new_row], ignore_index=True)

    # Save the report as a CSV file
    report_path = os.path.join(args.output_dir, 'metrics_report.csv')
    report.to_csv(report_path, index=False)
    print(f'Metrics report saved to {report_path}')

    # 4) Optionally, make a movie out of the tiled images
    images_output_dir = os.path.join(args.output_dir, 'processed_tiles')
    make_movie(images_output_dir, args.output_dir, video_name='output_video_test')
    print(f'Processing complete. Results saved to {args.output_dir}')

    avg_metrics = report[['MSE', 'MAE', 'MAPE', 'R2']].mean().to_dict()
    print(f'Average metrics from frame {args.start_frame} to {args.end_frame}:')
    print(f'Global scale: {args.global_scale}')
    for key, value in avg_metrics.items():
        print(f'{key}: {value:.4g}')

    return avg_metrics, num_accum, den_accum


if __name__ == '__main__':
    
    global_scale = 0.05 # can be None for no scaling calibration or real number (model training with 0.05)
    print(f"Global scale factor: {global_scale}")
    calculate_scale = True # if True, will re-calculate global scale factor from training set and rerun inference
    
    # video_path = 'JIGSAWS/Suturing/video'
    # box_annotations = os.listdir('JIGSAWS/Suturing/box_annotations') if os.path.exists('JIGSAWS/Suturing/box_annotations') else []
    # segment_annotations = os.listdir('JIGSAWS/Suturing/segment_annotations') if os.path.exists('JIGSAWS/Suturing/segment_annotations') else []
    # video_list = [f for f in os.listdir(video_path) if '.mp4' in f]
    # video_list = min([video_list, box_annotations, segment_annotations], key=len) if video_list and box_annotations and segment_annotations else video_list
    THIS_FILE = Path(__file__).resolve()             # …/your_script.py
    PROJECT_ROOT = THIS_FILE.parents[1]              # adjust depth as needed
    DATA_DIR = PROJECT_ROOT / "datasets" / "HAMLYN" / "hamlyn_data"
    OUTPUT_DIR = PROJECT_ROOT / "scene3d" / "results" / "Northwell_hamlyntrain_scaling"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # rectified_dirs = [d for d in os.listdir(args.data_dir) if d.startswith('rectified') and d.endswith('.zip')]

    # rectified_ids = ['01']
    # # rectified_ids = ['24', '26', '27']
    test_ids = ['01', '11',  '18', '26'] 
    # rectified_ids = ['08']
    rectified_folders = [
        (DATA_DIR / f'rectified{id}' / 'image01_crop', DATA_DIR / f'rectified{id}' / 'NVFS_crop', DATA_DIR / 'calibration' / id / 'intrinsics_crop.txt')
        for id in test_ids
    ]

    # rectified_folders = [
    #     ('scene3d/acq_data/04_08_2025/cholecystectomy/image01_crop', 
    #      'scene3d/acq_data/04_08_2025/cholecystectomy/NVFS_crop',
    #      'scene3d/stereo_utils/calibration/calibr_results/05_16_2025/25mmx60pct/01/intrinsics_crop.txt')
    # ]

    num_accumulated = torch.zeros(1, device=device)   # ∑ g·p
    den_accumulated = torch.zeros(1, device=device)   # ∑ p²
    for i, (img_dir, depth_dir, intrinsics_file) in enumerate(rectified_folders):
        if 'rectified_ids' in locals() or 'rectified_ids' in globals():
            output_dir_folder = os.path.join(f'{OUTPUT_DIR}', f'folder{test_ids[i]}')
        else:
            output_dir_folder = os.path.join(f'{OUTPUT_DIR}', f'folder{i}')
        # print(f"output_dir_folder: {output_dir_folder}")
        os.makedirs(output_dir_folder, exist_ok=True)
    
        sys.argv = [
            'metric3D_training_inferencetests.py',
            '--repo', 'yvanyin/metric3d',
            '--model_name', 'metric3d_vit_small', # 'metric3d_vit_small' or 'depth_anything'
            '--model_ckpt', 'scene3d/model_training/Metric3D_allhamlyn/model_ckpts/ckpt_20250513_epoch8.pth', # 'depth_anything_v2_vits.pth' or 'metric3d_vit_small.pth'
            '--encoder', 'vits', # only used in depth_anything. 'vits', 'vitb', 'vitl'
            '--start_frame', '0',
            '--end_frame', '750',
            '--global_scale', str(global_scale) if global_scale is not None else '0',
            '--img_dir', str(img_dir),
            '--depth_dir', str(depth_dir),
            '--intrinsics_file', str(intrinsics_file),
            '--output_dir', str(output_dir_folder),
            '--batch_size', '1'
        ]

        metrics, num_accum, den_accum = hamlyn_inference_main()

        num_accumulated += num_accum
        den_accumulated += den_accum
        if i == 0:
            metrics_df = pd.DataFrame(metrics, index=[0])
        else:
            metrics_df = pd.concat([metrics_df, pd.DataFrame(metrics, index=[0])], ignore_index=True)

    avg_metrics = metrics_df.mean().to_dict()
    print("Average metrics across test folders:")
    print(f"Global scale: {global_scale}")
    for key, value in avg_metrics.items():
        print(f"{key}: {value:.4f}")

    if global_scale is None or calculate_scale: # compute global scale
        if global_scale is None:
            global_scale = (num_accum / (den_accum + 1e-8)).item()
        else:
            global_scale = global_scale*(num_accum / (den_accum + 1e-8)).item()
        print(f"⇢ CALCULATED GLOBAL SCALE FACTOR: {global_scale:.4f}")
        # rerun inference with global scale USE TEST SET
        
        for i, (img_dir, depth_dir, intrinsics_file) in enumerate(rectified_folders):
            if 'rectified_ids' in locals() or 'rectified_ids' in globals():
                output_dir_folder = os.path.join(f'{OUTPUT_DIR}', f'folder{test_ids[i]}')
            else:
                output_dir_folder = os.path.join(f'{OUTPUT_DIR}', f'folder{i}')
            # print(f"output_dir_folder: {output_dir_folder}")
            os.makedirs(output_dir_folder, exist_ok=True)

            # Properly remove --global_scale and its value from sys.argv
            new_argv = []
            i = 0
            while i < len(sys.argv):
                if sys.argv[i] == '--global_scale' and i + 1 < len(sys.argv):
                    i += 2  # Skip both the flag and its value
                else:
                    new_argv.append(sys.argv[i])
                    i += 1
            sys.argv = new_argv
            sys.argv.extend(['--global_scale', str(global_scale)])
            # output_dir = 'scene3d/inference_testresults/depth_anything_v2_small_scaled'
            metrics, num_accum, den_accum = hamlyn_inference_main()

            num_accumulated += num_accum
            den_accumulated += den_accum
            if i == 0:
                metrics_df = pd.DataFrame(metrics, index=[0])
            else:
                metrics_df = pd.concat([metrics_df, pd.DataFrame(metrics, index=[0])], ignore_index=True)

        avg_metrics = metrics_df.mean().to_dict()
        print("Average metrics across test folders:")
        for key, value in avg_metrics.items():
            print(f"{key}: {value:.4f}")
