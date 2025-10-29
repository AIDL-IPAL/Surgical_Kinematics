import os, sys, cv2, argparse, torch, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

def visualize_depth(depth, mask=None, depth_min=None, depth_max=None):
    """
    Convert a floating depth map [H,W], with values ~[0..1], into an 8-bit grayscale image.
    We apply the mask if provided, setting invalid pixels to black.
    """
    depth_viz = depth.copy()
    if mask is not None:
        depth_viz[mask <= 0] = 0.0
        # Scale to 0..255 for display
        calc_depth_min = np.min(depth_viz[mask > 0])
        calc_depth_max = np.max(depth_viz[mask > 0])
        if depth_min is not None:
            depth_min = min(depth_min, calc_depth_min)
        else:
            depth_min = calc_depth_min
        if depth_max is not None:
            depth_max = max(depth_max, calc_depth_max)
        else:
            depth_max = calc_depth_max

    # if depth_min is None or depth_min < 15:
    #     depth_min = 15
    # if depth_max is None or depth_max > 75:
    #     depth_max = 150.0
    depth_min = 15
    depth_max = 150.0
    depth_viz = np.clip((depth_viz - depth_min) * (255 / (depth_max - depth_min)), 0, 255)
    depth_viz = depth_viz.astype(np.uint8)  # Ensure depth_viz is uint8
    depth_color = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)

    return depth_viz, depth_color, depth_min, depth_max


def visualize_diff(diff, mask):
    """
    Produce a color-coded difference map of absolute difference:
      diff = abs(gt_depth - pred_depth), masked
    The final map is color-coded with cv2.COLORMAP_JET.
    """
    # Apply mask to the difference map
    if mask.shape != diff.shape:
        raise ValueError("Mask and diff must have the same shape")
    valid_diff = diff[mask > 0]  # Use the mask to filter valid pixels
    diff_min, diff_max = np.min(valid_diff), np.max(valid_diff)

    # Normalize the difference map to the range [0, 255]
    diff_norm = (diff - diff_min) / (diff_max - diff_min + 1e-8)
    diff_norm[mask <= 0] = 0  # Set invalid pixels to 0
    diff_norm = (np.clip(diff_norm, 0, 1) * 255).astype(np.uint8)

    # Apply a color map to the normalized difference map
    diff_color = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)

    return diff_color, diff_norm, diff_min, diff_max


def tile_output(rgb, gt_depth_viz, pred_depth_viz, diff_color):
    """
    Make a 2x2 tile:
       top-left:   original color image
       top-right:  ground-truth depth (grayscale)
       bot-left:   predicted depth (grayscale)
       bot-right:  difference color map
    All must have same height if we want to vstack/hstack easily.
    We can do e.g. resize the color images to match grayscale shape or vice versa.
    """
    # Convert the original color image from [H,W,3] to 3-channel BGR or something for hconcat
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)

    # If the shapes differ, unify them:
    h, w = gt_depth_viz.shape[:2]
    color_h, color_w = rgb.shape[:2]
    if color_h != h or color_w != w:
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)

    pred_depth_viz_3c = pred_depth_viz if pred_depth_viz.ndim == 3 else cv2.cvtColor(pred_depth_viz, cv2.COLOR_GRAY2BGR)
    gt_depth_viz_3c = gt_depth_viz if gt_depth_viz.ndim == 3 else cv2.cvtColor(gt_depth_viz, cv2.COLOR_GRAY2BGR)

    row1 = np.hstack([rgb, gt_depth_viz_3c])
    # Resize diff_color to match the height of pred_depth_viz_3c
    if diff_color.shape[0] != pred_depth_viz_3c.shape[0]:
        diff_color = cv2.resize(diff_color, (diff_color.shape[1], pred_depth_viz_3c.shape[0]), interpolation=cv2.INTER_LINEAR)
    if gt_depth_viz_3c.shape[0] != pred_depth_viz_3c.shape[0]:
        gt_depth_viz_3c = cv2.resize(gt_depth_viz_3c, (gt_depth_viz_3c.shape[1], pred_depth_viz_3c.shape[0]), interpolation=cv2.INTER_LINEAR)

    row2 = np.hstack([pred_depth_viz_3c, diff_color])
    tiled = np.vstack([row1, row2])
    return tiled

def add_scaled_error_bar(frame_color, frame_min, frame_max):
    """
    Add a scaled error bar to the difference map.
    """
    bar_height = 20
    bar_width = frame_color.shape[1]
    error_bar = np.zeros((bar_height, bar_width, 3), dtype=np.uint8)

    for x in range(bar_width):
        value = frame_min + (frame_max - frame_min) * (x / bar_width)
        color = cv2.applyColorMap(np.array([[x * 255 // bar_width]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0]
        error_bar[:, x, :] = color

    # Add text labels for min and max values
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    text_color = (255, 255, 255)  # White text
    cv2.putText(error_bar, f'{frame_min:.2f}', (5, bar_height - 5), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    cv2.putText(error_bar, f'{frame_max:.2f}', (bar_width - 50, bar_height - 5), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    # Append the error bar to the difference map
    if diff_color.ndim == 2:  # If diff_color is 2D, expand it to 3D
        diff_color = np.expand_dims(diff_color, axis=-1)
        diff_color = np.repeat(diff_color, 3, axis=-1)  # Repeat along the color channels
    diff_color_with_bar = np.vstack([diff_color, error_bar])
    return diff_color_with_bar


def comparison_vis(compare, img_rgb, depth_gt_scaled, pred_depth_scaled, diff, mask, errors, fname, output_dir=None, show=False):
    if compare is not None:
        gt_min = compare["ground_truth"]["min"]
        gt_max = compare["ground_truth"]["max"]
        pred_min = compare["predicted"]["min"]
        pred_max = compare["predicted"]["max"]
        diff_min = compare["difference"]["min"]
        diff_max = compare["difference"]["max"]
    else:
        gt_min = None
        gt_max = None
        pred_min = None
        pred_max = None
        diff_min = None
        diff_max = None
        
    # Ensure the inputs are numpy arrays and handle batch size
    depth_gt_scaled = depth_gt_scaled.squeeze().cpu().numpy() if isinstance(depth_gt_scaled, torch.Tensor) else depth_gt_scaled
    pred_depth_scaled = pred_depth_scaled.squeeze().cpu().numpy() if isinstance(pred_depth_scaled, torch.Tensor) else pred_depth_scaled
    diff = diff.squeeze().cpu().numpy() if isinstance(diff, torch.Tensor) else diff
    mask = mask.squeeze().cpu().numpy() if isinstance(mask, torch.Tensor) else mask
    # print(f"img_rgb shape: {img_rgb.shape}, depth_gt_scaled shape: {depth_gt_scaled.shape}, pred_depth_scaled shape: {pred_depth_scaled.shape}, diff shape: {diff.shape}, mask shape: {mask.shape}")
    # Visualize the depth and difference maps
    gt_viz, gt_color, gt_min, gt_max = visualize_depth(depth_gt_scaled, mask, gt_min, gt_max)
    pred_mask = None
    pred_viz, pred_color, pred_min, pred_max = visualize_depth(pred_depth_scaled, pred_mask, pred_min, pred_max)

    # Reshape diff to match the shape of the mask for visualization
    diff_map = np.zeros_like(mask, dtype=np.float32)
    diff_map[mask > 0] = diff  # Assign the flattened diff values to valid mask locations
    diff_color, diff_norm, diff_min, diff_max = visualize_diff(diff_map, mask.astype(np.float32))

    diff_color_with_bar = add_scaled_error_bar(diff_color, diff_min, diff_max)

    # Add metrics as text to the diff_color image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    text_color = (255, 255, 255)  # White text
    bg_color = (0, 0, 0)  # Black background for text

    # Create a black rectangle for the text background
    text_bg = np.zeros((50, diff_color_with_bar.shape[1], 3), dtype=np.uint8)
    cv2.putText(text_bg, f'MSE: {errors["MAE"]:.4f}', (10, 20), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    cv2.putText(text_bg, f'MAPE: {errors["MAPE"]:.4f}', (10, 40), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    cv2.putText(text_bg, f'R2: {errors["R2"]:.4f}', (200, 20), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    # Append the text background to the diff_color_with_bar
    diff_color_with_bar = np.vstack([diff_color_with_bar, text_bg])

    gt_color_with_bar = add_scaled_error_bar(gt_color, 15, 100)
    pred_color_with_bar = add_scaled_error_bar(pred_color, 15, 100)

    # Make a 2x2 tile
    # Convert img_rgb from torch tensor to numpy array for visualization
    img_rgb = img_rgb.squeeze().cpu().numpy() if isinstance(img_rgb, torch.Tensor) else img_rgb
    tiled = tile_output(img_rgb, gt_color_with_bar, pred_color_with_bar, diff_color_with_bar)

    # Add labels to each tile
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    color = (255, 255, 255)  # White text
    bg_color = (0, 0, 0)  # Black background for text

    # Add labels to the tiles
    cv2.putText(tiled, 'Original RGB', (10, 30), font, font_scale, color, font_thickness, cv2.LINE_AA)
    cv2.putText(tiled, 'Stereo Depth Annotation', (tiled.shape[1] // 2 + 10, 30), font, font_scale, color, font_thickness, cv2.LINE_AA)
    cv2.putText(tiled, 'Predicted Depth', (10, tiled.shape[0] // 2 + 30), font, font_scale, color, font_thickness, cv2.LINE_AA)
    cv2.putText(tiled, 'Difference Map', (tiled.shape[1] // 2 + 10, tiled.shape[0] // 2 + 30), font, font_scale, color, font_thickness, cv2.LINE_AA)

    # Save the result
    if output_dir is not None:
        # Save or display
        images_output_dir = os.path.join(output_dir, 'processed_tiles')
        os.makedirs(images_output_dir, exist_ok=True)
        out_name = os.path.splitext(fname[0])[0] + '_compare.png'
        out_path = os.path.join(images_output_dir, out_name)
        cv2.imwrite(out_path, cv2.cvtColor(tiled, cv2.COLOR_RGB2BGR))  # convert back to BGR for saving
    # Optionally display or just continue
    if show:
        cv2.imshow('Comparison', tiled[..., ::-1]) # if you want a pop-up
        cv2.waitKey(1)
    def clamp(value, min_value=0.0, max_value=250.0):
        return max(min_value, min(float(value), max_value))

    new_compare = {
        "ground_truth": {"min": clamp(gt_min), "max": clamp(gt_max)},
        "predicted": {"min": clamp(pred_min), "max": clamp(pred_max)},
        "difference": {"min": clamp(diff_min), "max": clamp(diff_max)},
    }
    if compare is not None:
        # Update only if a true new extremum is found
        for key in new_compare:
            if new_compare[key]["min"] < compare[key]["min"]:
                compare[key]["min"] = new_compare[key]["min"]
            if new_compare[key]["max"] > compare[key]["max"]:
                compare[key]["max"] = new_compare[key]["max"]
    else:
        # Initialize compare if it doesn't exist
        compare = new_compare

    return compare, tiled, pred_color, gt_color, diff_color, errors
    

def make_movie(images_dir, output_dir, video_name='output_video.mp4', framerate=30):
    first_image_path = os.path.join(images_dir, os.listdir(images_dir)[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape

    # Ensure the 'video' subdirectory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the video writer
    out_video = cv2.VideoWriter(
        os.path.join(output_dir, f'{video_name}.mp4'),
        cv2.VideoWriter_fourcc(*'mp4v'),
        framerate,
        (width, height)
    )
    for fname in os.listdir(images_dir):
        img_path = os.path.join(images_dir, fname)
        img = cv2.imread(img_path)
        out_video.write(img)

    out_video.release()
