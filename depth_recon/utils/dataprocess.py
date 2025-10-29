import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from typing import List, Tuple
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_intrinsics(path):
    """
    Reads a 3x4 camera matrix from `path` and returns [fx, fy, cx, cy].
    Assumes the file is plain text with 3 rows of 4 numbers each.
    """
    # load the full 3x4 matrix
    M = np.loadtxt(path)       # shape (3,4)

    # fx = M[0,0], fy = M[1,1], cx = M[0,2], cy = M[1,2]
    fx = M[0, 0]
    fy = M[1, 1]
    cx = M[0, 2]
    cy = M[1, 2]

    return np.array([fx, fy, cx, cy], dtype=float)

def crop_with_intrinsics(frame: np.ndarray, intrinsic: List[float]) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    # Crop the image using x1, x2, y1, y2 as distances from edges
    # x1: distance from left edge, x2: distance from right edge
    # y1: distance from top edge, y2: distance from bottom edge
    x1, x2, y1, y2 = 200, 200, 50, 50
    H, W = frame.shape[:2]
    cropped_frame = frame[y1:H - y2, x1:W - x2]

    # Adjust intrinsics
    intrinsic_cropped = intrinsic[:]  # shallow copy for list
    intrinsic_cropped[2] -= x1  # cx
    intrinsic_cropped[3] -= y1  # cy

    return cropped_frame, intrinsic_cropped, (x1, x2, y1, y2)

def uncrop_depth_map(depth_map_np: np.ndarray, crop_info: Tuple[int, int, int, int], frame_shape: Tuple[int, int]) -> np.ndarray:
    # Uncrop depth map by padding with black (zeros) according to crop_info
    if depth_map_np is not None and crop_info is not None and frame_shape is not None:
        # crop_info: (y0, y1, x0, x1) are paddings: top, bottom, left, right
        pad_top, pad_bottom, pad_left, pad_right = crop_info
        H, W = frame_shape
        uncropped = np.zeros((H, W), dtype=depth_map_np.dtype)
        uncropped[pad_top:H-pad_bottom, pad_left:W-pad_right] = depth_map_np
        depth_map_np = uncropped
    return depth_map_np

def get_bounds(img, threshold=5):
    """
    Get the bounding box coordinates of non-threshold regions in the image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        return x, y, x + w, y + h
    else:
        return 0, 0, img.shape[1], img.shape[0]

def global_crop_black_borders(self, left_img, right_img, threshold=5):

    x0_left, y0_left, x1_left, y1_left = get_bounds(left_img, threshold)
    x0_right, y0_right, x1_right, y1_right = get_bounds(right_img, threshold)

    # Global crop: take the worst-case crop for each side
    x0_global = int(max(x0_left, x0_right)*1.05)
    y0_global = max(y0_left, y0_right)
    x1_global = int(min(x1_left, x1_right)*0.95)
    y1_global = min(y1_left, y1_right)

    left_cropped = left_img[y0_global:y1_global, x0_global:x1_global]
    right_cropped = right_img[y0_global:y1_global, x0_global:x1_global]

    return left_cropped, right_cropped, (x0_global, y0_global)

def pad_collate_fn(batch):
    """
    Pads each tensor in the batch to the max height/width in the batch.
    Assumes RGB, depth, and mask are all [C, H, W] or [H, W] format.
    """
    from torch.nn.functional import pad

    # Unpack batch
    rgb_batch, depth_batch, mask_batch, rgb_og_batch, img_name, pad_info_batch, intrinsic_batch, hw_batch = zip(*batch)

    max_h = max([rgb.shape[-2] for rgb in rgb_batch])
    max_w = max([rgb.shape[-1] for rgb in rgb_batch])

    def pad_tensor(t, pad_value=0):
        pad_h = max_h - t.shape[-2]
        pad_w = max_w - t.shape[-1]
        padding = (0, pad_w, 0, pad_h)
        return pad(t, padding, value=pad_value)

    rgb_batch = torch.stack([pad_tensor(rgb) for rgb in rgb_batch])
    rgb_og_batch = torch.stack([pad_tensor(rgb) for rgb in rgb_og_batch])
    depth_batch = torch.stack([pad_tensor(depth) for depth in depth_batch])
    mask_batch = torch.stack([pad_tensor(mask) for mask in mask_batch])

    img_name = [name for name in img_name]

    pad_info_batch = torch.stack(pad_info_batch)
    intrinsic_batch = torch.stack(intrinsic_batch)

    return rgb_batch, depth_batch, mask_batch, rgb_og_batch, img_name, pad_info_batch, intrinsic_batch, hw_batch

def metric3d_prep_img(rgb_origin, intrinsic, device=DEVICE):    
    #### ajust input size to fit pretrained model
    # assuming rgb_origin is HWC, rgb format
    # keep ratio resize
    input_size = (616, 1064) # for vit model
    h, w = rgb_origin.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    # Scale intrinsic, hold depth
    intrinsic = [intrinsic[0] * scale, intrinsic[1] * scale, intrinsic[2] * scale, intrinsic[3] * scale]
    # padding to input_size
    padding = [123.675, 116.28, 103.53]
    h, w = rgb.shape[:2]
    pad_h = input_size[0] - h
    pad_w = input_size[1] - w
    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2
    rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)
    pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

    #### normalize
    mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
    rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
    rgb = torch.div((rgb - mean), std)
    return rgb, pad_info, intrinsic

import torch.nn.functional as F

def metric3d_prep_depth(depth_map, input_size=(616, 1064)):
    h, w = depth_map.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    depth_resized = cv2.resize(depth_map, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)
    
    h, w = depth_resized.shape
    pad_h = input_size[0] - h
    pad_w = input_size[1] - w
    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2

    depth_padded = np.pad(depth_resized, 
                          ((pad_h_half, pad_h - pad_h_half), (pad_w_half, pad_w - pad_w_half)), 
                          mode='constant', constant_values=0)

    pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
    return torch.tensor(depth_padded, dtype=torch.float32), pad_info

def get_mask(frame, frame02=None, scale=1000.0):
    """
    Preprocess the depth frame and create a mask tensor for valid depth values.
    Parameters:
        frame (torch.Tensor): The primary depth frame as a torch tensor.
        frame02 (torch.Tensor, optional): A secondary depth frame as a torch tensor. 
            Defaults to None.
    Returns:
        tuple:
            - torch.Tensor: The processed depth frame.
            - torch.Tensor: A boolean mask tensor indicating valid depth values.
    """
    depth_map = frame.clone()

    # Create mask for valid depth values
    mask = (depth_map > 1e-3) & (depth_map < 500)  # Accurate depth in 3mm-500mm range
    if frame02 is not None:
        frame02 = torch.tensor(frame02, dtype=torch.float32)
        difference = torch.abs(depth_map - frame02)
        threshold = 15 / scale  # 15mm threshold for depth difference
        mask = mask & (frame02 > 3) & (frame02 < 500)
        mask = mask & (difference <= threshold)

    return mask

def metric3d_unpad_and_scale(pred_depth, intrinsic, pad_info, target_shapes):
    """
    Unpad, upsample, and scale predicted depth to metric scale.
    """
    outputs = []
    B = pred_depth.shape[0]
    # print(f"shape of pred_depth: {pred_depth.shape}")

    # Find the max target H/W in the batch
    # max_h = max([shape[0] for shape in target_shapes])
    # max_w = max([shape[1] for shape in target_shapes])

    for i in range(B):
        # pred_depth may be [B, 1, H, W] or [B, H, W]
        d = pred_depth[i]
        # if d.ndim == 3 and d.shape[0] == 1:
        #     d = d[0]  # [H, W]
        # p = pad_info[i].tolist() if isinstance(pad_info[i], torch.Tensor) else pad_info[i]

        # # Unpad
        # h, w = d.shape
        # d = d[p[0]:h - p[1], p[2]:w - p[3]]

        # # Resize to original shape (optional, skip if already resizing to max)
        # target_h, target_w = target_shapes[i]
        # d = F.interpolate(d.unsqueeze(0).unsqueeze(0), size=(target_h, target_w), mode='bilinear', align_corners=False).squeeze()

        # Intrinsic scaling
        fx = intrinsic[i][0] if isinstance(intrinsic, (list, np.ndarray)) else intrinsic[i][0]
        d = d * fx

        # Resize to max shape
        # d = F.interpolate(d.unsqueeze(0).unsqueeze(0), size=(max_h, max_w), mode='bilinear', align_corners=False).squeeze()
        outputs.append(d)

    return torch.stack(outputs, dim=0)  # [B, 1, max_H, max_W]

def unpad_and_upsample(pred_depth, pad_info, orig_shape):
    pred_depth = pred_depth.squeeze()
    pred_depth = pred_depth[pad_info[0] : pred_depth.shape[0] - pad_info[1], pad_info[2] : pred_depth.shape[1] - pad_info[3]]
    # upsample to original size
    pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], orig_shape, mode='bilinear').squeeze()
    pred_depth = pred_depth.cpu().numpy()
    return pred_depth

def postprocess_depth(depth_map, mask=None, scale=None):
    """
    Postprocess the depth map by applying the mask and scaling it back to mm.
    Returns [B, 1, H, W] tensor.
    Two separate scale options to preserve training scaling & post-training optimizations
    """
    if mask is not None:
        if depth_map.ndim == 4 and mask.ndim == 3:
            mask = mask.unsqueeze(1)  # [B, 1, H, W]
        if mask.shape != depth_map.shape:
            mask = F.interpolate(mask.float(), size=depth_map.shape[-2:], mode='nearest')
    else:
        mask = (depth_map > (1.5 / (scale))) & (depth_map < (250 / (scale)))
        if depth_map.ndim == 4 and mask.ndim == 3:
            mask = mask.unsqueeze(1)  # [B, 1, H, W]
    mask = mask.bool()  # Ensure mask is boolean
    depth_map = torch.where(mask, depth_map, torch.tensor(0.0, device=depth_map.device))
    # Ensure shape is [B, 1, H, W]
    if depth_map.ndim == 3:
        depth_map = depth_map.unsqueeze(1)

    if scale is not None:
        depth_map = depth_map.mul_(scale)
    return depth_map

def denormalize_rgb(rgb_s, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
    # rgb_s = rgb[0].detach().cpu().numpy()
    
    # Denormalize using the same mean and std as in metric3d_prep_img
    mean = np.array(mean).reshape(3, 1, 1)
    std = np.array(std).reshape(3, 1, 1)
    rgb_sample = rgb_s * std + mean
    
    # Convert from CHW to HWC if needed
    if rgb_sample.shape[0] == 3:
        rgb_sample = rgb_sample.transpose(1, 2, 0)
    
    # Clip values to valid range and convert to uint8 for display
    rgb_sample = np.clip(rgb_sample, 0, 255).astype(np.uint8)
    
    # Also process raw rgb_s for plotting
    rgb_raw = rgb_s.copy()
    if rgb_raw.shape[0] == 3:
        rgb_raw = rgb_raw.transpose(1, 2, 0)
    # Normalize rgb_raw to [0, 1] for display
    min_val = np.min(rgb_raw)
    max_val = np.max(rgb_raw)
    if max_val > min_val:
        rgb_raw = (rgb_raw - min_val) / (max_val - min_val)
    else:
        # Handle the case where all values are the same
        rgb_raw = np.zeros_like(rgb_raw) # Or set to a constant like 0.5
    return rgb_sample, rgb_raw
    
class DepthDataset(Dataset):
    def __init__(self, img_dir, intrinsic, depth_dir, start_idx=0, end_idx=None, transform=None):
        self.img_dir = img_dir
        self.depth_dir = depth_dir
        self.intrinsic = intrinsic

        all_images = sorted(os.listdir(img_dir))
        all_depths = sorted(os.listdir(depth_dir))
        min_len = min(len(all_images), len(all_depths))
        if end_idx is not None:
            min_len = min(min_len, end_idx)
        
        self.images = []
        self.depths = []
        for img_name, depth_name in zip(all_images[:min_len], all_depths[:min_len]):
            img_path = os.path.join(img_dir, img_name)
            depth_path = os.path.join(depth_dir, depth_name)
            img = cv2.imread(img_path)
            depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
            if img is None:
                print(f"[DepthDataset] Skipping invalid RGB image: {img_path}")
                continue
            if depth is None:
                print(f"[DepthDataset] Skipping invalid depth file: {depth_path}")
                continue

            self.images.append(img_name)
            self.depths.append(depth_name)

        self.transform = transform
        self.start_idx = start_idx
        self.end_idx = end_idx if end_idx is not None else min_len

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        if self.start_idx > 0:
            idx = (idx + self.start_idx) % len(self.images)
        else:
            idx = idx % len(self.images)

        img_path = os.path.join(self.img_dir, self.images[idx])
        img_name = self.images[idx].split('.')[0]
        rgb_origin = cv2.imread(img_path)
        h, w = rgb_origin.shape[:2]
        rgb_origin = cv2.cvtColor(rgb_origin, cv2.COLOR_BGR2RGB).copy()  # ensures positive strides
        
        depth_path = os.path.join(self.depth_dir, self.depths[idx])
        depth_gt = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        if depth_gt is None:
            raise FileNotFoundError(f"Failed to read depth image: {depth_path}")
        depth_gt = depth_gt.astype(np.float32).copy()
        # print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024 ** 2} MB")

        # Crop black borders automatically
        # rgb_crop = auto_crop_black_borders(rgb_origin)
        # depth_gt = auto_crop_black_borders(depth_gt)

        rgb_prep, pad_info, intrinsic = metric3d_prep_img(rgb_origin, self.intrinsic)  # Preprocess image
        # Preprocess depth map
        depth_gt, _ = metric3d_prep_depth(depth_gt)
        # depth_gt = torch.tensor(depth_gt, dtype=torch.float32).detach().clone()
        mask = get_mask(depth_gt)
        # optional scale depth...
        depth_gt_prep = depth_gt * mask  # Apply mask to filter invalid depth values
        # print(f"Post-depth prep Allocated memory: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
        rgb_origin = torch.tensor(rgb_origin, dtype=torch.float32).permute(2, 0, 1)
        
        if self.transform:
            rgb_prep = self.transform(rgb_prep)

        return rgb_prep, depth_gt_prep, mask, rgb_origin, img_name, torch.tensor(pad_info), torch.tensor(intrinsic), (h,w)

def unzip_hamlyn_data(args):
    import zipfile
    # unzip HAMLYN data
    rectified_zips = [d for d in os.listdir(args.data_dir) if d.startswith('rectified') and d.endswith('.zip')]
    dataset_list = []
    for zip_name in rectified_zips:
        dir_id = zip_name.replace('rectified', '').replace('.zip', '')
        unzip_path = os.path.join(args.data_dir, f'rectified{dir_id}')
        if not os.path.exists(unzip_path):
            try:
                with zipfile.ZipFile(os.path.join(args.data_dir, zip_name), 'r') as zip_ref:
                    zip_ref.extractall(args.data_dir)
                print(f"Unzipped {zip_name} to {unzip_path}")
            except zipfile.BadZipFile:
                print(f"Failed to unzip {zip_name}. Skipping this file.")
                continue
    # unzip calibration data
    calibration_zips = [d for d in os.listdir(args.data_dir) if d.startswith('calibration') and d.endswith('.zip')]
    for zip_name in calibration_zips:
        unzip_path = os.path.join(args.data_dir, 'calibration') # Assuming calibration zips extract to a 'calibration' folder
        if not os.path.exists(unzip_path): # Check if the general calibration folder exists
            try:
                with zipfile.ZipFile(os.path.join(args.data_dir, zip_name), 'r') as zip_ref:
                    zip_ref.extractall(args.data_dir) # Extract to data_dir, it should create 'calibration'
                print(f"Unzipped {zip_name} to {args.data_dir}")
            except zipfile.BadZipFile:
                print(f"Failed to unzip {zip_name}. Skipping this file.")
                continue
        else:
            # If the main calibration folder exists, check for specific subfolders if needed
            # For now, we assume if 'calibration' exists, its contents are likely there.
            # More specific checks could be added if calibration zips have varying structures.
            print(f"Calibration folder {unzip_path} already exists. Assuming unzipped.")
            break # Assuming one calibration.zip contains all necessary calibration files
