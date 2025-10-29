"""
Metric3D Depth Estimation Pipeline
A clean implementation for depth estimation using Metric3D models
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import argparse
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib


class DepthEstimator:
    """Metric3D depth estimation wrapper"""
    
    # Model configuration
    MODEL_CONFIGS = {
        'vit': {
            'input_size': (616, 1064),
            'model_name': 'metric3d_vit_small'
        },
        'convnext': {
            'input_size': (544, 1216),
            'model_name': 'metric3d_convnext_small'
        }
    }
    
    # Default normalization parameters for ImageNet
    IMAGENET_MEAN = [123.675, 116.28, 103.53]
    IMAGENET_STD = [58.395, 57.12, 57.375]
    
    def __init__(self, 
                 model_type: str = 'vit',
                 checkpoint_path: Optional[str] = None,
                 device: str = 'cuda'):
        """
        Initialize the depth estimator
        
        Args:
            model_type: Type of model ('vit' or 'convnext')
            checkpoint_path: Path to custom checkpoint (optional)
            device: Device to run on ('cuda' or 'cpu')
        """
        self.model_type = model_type
        self.config = self.MODEL_CONFIGS[model_type]
        self.input_size = self.config['input_size']
        self.device = device
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        
    def _load_model(self, checkpoint_path: Optional[str] = None):
        """Load the Metric3D model"""
        print(f"Loading {self.config['model_name']} model...")
        
        # Load pretrained model from torch hub
        model = torch.hub.load('yvanyin/metric3d', 
                              self.config['model_name'], 
                              pretrain=True).to(self.device)
        
        # Load custom checkpoint if provided
        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"Loading checkpoint from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, 
                                   map_location=self.device, 
                                   weights_only=True)
            model.load_state_dict(state_dict)
        
        model.eval()
        return model
    
    def crop_with_intrinsics(self, frame: np.ndarray, intrinsic: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
        # Crop the image using x1, x2, y1, y2 as distances from edges
        # x1: distance from left edge, x2: distance from right edge
        # y1: distance from top edge, y2: distance from bottom edge
        x1, x2, y1, y2 = 200, 200, 50, 50
        H, W = frame.shape[:2]
        cropped_frame = frame[y1:H - y2, x1:W - x2]

        # Adjust intrinsics
        intrinsic_cropped = intrinsic.copy()
        intrinsic_cropped[2] -= x1  # cx
        intrinsic_cropped[3] -= y1  # cy

        return cropped_frame, intrinsic_cropped, (x1, x2, y1, y2)

    def preprocess_image(self, 
                        image: np.ndarray, 
                        intrinsics: List[float]) -> Tuple[torch.Tensor, List[float], List[int]]:
        """
        Preprocess image for model input
        
        Args:
            image: Input RGB image (H, W, 3)
            intrinsics: Camera intrinsics [fx, fy, cx, cy]
            
        Returns:
            Preprocessed tensor, scaled intrinsics, padding info
        """
        h, w = image.shape[:2]
        
        # Calculate scale to fit input size while maintaining aspect ratio
        scale = min(self.input_size[0] / h, self.input_size[1] / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), 
                            interpolation=cv2.INTER_LINEAR)
        
        # Scale intrinsics accordingly
        scaled_intrinsics = [i * scale for i in intrinsics]
        
        # Calculate padding
        pad_h = self.input_size[0] - new_h
        pad_w = self.input_size[1] - new_w
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        
        # Apply padding with ImageNet mean values
        padded = cv2.copyMakeBorder(
            resized,
            pad_h_half, pad_h - pad_h_half,
            pad_w_half, pad_w - pad_w_half,
            cv2.BORDER_CONSTANT,
            value=self.IMAGENET_MEAN
        )
        
        pad_info = [pad_h_half, pad_h - pad_h_half, 
                   pad_w_half, pad_w - pad_w_half]
        
        # Convert to tensor and normalize
        tensor = torch.from_numpy(padded.transpose((2, 0, 1))).float()
        
        mean = torch.tensor(self.IMAGENET_MEAN).float()[:, None, None]
        std = torch.tensor(self.IMAGENET_STD).float()[:, None, None]
        tensor = (tensor - mean) / std
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0).to(self.device)
        
        return tensor, scaled_intrinsics, pad_info
    
    def postprocess_depth(self,
                         pred_depth: torch.Tensor,
                         pad_info: List[int],
                         original_shape: Tuple[int, int],
                         scaled_intrinsics: List[float],
                         global_scale: float | None = None) -> np.ndarray:
        """
        Postprocess predicted depth map
        
        Args:
            pred_depth: Predicted depth tensor
            pad_info: Padding information
            original_shape: Original image shape (H, W)
            scaled_intrinsics: Scaled camera intrinsics (after resize)
            
        Returns:
            Processed depth map in original resolution
        """
        # Remove batch dimension
        depth = pred_depth.squeeze()
        
        # Remove padding
        depth = depth[pad_info[0]:depth.shape[0] - pad_info[1],
                     pad_info[2]:depth.shape[1] - pad_info[3]]
        
        # Upsample to original resolution
        depth = torch.nn.functional.interpolate(
            depth[None, None, :, :],
            original_shape,
            mode='bilinear'
        ).squeeze()
        
        # De-canonical transform
        # Convert from canonical camera space to real metric depth
        canonical_focal = 1000.0  # Focal length of canonical camera
        canonical_to_real_scale = scaled_intrinsics[0] / canonical_focal
        depth = depth * canonical_to_real_scale
        
        # Apply global scale if provided
        if global_scale is not None:
            depth = depth * global_scale

        # Clamp depth values
        depth = torch.clamp(depth, 0, 255)
        
        return depth.cpu().numpy()
    
    @torch.no_grad()
    def estimate_depth(self, 
                      image_path: str,
                      intrinsics: List[float]) -> Dict:
        """
        Estimate depth from a single image
        
        Args:
            image_path: Path to input image
            intrinsics: Camera intrinsics [fx, fy, cx, cy]
            
        Returns:
            Dictionary containing depth map and confidence
        """
        # Load image (BGR to RGB)
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")
        
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        original_shape = image_rgb.shape[:2]
        
        # Preprocess
        input_tensor, scaled_intrinsics, pad_info = self.preprocess_image(
            image_rgb, intrinsics
        )
        
        # Run inference
        pred_depth, confidence, output_dict = self.model.inference({
            'input': input_tensor
        })
        
        # Postprocess - pass scaled intrinsics for de-canonical transform
        depth_map = self.postprocess_depth(
            pred_depth, pad_info, original_shape, scaled_intrinsics
        )
        
        return {
            'depth': depth_map,
            'confidence': confidence.cpu().numpy() if confidence is not None else None,
            'output_dict': output_dict
        }
    
    @torch.no_grad()
    def infer_depth(self, frame_bgr: np.ndarray, intrinsics: List[float], global_scale: float | None = None):
        # Crop the image using x1, x2, y1, y2 as distances from edges
        from scene3d.utils.dataprocess import crop_with_intrinsics, uncrop_depth_map
        original_shape = frame_bgr.shape[:2]
        frame_bgr, intrinsics, crop_info = crop_with_intrinsics(frame_bgr, intrinsics)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # Metric3D expects RGB input
        og_shape = frame_rgb.shape[:2]

        rgb_t, scaled_intrinsics, pad_info = self.preprocess_image(frame_rgb, intrinsics)  # type: ignore

        if rgb_t.ndim == 3:
            rgb_t = rgb_t.unsqueeze(0)  # Add batch dimension if missing
        rgb_t = rgb_t.to(self.device)

        # Run inference
        pred_depth, _, _ = self.model.inference({
            'input': rgb_t
        })
    
        if torch.isnan(pred_depth).any():
            print(f"NaN values in predicted depth.")
            return None

        # pred_depth = metric3d_unpad_and_scale(pred_depth, scaled_intrinsics, pad_info, frame_bgr.shape[:2])  # type: ignore
        pred_depth = self.postprocess_depth(pred_depth, pad_info, og_shape, scaled_intrinsics, global_scale)

        # Reorder crop_info from (x1, x2, y1, y2) to (pad_top, pad_bottom, pad_left, pad_right)
        pad_top, pad_bottom, pad_left, pad_right = crop_info[2], crop_info[3], crop_info[0], crop_info[1]
        pred_depth = uncrop_depth_map(pred_depth, (pad_top, pad_bottom, pad_left, pad_right), original_shape)  # type: ignore
        # pred_depth = pred_depth.squeeze(1) # squeeze color channel dim
        return pred_depth

def save_depth_map(depth_map: np.ndarray, 
                  output_path: str,
                  scale_factor: float = 100.0) -> None:
    """
    Save depth map as 16-bit PNG
    
    Args:
        depth_map: Depth map in meters
        output_path: Output file path
        scale_factor: Scale factor for saving (default 100 for cm)
    """
    # Convert to specified units and round
    depth_scaled = depth_map * scale_factor
    depth_int = np.round(depth_scaled).astype(np.uint16)
    
    # Print statistics
    print(f"Depth statistics:")
    print(f"  Min: {np.min(depth_map):.3f} m ({np.min(depth_int)} scaled units)")
    print(f"  Max: {np.max(depth_map):.3f} m ({np.max(depth_int)} scaled units)")
    print(f"  Mean: {np.mean(depth_map):.3f} m")
    
    # Save as 16-bit PNG
    cv2.imwrite(output_path, depth_int)
    print(f"Saved depth map to {output_path}")

def visualize_depth(depth_map: np.ndarray, 
                   output_path: str,
                   colormap: int = cv2.COLORMAP_JET) -> None:
    """
    Create a colored visualization of the depth map
    
    Args:
        depth_map: Depth map in meters
        output_path: Output file path
        colormap: OpenCV colormap to use
    """
    # Normalize for visualization
    depth_norm = depth_map - np.min(depth_map)
    depth_norm = (depth_norm / np.max(depth_norm) * 255).astype(np.uint8)
    
    # Create visualization with colorbar using matplotlib
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Display depth map with colormap
    norm = Normalize(vmin=np.min(depth_map), vmax=np.max(depth_map))
    cmap = matplotlib.colormaps.get_cmap('jet')
    im = ax.imshow(depth_map, cmap=cmap, norm=norm)
    
    # Add colorbar on the side
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Depth (meters)', fontsize=12)
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Save combined visualization
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved depth visualization to {output_path}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Metric3D Depth Estimation')
    parser.add_argument('--image', type=str, default='0000001136.png',
                       help='Path to input image')
    parser.add_argument('--model', type=str, default='vit',
                       choices=['vit', 'convnext'],
                       help='Model type to use')
    parser.add_argument('--checkpoint', type=str, default='ckpt_20250513_epoch8.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='pred.png',
                       help='Output depth map path')
    parser.add_argument('--viz', type=str, default='depth_viz.png',
                       help='Output visualization path')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run on')
    parser.add_argument('--fx', type=float, default=707.0493,
                       help='Focal length x')
    parser.add_argument('--fy', type=float, default=707.0493,
                       help='Focal length y')
    parser.add_argument('--cx', type=float, default=604.0814,
                       help='Principal point x')
    parser.add_argument('--cy', type=float, default=180.5066,
                       help='Principal point y')
    
    args = parser.parse_args()
    
    # Camera intrinsics
    intrinsics = [args.fx, args.fy, args.cx, args.cy]
    
    try:
        # Initialize estimator
        print(f"Initializing {args.model.upper()} model...")
        estimator = DepthEstimator(
            model_type=args.model,
            checkpoint_path=args.checkpoint if Path(args.checkpoint).exists() else None,
            device=args.device
        )
        
        # Estimate depth
        print(f"\nProcessing {args.image}...")
        results = estimator.estimate_depth(args.image, intrinsics)
        
        # Save results
        save_depth_map(results['depth'], args.output)
        visualize_depth(results['depth'], args.viz)
        
        print("\nDepth estimation completed successfully!")
        
    except Exception as e:
        print(f"\nError: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())