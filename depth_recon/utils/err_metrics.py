import torch
import numpy as np
import cv2
from tqdm import tqdm


def get_error_metrics(gt_depth, pred_depth, diff, mask):
    """
    Calculate error metrics for depth estimation.
    params:
    gt_depth: Ground truth depth tensor.
    pred_depth: Predicted depth tensor.
    diff: Difference tensor between predicted and ground truth depth.
    mask: Mask tensor indicating valid pixels. (1 for valid, 0 for invalid).
    returns:
    Dictionary containing:
        Mean Squared Error (MSE), Mean Absolute Error (MAE), Mean Absolute Percent Error (MAPE),
        R-squared (R2).
    """

    mse = torch.mean(diff ** 2)
    mae = torch.mean(torch.abs(diff))
    mape = (torch.mean(torch.abs(diff / gt_depth[mask])) * 100) if torch.sum(gt_depth[mask]) > 0 else float('nan')

    # R-squared calculation
    gt_mean = torch.mean(gt_depth[mask])
    ss_total = torch.sum((gt_depth[mask] - gt_mean) ** 2)
    ss_residual = torch.sum((gt_depth[mask] - pred_depth[mask]) ** 2)
    r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else float('nan')

    errors = {
    'MSE': mse.item(),
    'MAE': mae.item(),
    'MAPE': mape if isinstance(mape, float) else mape.item(),
    'R2': r2 if isinstance(r2, float) else r2.item()
    }

    return errors
