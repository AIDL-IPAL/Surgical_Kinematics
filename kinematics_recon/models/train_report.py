

from glob import glob
import os
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple, Optional


def read_csv_file(file_path: str) -> pd.DataFrame:
    """Read CSV file and return DataFrame.
    
    Args:
        file_path (str): Path to CSV file
        
    Returns:
        pd.DataFrame: DataFrame containing the CSV data
    """
    return pd.read_csv(file_path)


def plot_metrics(csv_files: List[str], metrics: Optional[List[str]] = None, 
                 output_dir: str = "plots", figsize: Tuple[int, int] = (12, 8)) -> None:
    """Plot training and validation metrics from multiple CSV files.
    
    Args:
        csv_files (List[str]): List of paths to CSV files
        metrics (Optional[List[str]]): List of metrics to plot. If None, plot standard metrics
        output_dir (str): Directory to save the plots
        figsize (Tuple[int, int]): Size of the figure
    """
    if metrics is None:
        # Default metrics to plot (train and val pairs)
        metrics = [
            ('train/box_loss', 'val/box_loss'),
            ('train/pose_loss', 'val/pose_loss'),
            ('train/cls_loss', 'val/cls_loss'),
            ('metrics/mAP50(B)', 'metrics/mAP50(P)'),
            ('metrics/mAP50-95(B)', 'metrics/mAP50-95(P)')
        ]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    _, axs = plt.subplots(2, 1, figsize=figsize, sharex=True)
    # Group metrics for tiled plots
    # 1. mAP plots in a 2x1 grid
    map_metrics = [
        ('metrics/mAP50(B)', 'metrics/mAP50(P)'),
        ('metrics/mAP50-95(B)', 'metrics/mAP50-95(P)')
    ]
    
    # 2. Box and cls loss in a 2x1 grid
    loss_metrics = [
        ('train/box_loss', 'val/box_loss'),
        ('train/cls_loss', 'val/cls_loss')
    ]
    
    # 3. Pose loss separate
    pose_metrics = [('train/pose_loss', 'val/pose_loss')]
    # Reorganize metrics for a 2x2 tiled figure
    # Top row: Box mAP and Pose mAP
    # Bottom row: Box/cls losses and Pose losses
    
    # Create a 1x2 tiled figure for only mAP50-95 metrics
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    
    # Plot Box mAP50-95 metric (left)
    ax = axs[0]
    for j, file_path in enumerate(csv_files):
        dir_path = os.path.dirname(file_path)
        filename = os.path.basename(dir_path) or os.path.basename(file_path)
        df = read_csv_file(file_path)
        color = plt.cm.tab10(j % 10)
        
        metric = 'metrics/mAP50-95(B)'
        if metric in df.columns:
            ax.plot(df['epoch'], df[metric], 
                   label=f"{filename} - {metric}", 
                   color=color, linestyle='-', marker='o', markersize=4)
    
    ax.set_ylabel('Value')
    ax.set_xlabel('Epoch')
    ax.set_title('Box mAP50-95')
    ax.legend(loc='best', fontsize='small')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Plot Pose mAP50-95 metric (right)
    ax = axs[1]
    for j, file_path in enumerate(csv_files):
        dir_path = os.path.dirname(file_path)
        filename = os.path.basename(dir_path) or os.path.basename(file_path)
        df = read_csv_file(file_path)
        color = plt.cm.tab10(j % 10)
        
        metric = 'metrics/mAP50-95(P)'
        if metric in df.columns:
            ax.plot(df['epoch'], df[metric], 
                   label=f"{filename} - {metric}", 
                   color=color, linestyle='-', marker='o', markersize=4)
    
    ax.set_ylabel('Value')
    ax.set_xlabel('Epoch')
    ax.set_title('Pose mAP50-95')
    ax.legend(loc='best', fontsize='small')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "map50-95_metrics.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Plot first two loss metrics (box_loss and cls_loss)
    for i, metric_pair in enumerate(loss_metrics):
        ax = axs[i]
        for j, file_path in enumerate(csv_files):
            dir_path = os.path.dirname(file_path)
            filename = os.path.basename(dir_path) or os.path.basename(file_path)
            df = read_csv_file(file_path)
            color = plt.cm.tab10(j % 10)
            
            if metric_pair[0] in df.columns:
                ax.plot(df['epoch'], df[metric_pair[0]], 
                       label=f"{filename} - {metric_pair[0]}", 
                       color=color, linestyle='-', marker='o', markersize=4)
            
            if metric_pair[1] in df.columns:
                ax.plot(df['epoch'], df[metric_pair[1]], 
                       label=f"{filename} - {metric_pair[1]}", 
                       color=color, linestyle='--', marker='x', markersize=4)
        
        ax.set_ylabel('Value')
        ax.set_title(f"{metric_pair[0]} and {metric_pair[1]}")
        ax.legend(loc='best', fontsize='small')
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Plot pose loss in the third subplot
    metric_pair = pose_metrics[0]
    ax = axs[2]
    for j, file_path in enumerate(csv_files):
        dir_path = os.path.dirname(file_path)
        filename = os.path.basename(dir_path) or os.path.basename(file_path)
        df = read_csv_file(file_path)
        color = plt.cm.tab10(j % 10)
        
        if metric_pair[0] in df.columns:
            ax.plot(df['epoch'], df[metric_pair[0]], 
                  label=f"{filename} - {metric_pair[0]}", 
                  color=color, linestyle='-', marker='o', markersize=4)
        
        if metric_pair[1] in df.columns:
            ax.plot(df['epoch'], df[metric_pair[1]], 
                  label=f"{filename} - {metric_pair[1]}", 
                  color=color, linestyle='--', marker='x', markersize=4)
    
    ax.set_ylabel('Value')
    ax.set_title(f"{metric_pair[0]} and {metric_pair[1]}")
    ax.legend(loc='best', fontsize='small')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    for ax in axs:
        ax.set_xlabel('Epoch')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_metrics_tiled.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot pose loss separately (keeping original implementation)
    for metric_pair in pose_metrics:
        plt.figure(figsize=figsize)
        
        for i, file_path in enumerate(csv_files):
            dir_path = os.path.dirname(file_path)
            filename = os.path.basename(dir_path) or os.path.basename(file_path)
            df = read_csv_file(file_path)
            color = plt.cm.tab10(i % 10)
            
            if metric_pair[0] in df.columns:
                plt.plot(df['epoch'], df[metric_pair[0]], 
                         label=f"{filename} - {metric_pair[0]}", 
                         color=color, linestyle='-', marker='o', markersize=4)
            
            if metric_pair[1] in df.columns:
                plt.plot(df['epoch'], df[metric_pair[1]], 
                         label=f"{filename} - {metric_pair[1]}", 
                         color=color, linestyle='--', marker='x', markersize=4)
        
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title(f"{metric_pair[0]} and {metric_pair[1]}")
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        metric_name = f"{metric_pair[0].replace('/', '_')}_{metric_pair[1].replace('/', '_')}"
        plt.savefig(os.path.join(output_dir, f"{metric_name}.png"), dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    # Example usage
    csv_files = [
        "pose_analytics/models/yolov8nano/results.csv",
        "pose_analytics/models/yolov8small/results.csv",
        "pose_analytics/models/yolov8med/results.csv",
        "pose_analytics/models/yolov8large/results.csv"
    ]
    
    plot_metrics(csv_files)
