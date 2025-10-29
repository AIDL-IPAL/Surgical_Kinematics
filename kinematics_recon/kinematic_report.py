import cv2
import numpy as np
import random
import os

import matplotlib.pyplot as plt

def extract_random_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = sorted(random.sample(range(total_frames), num_frames))
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    cap.release()
    return frames

def display_frames_grid(frames, grid_shape=(4, 4), save_path=None):
    fig, axes = plt.subplots(*grid_shape, figsize=(12, 12))
    for ax, frame in zip(axes.flatten(), frames):
        ax.imshow(frame)
        ax.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    video_path = "pose_analytics/models/yolo11s_0907/test_results/camera0_20250806_172903_chunk_annotated.mp4"  # Replace with your video file path
    frames = extract_random_frames(video_path, num_frames=16)
    parent_folder = os.path.dirname(video_path)
    save_path = os.path.join(parent_folder, "random_frames_grid.png")
    display_frames_grid(frames, grid_shape=(4, 4), save_path=save_path)