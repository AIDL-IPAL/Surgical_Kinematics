import cv2
import numpy as np
import sys
import glob
import os
import random

# List your video files here or use glob to find them
video_files = sorted(glob.glob('datasets/SurgVU/evals_case_090_case_098/*.mp4'))
mode = 'randomclips'  # or 'fullstitch'

if not video_files:
    print("No video files found.")
    sys.exit(1)

# Read first video to get properties
cap = cv2.VideoCapture(video_files[0])
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

# Define fourcc before mode checks
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

if mode == 'fullstitch':
    # Output video writer for full stitch
    out = cv2.VideoWriter(f'stitched_output_{mode}.mp4', fourcc, fps, (width, height))

elif mode == 'randomclips':
    # Output video writer for sample stitch (10s random from each)
    out = cv2.VideoWriter(f'stitched_output_{mode}.mp4', fourcc, fps, (width, height))

# Create 10 black frames
black_frame = np.zeros((height, width, 3), dtype=np.uint8)
black_frames = [black_frame] * 10

for idx, vf in enumerate(video_files):
    if mode == 'fullstitch':
        # --- Full video stitch ---
        cap = cv2.VideoCapture(vf)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        cap.release()
        # Add black frames between videos, except after the last one
        if idx < len(video_files) - 1:
            for bf in black_frames:
                out.write(bf)
    elif mode == 'randomclips':
        # --- Sample 10s random segment for sample stitch ---
        N = 10
        cap = cv2.VideoCapture(vf)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        N_sec_frames = int(fps * N)
        if total_frames <= N_sec_frames:
            start_frame = 0
        else:
            start_frame = random.randint(0, total_frames - N_sec_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames_written = 0
        while frames_written < N_sec_frames:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            frames_written += 1
        cap.release()
        # Add black frames between videos, except after the last one
        if idx < len(video_files) - 1:
            for bf in black_frames:
                out.write(bf)
    else:
        print("Invalid mode. Use 'fullstitch' or 'randomclips'.")
        sys.exit(1)

out.release()
print(f"Stitched video saved as stitched_output_{mode}.mp4")
