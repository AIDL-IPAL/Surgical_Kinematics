import cv2
import os
import argparse
import sys

def process_video(video_path, camera_index, frame_start=0, frame_max=None):
    """
    Extracts frames from a video file and saves them as PNG images.

    Args:
        video_path (str): Path to the input video file.
        camera_index (int): Index of the camera (e.g., 0 or 1).
        frame_start (int, optional): Frame number to start extraction from. Defaults to 0.
        frame_max (int, optional): Maximum number of frames to extract. If None, all frames are extracted.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}", file=sys.stderr)
        return

    video_dir = os.path.dirname(video_path)
    output_folder_name = f"image0{camera_index+1}"
    output_dir = os.path.join(video_dir, output_folder_name)

    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory created/exists: {output_dir}")
    except OSError as e:
        print(f"Error creating directory {output_dir}: {e}", file=sys.stderr)
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}", file=sys.stderr)
        return

    frame_count = 0
    print(f"Processing video: {video_path}...")

    # Skip frames until frame_start
    if frame_start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
        frame_count = frame_start

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video or error reading frame

        # Format frame count with leading zeros (9 digits total)
        frame_filename = f"{frame_count:09d}.png"
        output_path = os.path.join(output_dir, frame_filename)

        # Skip if frame already exists
        if os.path.exists(output_path):
            frame_count += 1
            if frame_count % 100 == 0:  # Still print progress even when skipping
                print(f"  Skipped/Processed {frame_count} frames...")
            if frame_max is not None and frame_count >= frame_start + frame_max:
                break
            continue  # Skip to the next frame

        try:
            # Save frame as PNG
            if not cv2.imwrite(output_path, frame):
                print(f"Warning: Failed to write frame {frame_count} to {output_path}", file=sys.stderr)
        except Exception as e:
            print(f"Error writing frame {frame_count} to {output_path}: {e}", file=sys.stderr)
            # Optionally break or continue based on desired error handling
            # break

        frame_count += 1
        if frame_count % 100 == 0:  # Print progress every 100 frames
            print(f"  Processed {frame_count} frames...")
        if frame_max is not None and frame_count >= frame_start + frame_max:
            break

    cap.release()
    print(f"Finished processing {video_path}. Total frames extracted: {frame_count - frame_start}")

if __name__ == "__main__":
    # Hardcoded video file paths
    video_files = [
        "scene3d/acq_data/06_25_2025/cholec_10_51_25/camera0_main_20250625_112815_chunk.avi",
        "scene3d/acq_data/06_25_2025/cholec_10_51_25/camera1_main_20250625_112815_chunk.avi"
    ]
    frame_start = 0
    frame_max = None

    if len(video_files) == 0:
        print("Error: Please provide at least one video file path.", file=sys.stderr)
        sys.exit(1)

    for i, video_file in enumerate(video_files):
        process_video(video_file, i, frame_start=frame_start, frame_max=frame_max)

    print("Script finished.")