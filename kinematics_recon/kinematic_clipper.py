"""
Idle period video clipper.

This script analyzes instrument tracking data to identify periods of low movement ("idle time")
and extracts corresponding video clips.

An idle period is defined as a contiguous sequence of frames where the instrument's
smoothed velocity remains below a specified threshold for a minimum duration.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from collections import defaultdict
from kinematic_utils import find_idle_periods

import numpy as np
import pandas as pd
import cv2

# --- Configuration ---
CSV_FPS = 15 # The frame rate of the analysis data (from CSV files).
KEYPOINT_IDX = 2  # Index of the keypoint to analyze for idle detection. KP-2 is end effector joint
VELOCITY_THRESHOLD = 0.5 # Velocity threshold (pixels per frame) to be considered idle.

# Minimum number of consecutive frames below the threshold to be an idle period.
MIN_IDLE_FRAMES = 30  # e.g., 2 seconds at 15 FPS
# Minimum duration in seconds for a final video clip to be saved.
MIN_CLIP_DURATION_SECONDS = 2.5
# Size of the moving average window for smoothing position data.
SMOOTHING_WINDOW = 5
# Minimum number of frames a track must have to be processed.
MIN_TRACK_LEN = 120

# ------------------------------- IO ------------------------------------------

def _expand_paths(paths: List[Path], glob_pattern: str) -> List[Path]:
    """Expand a list of paths, supporting both files and directories."""
    out: List[Path] = []
    for p in paths:
        if p.is_dir():
            out.extend(sorted(p.glob(glob_pattern)))
        elif p.is_file():
            out.append(p)
    return out

def load_tracks_by_source(csv_paths: List[Path]) -> Dict[int, pd.DataFrame]:
    """Load and group all track data by the source CSV index."""
    tracks_by_source: Dict[int, pd.DataFrame] = {}
    csvs = _expand_paths(csv_paths, "*.csv")
    for src_idx, csvp in enumerate(csvs):
        df = pd.read_csv(csvp)
        if 'instrument' not in df.columns:
            df['instrument'] = 'unknown'
        df['source_idx'] = src_idx
        tracks_by_source[src_idx] = df
    return tracks_by_source

def get_video_path(csv_path: Path, video_dir: Path) -> Optional[Path]:
    """Find the corresponding video file for a given CSV file."""
    # Assumes video has a common name pattern with the CSV.
    # e.g., 'case_001_tracking_report.csv' -> 'case_001.mp4'
    vid_name_stem = csv_path.stem
    
    # Handle suffixes like '_tracking_report'
    suffix_to_remove = '_tracking_report'
    if vid_name_stem.endswith(suffix_to_remove):
        vid_name_stem = vid_name_stem[:-len(suffix_to_remove)]

    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        potential_vid_path = video_dir / (vid_name_stem + ext)
        if potential_vid_path.exists():
            return potential_vid_path
    return None


# ---------------------------- Clip Export ----------------------------

def write_video_clip(cap: cv2.VideoCapture, start_f: int, end_f: int, out_path: Path):
    """Writes a segment of a video capture to a new file."""
    if not cap.set(cv2.CAP_PROP_POS_FRAMES, start_f):
        print(f"      [Warning] Failed to seek to frame {start_f}. Clip may be incorrect.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    if not vw.isOpened():
        print(f"      [Error] Failed to open VideoWriter for {out_path}. Skipping.")
        return

    for current_frame_num in range(start_f, end_f + 1):
        # The read() advances the capture, so we don't need to manually seek in the loop
        ret, frame = cap.read()
        if not ret:
            print(f"      [Warning] Could not read frame {current_frame_num}. Clip will be shorter.")
            break
        vw.write(frame)

    vw.release()
    print(f"      ✓ Saved clip: {out_path.name}")

# --------------------------------- CLI ---------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Extract video clips of instrument idle time.")
    ap.add_argument('--csv', type=Path, nargs='+', default=[Path('datasets/SurgVU/surgvu_049_inference')], help="Path to one or more CSV files or directories. Defaults to 'datasets/SurgVU/surgvu_049_inference'.")
    ap.add_argument('video_dir', type=Path, nargs='?', default=Path('datasets/SurgVU/surgvu_case_0_49_flattened'), help="Directory containing the corresponding video files. Defaults to 'datasets/SurgVU/surgvu_049_flattened'.")
    ap.add_argument('--out', type=Path, default=Path('datasets/SurgVU/idle_clips'), help="Output directory for video clips.")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Saving clips to: {args.out.resolve()}")

    csv_files = _expand_paths(args.csv, "*.csv")
    if not csv_files:
        print("[Error] No CSV files found.")
        return

    for csv_path in csv_files:
        print(f"\nProcessing CSV: {csv_path.name}")
        video_path = get_video_path(csv_path, args.video_dir)

        if not video_path:
            print(f"  [Warning] No matching video found for {csv_path.name} in {args.video_dir}. Skipping.")
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  [Error] Could not open video file: {video_path}. Skipping.")
            continue
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_rate_ratio = video_fps / CSV_FPS
        print(f"  -> Found video: {video_path.name} ({video_fps} FPS, ratio: {frame_rate_ratio:.2f})")

        df_source = pd.read_csv(csv_path)
        if 'instrument' not in df_source.columns:
            df_source['instrument'] = 'unknown'

        # List to store info for the summary CSV for this video
        clip_metadata = []

        # Group by instrument and track_id to process each track individually
        for (instrument, track_id), df_track in df_source.groupby(['instrument', 'track_id']):
            if len(df_track) < MIN_TRACK_LEN:
                continue

            print(f"    - Analyzing track: {instrument} (ID: {track_id})")
            idle_periods = find_idle_periods(
                df_track=df_track,
                keypoint_idx=KEYPOINT_IDX,
                velocity_threshold=VELOCITY_THRESHOLD,
                min_idle_frames=MIN_IDLE_FRAMES,
                smoothing_window=SMOOTHING_WINDOW
            )

            if not idle_periods:
                print("      No significant idle periods found.")
                continue

            for start_f_csv, end_f_csv in idle_periods:
                # Scale frame numbers to match the video's frame rate
                start_f_vid = int(start_f_csv * frame_rate_ratio)
                end_f_vid = int(end_f_csv * frame_rate_ratio)

                # Check if the final clip duration meets the minimum requirement
                clip_duration_seconds = (end_f_vid - start_f_vid + 1) / video_fps
                if clip_duration_seconds < MIN_CLIP_DURATION_SECONDS:
                    print(f"      - Skipping short clip ({clip_duration_seconds:.2f}s < {MIN_CLIP_DURATION_SECONDS}s)")
                    continue

                clip_name = f"{video_path.stem}_{instrument}_{track_id}_{start_f_csv}-{end_f_csv}.mp4"
                out_path = args.out / clip_name
                write_video_clip(cap, start_f_vid, end_f_vid, out_path)

                # Record metadata for the summary CSV
                start_ts = start_f_vid / video_fps
                end_ts = end_f_vid / video_fps
                clip_metadata.append({
                    'clip_name': clip_name,
                    'start_timestamp': f"{start_ts:.3f}",
                    'end_timestamp': f"{end_ts:.3f}"
                })

        cap.release()

        # Write the summary CSV for the processed video
        if clip_metadata:
            summary_csv_path = args.out / f"{video_path.stem}_clips.csv"
            df_summary = pd.DataFrame(clip_metadata)
            df_summary.to_csv(summary_csv_path, index=False)
            print(f"  ✓ Saved clip summary: {summary_csv_path.name}")

    print("\n✓ Processing complete.")

if __name__ == '__main__':
    main()
