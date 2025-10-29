#!/usr/bin/env python3
import os
import re
import sys
import json
import math
import shutil
import tempfile
import subprocess
from datetime import datetime
from tkinter import Tk, filedialog, messagebox

# -----------------------------
# Configuration / Defaults
# -----------------------------
VIDEO_PATTERNS = ("*.mp4", "*.mov", "*.mkv", "*.avi", "*.m4v", "*.mts", "*.m2ts")
TS_REGEX = re.compile(r'(\d{8})[_-]?(\d{6})')  # e.g., 20250101_123456 or 20250101-123456
TARGET_FPS = 30
CRF = "20"
X264_PRESET = "veryfast"
X264_PROFILE = "high"
X264_LEVEL = "4.1"  # safe for up to 4K@30 (depending on refs/b-frames)
AUDIO_BITRATE = "192k"
AUDIO_RATE = "48000"
AUDIO_CHANNELS = "2"

# Optional: cap extreme resolutions (set to None to disable). Values must be even.
MAX_WIDTH = None   # e.g., 3840
MAX_HEIGHT = None  # e.g., 2160


# -----------------------------
# Utilities
# -----------------------------
def run_ok(cmd):
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def which_ok(name):
    from shutil import which
    return which(name) is not None

def ffprobe_json(path, args):
    cmd = ["ffprobe", "-v", "error", "-print_format", "json"] + args + [path]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}:\n{r.stderr}")
    return json.loads(r.stdout or "{}")

def extract_timestamp(path):
    base = os.path.basename(path)
    m = TS_REGEX.search(base)
    if m:
        s = f"{m.group(1)}{m.group(2)}"
        try:
            return datetime.strptime(s, "%Y%m%d%H%M%S")
        except ValueError:
            pass
    try:
        return datetime.fromtimestamp(os.path.getmtime(path))
    except Exception:
        return datetime.min

def even(n):
    return int(math.floor(n / 2.0) * 2)

def choose_files():
    root = Tk()
    root.withdraw()
    filetypes = [("Video files", " ".join(VIDEO_PATTERNS)), ("All files", "*.*")]
    paths = filedialog.askopenfilenames(title="Select videos to combine", filetypes=filetypes)
    root.update()
    return list(paths)

def choose_output(initial_dir, default_name="combined.mp4"):
    out = filedialog.asksaveasfilename(
        title="Save combined video as",
        defaultextension=".mp4",
        filetypes=[("MP4 Video", "*.mp4")],
        initialfile=default_name,
        initialdir=initial_dir
    )
    if out and not out.lower().endswith(".mp4"):
        out += ".mp4"
    return out

def get_video_info(path):
    """Return dict with width, height, rotation, has_audio."""
    data = ffprobe_json(
        path,
        ["-select_streams", "v:0",
         "-show_entries", "stream=width,height,avg_frame_rate,side_data_list:stream_tags=rotate",
         "-show_streams"]
    )

    width = height = None
    rotation = 0
    has_audio = False

    # Video stream info
    for s in data.get("streams", []):
        if s.get("codec_type") == "video" and width is None:
            width = int(s.get("width") or 0)
            height = int(s.get("height") or 0)
            # Rotation can be in tags.rotate or side_data_list
            rotate_tag = None
            if "tags" in s and "rotate" in s["tags"]:
                rotate_tag = s["tags"]["rotate"]
            elif "side_data_list" in s:
                for sd in s["side_data_list"]:
                    if "rotation" in sd:
                        rotate_tag = sd.get("rotation")
                        break
            try:
                if rotate_tag is not None:
                    rotation = int(rotate_tag) % 360
            except Exception:
                rotation = 0

        if s.get("codec_type") == "audio":
            has_audio = True

    if not width or not height:
        raise RuntimeError(f"Could not read dimensions for {path}")

    return {
        "width": width,
        "height": height,
        "rotation": rotation,
        "has_audio": has_audio,
    }

def decide_target_size(infos):
    """Pick a common target frame size (even dimensions); cap if configured."""
    max_w = max(i["width"] if i["rotation"] in (0,180) else i["height"] for i in infos)
    max_h = max(i["height"] if i["rotation"] in (0,180) else i["width"] for i in infos)

    if MAX_WIDTH and max_w > MAX_WIDTH:
        max_h = int(round(max_h * (MAX_WIDTH / max_w)))
        max_w = MAX_WIDTH
    if MAX_HEIGHT and max_h > MAX_HEIGHT:
        max_w = int(round(max_w * (MAX_HEIGHT / max_h)))
        max_h = MAX_HEIGHT

    return even(max_w), even(max_h)

def rotation_filter(rotation):
    if rotation == 90:
        return "transpose=clock"
    if rotation == 270:
        return "transpose=cclock"
    if rotation == 180:
        # 180 can be done via hflip+vflip
        return "hflip,vflip"
    return None

def build_filter_chain(rotation, tw, th):
    """
    Normalize:
      - apply rotation into pixels (and clear metadata)
      - scale to fit in target while preserving AR
      - pad to target size centered
      - set SAR/DAR
      - enforce CFR and pixel format for Encord
    """
    filters = []
    rf = rotation_filter(rotation)
    if rf:
        filters.append(rf)

    # scale to fit within target (even dims), then pad
    # force_original_aspect_ratio=decrease keeps AR; even size ensured by pad
    filters.append(f"scale={tw}:{th}:force_original_aspect_ratio=decrease")
    # center pad; ow/oh are target; iw/ih are post-scale
    filters.append(f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2:color=black")
    # Make storage square pixels, set display aspect to target
    filters.append("setsar=1")
    # Enforce CFR
    filters.append(f"fps={TARGET_FPS}")
    # Pixel format for broad compatibility
    filters.append("format=yuv420p")
    # Optional: fix DAR explicitly (not strictly necessary with setsar+pad)
    # filters.append(f"setdar={tw}/{th}")

    return ",".join(filters)

def write_concat_list(path_list, list_txt_path):
    # ffmpeg concat demuxer list file: one per line: file 'fullpath'
    with open(list_txt_path, "w", encoding="utf-8") as f:
        for p in path_list:
            # Escape single quotes if any (rare on Windows/macOS)
            q = p.replace("'", r"'\''")
            f.write(f"file '{q}'\n")


# -----------------------------
# Main flow
# -----------------------------
def main():
    # Basic availability checks
    if not which_ok("ffmpeg") or not which_ok("ffprobe"):
        messagebox.showerror("FFmpeg", "ffmpeg/ffprobe not found on PATH. Please install FFmpeg and try again.")
        return

    # Select inputs
    paths = choose_files()
    if not paths:
        messagebox.showwarning("Combine Videos", "No files selected.")
        return

    # Sort by timestamp in filename (fallback to mtime)
    paths = sorted(paths, key=extract_timestamp)

    # Choose output
    first_path = paths[0]
    base_no_ext = os.path.splitext(os.path.basename(first_path))[0]
    out_path = choose_output(
        initial_dir=os.path.dirname(first_path),
        default_name=f"{base_no_ext}_combined.mp4"
    )
    if not out_path:
        messagebox.showinfo("Combine Videos", "No output file selected. Aborting.")
        return

    # Probe inputs and decide common target size
    try:
        infos = [get_video_info(p) for p in paths]
    except Exception as e:
        messagebox.showerror("FFprobe Error", str(e))
        return

    tw, th = decide_target_size(infos)

    tmpdir = tempfile.mkdtemp(prefix="combine_clips_")
    try:
        # Normalize each clip to uniform MP4 (H.264/AAC) with CFR and consistent WxH
        norm_files = []
        for i, (inp, info) in enumerate(zip(paths, infos)):
            nf = os.path.join(tmpdir, f"clip_{i:04d}.mp4")
            vf = build_filter_chain(info["rotation"], tw, th)

            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", inp,
                # Burn rotation into pixels and clear metadata
                "-metadata:s:v:0", "rotate=0",
                "-vf", vf,
                "-c:v", "libx264",
                "-profile:v", X264_PROFILE,
                "-level", X264_LEVEL,
                "-pix_fmt", "yuv420p",
                "-preset", X264_PRESET,
                "-crf", CRF,
                "-r", str(TARGET_FPS),         # ensure CFR timing at mux
                "-vsync", "cfr",
                "-movflags", "+faststart",
            ]

            if info["has_audio"]:
                cmd += ["-c:a", "aac", "-b:a", AUDIO_BITRATE, "-ar", AUDIO_RATE, "-ac", AUDIO_CHANNELS]
            else:
                cmd += ["-an"]

            cmd.append(nf)
            subprocess.run(cmd, check=True)
            norm_files.append(nf)

        # Concat using concat demuxer with stream copy
        list_txt = os.path.join(tmpdir, "list.txt")
        write_concat_list(norm_files, list_txt)

        concat_cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", list_txt,
            "-c", "copy", "-movflags", "+faststart", out_path
        ]
        subprocess.run(concat_cmd, check=True)
        # Force the info dialog to the foreground and focus it
        try:
            root.attributes("-topmost", True)
            root.lift()
            root.focus_force()
        except Exception:
            pass

        messagebox.showinfo(
            "Combine Videos",
            f"Combined {len(paths)} video(s) into:\n{out_path}\n\n"
            f"Codec: H.264 (yuv420p), CFR {TARGET_FPS} fps, AAC {AUDIO_RATE} Hz.",
            parent=root
        )

        # Reset topmost so it doesn't affect future dialogs
        try:
            root.attributes("-topmost", False)
        except Exception:
            pass
    except subprocess.CalledProcessError as e:
        messagebox.showerror("FFmpeg Error", f"FFmpeg failed.\n{e}")
    except Exception as e:
        messagebox.showerror("Error", str(e))
    finally:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass


if __name__ == "__main__":
    # Keep Tkinter root alive for dialogs
    root = Tk()
    root.withdraw()
    try:
        main()
    finally:
        # Ensure the root is destroyed to prevent hanging processes on some platforms
        try:
            root.destroy()
        except Exception:
            pass
