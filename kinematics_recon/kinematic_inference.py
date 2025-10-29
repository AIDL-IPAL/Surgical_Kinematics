#!/usr/bin/env python3
"""
YOLO‑v8 pose inference (multi‑video batch) with multi‑keypoint tracking and optional monocular depth.

What's new vs earlier script
----------------------------
• **Multi‑input**: `--input` accepts 1..N paths. Each path may be a video file or a directory
    containing multiple videos. Each video is processed independently, producing its own output dir.
• **Per‑video outputs**: under `--out-root`, we create `<video_stem>/tracking_report.csv`,
    `<video_stem>/annotated.mp4`, and optional `frames_annot/` when `--vis` is set.
• **Instrument label wiring**: we persist a `tid → class` map so the CSV `instrument` column is
    correct for each track, not just a placeholder.
• Depth/pose models are loaded **once** and reused across the batch.

Example run cmd (Windows CMD)
--------
python pose_analytics/kinematic_inference.py ^
    --model pose_analytics/models/yolo11s_1018/weights/best.pt ^
    --input datasets/SurgVU/surgvu_eval_randomclips.mp4 ^
    --tracker botsort ^
    --save-video ^
    --depth --depth-model metric3d_vit_small ^
    --depth-ckpt scene3d/model_training/Metric3D_allhamlyn/model_ckpts/ckpt_20250513_epoch8.pth ^
    --intrinsics datasets/Northwell_acq/calibration/01/intrinsics.txt ^
    --sam2 ^
    --out-root datasets/SurgVU/tests ^

other quick dataset paths:
datasets/SurgVU/surgvu_case_0_49_flattened/case_010_video_part_002.mp4 (SurgVU validation set)
24 & 25 also in validation set

"""
from __future__ import annotations
import argparse, csv, os, shutil, sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Tuple, Dict, List, Optional

import cv2
from matplotlib import scale
import numpy as np
import torch
from tqdm import tqdm

# YOLO and helpers
from ultralytics import YOLO
import supervision as sv
from torchvision.ops import box_iou, box_convert

# Ensure project root is in sys.path for local imports
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Compatibility shim for supervision API changes: ensure utils.box_iou_batch exists
try:
    from supervision.detection.utils import box_iou_batch as _sv_box_iou_batch  # type: ignore
except Exception:
    try:
        from supervision.detection.core import box_iou_batch as _sv_box_iou_batch  # type: ignore
        import supervision.detection.utils as _sv_utils  # type: ignore
        setattr(_sv_utils, "box_iou_batch", _sv_box_iou_batch)
    except Exception:
        _sv_box_iou_batch = None  # type: ignore
del _sv_box_iou_batch

from trackers import SORTTracker  # or: from trackers import BYTETracker, OC_SORTTracker

# define local sort tracker
local_tracker = SORTTracker(
    lost_track_buffer=30,                # keep IDs alive across brief gaps
    frame_rate=15,                       # should match args.fps, will be set per-video
    track_activation_threshold=0.20,     # detection confidence for new tracks
    minimum_consecutive_frames=3,        # require a few hits before confirming an ID
    minimum_iou_threshold=0.25,          # IOU threshold for association
)

try:
    from scene3d.utils.dataprocess import (
        read_intrinsics,
    )
except (ModuleNotFoundError, ImportError):
    print("scene3d utils not found, depth will be disabled")
    metric3d_prep_img = metric3d_unpad_and_scale = postprocess_depth = None  # type: ignore
    read_intrinsics = None  # type: ignore

try:
    # --- SAM2 (segmentation) ---
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    import hydra
    from omegaconf import OmegaConf
except (ModuleNotFoundError, ImportError):
    print("SAM2 not found, segmentation will be disabled")
    build_sam2 = SAM2ImagePredictor = hydra = OmegaConf = None  # type: ignore


# ------------------------------- Kalman utils ---------------------------------
from kinematic_utils import robust_match, update_kf  # your existing helpers

KEYPOINT_NAMES = ['jaw_upper', 'jaw_lower', 'wrist-1', 'wrist-2', 'shaft', 'wrist-1_alt', 'wrist-2_alt']

CLS_NAMES = [
    "clip applier",
    "grasper",
    "cautery hook",
    "scissors",
    "sealer",
    "stapler"
]

# BGR color palette for keypoint visualization
KPT_COLOURS = [
    (0, 225, 255),   # Yellow
    (255, 170, 0),   # Blue
    (45, 255, 0),    # Green
    (255, 80, 80),   # Light Blue
    (160, 0, 255),   # Pink/Magenta
    (255, 255, 0),   # Cyan
    (0, 130, 255),   # Orange
]

VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.m4v', '.mpg'}

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------------------------------
# Depth helpers
# -----------------------------------------------------------------------------

def load_depth_model(framework: str, repo: str | None = None, ckpt: str | None = None):
    if not framework:
        raise ValueError("Depth framework name required when --depth is enabled")
    if framework.startswith("depth_anything"):
        sys.path.append(Path(__file__).parent / "Depth-Anything-V2")
        from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore
        encoder = framework.split("_", 2)[-1]  # vits | vitb | vitl
        cfg = {
            "vits": dict(encoder="vits", features=64, out_channels=[48, 96, 192, 384]),
            "vitb": dict(encoder="vitb", features=128, out_channels=[96, 192, 384, 768]),
            "vitl": dict(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024]),
        }[encoder]
        model = DepthAnythingV2(**cfg)
        if ckpt:
            model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        else:
            model.load_state_dict(torch.load(Path("Depth-Anything-V2/checkpoints") / f"depth_anything_v2_{encoder}.pth", map_location=device, weights_only=True))
        return model.to(device).eval(), False
    
    elif framework.startswith("metric3d"):
        # ---- Metric3D
        repo = repo or "yvanyin/metric3d"
        model = torch.hub.load(repo, framework, pretrain=(ckpt is None))
        if ckpt:
            model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True), strict=False)
            return model.to(device).eval(), True
        return model.to(device).eval(), True
    
    else:
        raise ValueError(f"Unsupported depth framework: {framework}")

def compute_depth_map_once(depth_estimator, frame, intrinsics, depth_scale) -> Optional[np.ndarray]:
    if depth_estimator is None:
        return None
    # infer_depth returns (pred_depth, crop_info)
    pred_depth = depth_estimator.infer_depth(frame, intrinsics, depth_scale)
    return pred_depth

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Batch YOLO pose + depth inference")
    p.add_argument("--model", required=True, help="*.pt pose checkpoint")
    p.add_argument("--input", nargs='+', required=True, help="1..N video files or directories containing videos")
    p.add_argument("--fps", type=int, default=15, help="target FPS (video mode)")
    p.add_argument("--out-root", default=None, help="root folder for per‑video outputs (default: model dir / test_results)")
    p.add_argument("--device", default="cuda", help="cuda / cpu / 0,1")
    p.add_argument("--vis", action="store_true", help="save annotated frames to frames_annot/")
    p.add_argument("--save-video", action="store_true", help="write annotated.mp4 for each input video")

    # tracking
    p.add_argument("--conf", type=float, default=0.20, help="score threshold")
    p.add_argument("--kpt-thres", type=float, default=0.15, help="key‑point confidence thr")
    p.add_argument("--iou-thres", type=float, default=0.25, help="NMS IoU threshold")
    p.add_argument("--min-track-frames", type=int, default=15, help="minimum frames to keep a track in CSV")
    p.add_argument("--tracker", default="sort", choices=["sort", "botsort", "bytetrack"], help="tracker to use")

    # masking, depth
    p.add_argument("--sam2", action="store_true", help="enable SAM2 segmentation from track-ID bbox prompts")
    p.add_argument("--sam2-root", default=None,
              help="Path to SAM2 repo root (the folder that contains the 'sam2/' package and 'checkpoints/').")
    p.add_argument("--sam2-cfg", default="sam2.1/sam2.1_hiera_b+.yaml", help="SAM2 yaml under the hydra config path")
    p.add_argument("--sam2-ckpt", default="external/sam2/checkpoints/sam2.1_hiera_base_plus.pt", help="SAM2 checkpoint path")
    p.add_argument("--depth", action="store_true", help="enable monocular depth inference")
    p.add_argument("--depth-model", default="metric3d_vit_small", help="metric3d_vit_small | depth_anything_v2_vits | …")
    p.add_argument("--depth-ckpt", default="scene3d/model_training/Metric3D_allhamlyn/model_ckpts/ckpt_20250513_epoch8.pth", help="optional fine‑tuned checkpoint")
    p.add_argument("--depth-repo", default="yvanyin/metric3d", help="hub repo (Metric3D)")
    p.add_argument("--intrinsics", default="datasets/Northwell_acq/04_08_2025/calibration/01/intrinsics.txt", help="fx fy cx cy text‑file (Metric3D Hamlyn-style)")
    p.add_argument("--depth-scale", type=float, default=1.0, help="global scaling factor")
    return p.parse_args()

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

def discover_videos(paths: Iterable[str | Path]) -> List[Path]:
    vids: List[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            vids.append(p)
        elif p.is_dir():
            vids.extend(sorted([q for q in p.iterdir() if q.suffix.lower() in VIDEO_EXTS]))
    return vids

def draw_dets(img, dets, thres=0.1):
    for det in dets:
        x, y, bw, bh = det["bbox"]
        p1 = int(x - bw/2), int(y - bh/2)
        p2 = int(x + bw/2), int(y + bh/2)
        cv2.rectangle(img, p1, p2, (0, 0, 255), 2)
        cv2.putText(img, f'TID {det["tid"]}: {det["cls"]} {det["conf"]:.2f}', (p1[0], p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        for kp_idx, (kx, ky, kc) in enumerate(det["keypoints_smooth"]):
            if kc < thres:
                continue
            kpt_pos = (int(kx), int(ky))
            color = KPT_COLOURS[kp_idx % len(KPT_COLOURS)]
            cv2.circle(img, kpt_pos, 2, color, -1)
            cv2.putText(img, str(kp_idx), (kpt_pos[0] + 3, kpt_pos[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    return img

def sam2_segment_tid_bbox(
    predictor,
    frame_bgr: np.ndarray,
    xyxy_trk,
    tids: Optional[Iterable[int]] = None,
) -> Dict[int, np.ndarray]:
    """
    Segment instruments with SAM2 using xyxy track boxes.

    Accepts:
      - xyxy_trk as:
          • Dict[int, array-like(4)] mapping tid -> [x1,y1,x2,y2], OR
          • array-like of shape (M,4) (torch.Tensor / np.ndarray / list of lists)
        If array-like is provided, tids can be optionally provided as an iterable
        of length M. If not provided, tids default to [0..M-1].

    Returns:
      dict {tid: (H,W) uint8 mask in {0,1}}
    """
    H, W = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(frame_rgb)

    tids_list: List[int] = []
    boxes_xyxy: List[List[float]] = []

    # Normalize input xyxy_trk into tids_list and boxes_xyxy
    if isinstance(xyxy_trk, dict):
        for tid, xyxy in xyxy_trk.items():
            x1, y1, x2, y2 = map(float, xyxy)
            # clip to image bounds
            x1 = max(0.0, min(W - 1.0, x1))
            y1 = max(0.0, min(H - 1.0, y1))
            x2 = max(0.0, min(W - 1.0, x2))
            y2 = max(0.0, min(H - 1.0, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            tids_list.append(int(tid))
            boxes_xyxy.append([x1, y1, x2, y2])
    else:
        # array-like path
        if torch.is_tensor(xyxy_trk):
            boxes = xyxy_trk.detach().cpu().numpy()
        else:
            boxes = np.asarray(xyxy_trk)
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            return {}

        boxes = boxes.astype(np.float32)
        M = boxes.shape[0]
        if tids is None or len(list(tids)) != M:
            tids_seq = list(range(M))
        else:
            tids_seq = list(tids)

        for i in range(M):
            x1, y1, x2, y2 = map(float, boxes[i])
            x1 = max(0.0, min(W - 1.0, x1))
            y1 = max(0.0, min(H - 1.0, y1))
            x2 = max(0.0, min(W - 1.0, x2))
            y2 = max(0.0, min(H - 1.0, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            tids_list.append(int(tids_seq[i]))
            boxes_xyxy.append([x1, y1, x2, y2])

    if not boxes_xyxy:
        return {}

    box_prompts = np.asarray(boxes_xyxy, dtype=np.float32)

    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box_prompts,
        multimask_output=False,
    )

    # normalize to (N, H, W)
    if masks.ndim == 4:      # (N, 1, H, W)
        masks = masks[:, 0]
    elif masks.ndim == 2:    # (H, W)
        masks = masks[None, ...]
    elif masks.ndim == 3 and masks.shape[:2] == (H, W):  # (H, W, ?) unlikely
        # If shape is (H,W,N) rotate; handle conservatively
        if masks.shape[2] == 1:
            masks = masks[..., 0][None, ...]
        else:
            # fallback: treat as a single mask
            masks = masks[None, ...]

    masks_bin = (masks > 0).astype(np.uint8)

    return {tid: masks_bin[i] for i, tid in enumerate(tids_list) if i < masks_bin.shape[0]}

def build_viz_item_from_xyxy(xyxy_row: np.ndarray, cls_idx: int, tid: int, conf: float = 1.0) -> Dict:
    xywh = box_convert(torch.from_numpy(xyxy_row[None, :]), in_fmt="xyxy", out_fmt="xywh").squeeze(0).numpy()
    return {"bbox": xywh, "tid": tid, "cls": CLS_NAMES[cls_idx] if 0 <= cls_idx < len(CLS_NAMES) else "unknown",
            "conf": conf, "keypoints": np.zeros((0, 3), dtype=float)}

class FrameWriter:
    """
    Handles drawing, saving annotated frames, and writing video for a sequence.
    Initialize once per video, then call write() for each frame.
    """
    def __init__(self, video_out: Path, video_path: Path, args):
        self.video_out = video_out
        self.video_path = video_path
        self.args = args
        self.vw = None
        self.frame_dir = video_out / "frames_annot"
        # ensure unique suffix if exists
        suffix = 1
        while self.video_out.exists():
            cand = f"{self.video_out.name}_{suffix}"
            if not cand.exists():
                self.video_out = cand
                break
            suffix += 1 
        if args.vis:
            (self.video_out / "frames_annot").mkdir(parents=True, exist_ok=True)

    def write(self, frame: np.ndarray, dets_for_viz: List[Dict], frame_idx: int, depth_frame: np.ndarray = None, sam_masks: Dict[int, np.ndarray] = None):
        annotated = draw_dets(frame.copy(), dets_for_viz, thres=self.args.kpt_thres)

        if sam_masks:
            # Overlay SAM2 masks with some transparency
            overlay = annotated.copy()
            alpha = 0.4
            for tid, mask in sam_masks.items():
                color = tuple(np.random.randint(0, 256, size=3).tolist())
                colored_mask = np.zeros_like(overlay, dtype=np.uint8)
                colored_mask[mask == 1] = color
                overlay = cv2.addWeighted(overlay, 1.0, colored_mask, alpha, 0)
            annotated = overlay

        # If depth is enabled and depth_frame is provided, mirror and stack side-by-side
        if self.args.depth and depth_frame is not None:
            depth_map_np = depth_frame.cpu().numpy() if isinstance(depth_frame, torch.Tensor) else depth_frame
            # Use a colormap to visualize depth (e.g., grey2jet or cv2.COLORMAP_JET)
            # Use the existing visualize_depth function for depth visualization

            # Create depth visualization without matplotlib (convert to 0..255 and apply colormap)
            finite_mask = np.isfinite(depth_map_np)
            if finite_mask.any():
                dmin = float(np.nanmin(depth_map_np))
                dmax = float(np.nanmax(depth_map_np))
                # print(f"dmin: {dmin}, dmax: {dmax}")
                if dmax - dmin < 1e-6:
                    dmax = dmin + 1e-6
                # TODO update normalization to use rolling min/max window or per-video numbers
                depth_clipped = np.where(finite_mask, np.clip(depth_map_np, 0, 1), dmin)
                depth_norm = (depth_clipped - 0) / (1 - 0)  # normalize to 0..1
            else:
                depth_norm = np.zeros_like(depth_map_np, dtype=np.float32)
            depth_u8 = (depth_norm * 255.0).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
            
            # Resize depth visualization to match annotated if needed
            if depth_vis.shape[:2] != annotated.shape[:2]:
                depth_vis = cv2.resize(depth_vis, (annotated.shape[1], annotated.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Draw detections on depth visualization as well
            depth_vis_annot = draw_dets(depth_vis.copy(), dets_for_viz, thres=self.args.kpt_thres)

            # Build a right-side colorbar (no matplotlib)
            Hc, Wc = depth_vis_annot.shape[:2]
            cb_w, txt_w = 40, 70

            # Compute display range from the current depth map
            finite_cb = np.isfinite(depth_map_np)
            if finite_cb.any():
                dmin_cb = float(np.nanmin(depth_map_np))
                dmax_cb = float(np.nanmax(depth_map_np))
                if dmax_cb - dmin_cb < 1e-6:
                    dmax_cb = dmin_cb + 1e-6
            else:
                dmin_cb, dmax_cb = 0.0, 1.0

            # Vertical gradient: top = max (hot), bottom = min (cold), matching COLORMAP_JET
            grad = np.linspace(255, 0, Hc, dtype=np.uint8)[:, None]
            cb_gray = np.repeat(grad, cb_w, axis=1)
            cb_color = cv2.applyColorMap(cb_gray, cv2.COLORMAP_JET)

            # Legend panel with tick labels
            legend = np.zeros((Hc, cb_w + txt_w, 3), dtype=np.uint8)
            legend[:, :cb_w] = cb_color
            cv2.rectangle(legend, (0, 0), (cb_w - 1, Hc - 1), (255, 255, 255), 1)

            # Ticks and labels
            n_ticks = 5
            for i in range(n_ticks):
                y = int(round(i * (Hc - 1) / (n_ticks - 1)))
                val = dmax_cb - (y / max(1, (Hc - 1))) * (dmax_cb - dmin_cb)
                cv2.line(legend, (cb_w - 4, y), (cb_w - 1, y), (255, 255, 255), 1)
                cv2.putText(
                    legend,
                    f"{val:.2f}",
                    (cb_w + 5, min(Hc - 5, max(12, y + 4))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # Title
            cv2.putText(
                legend,
                "Depth [mm]",
                (cb_w + 5, 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Attach legend to the right of the depth view
            depth_vis_annot = np.concatenate([depth_vis_annot, legend], axis=1)

            annotated = np.concatenate([annotated, depth_vis_annot], axis=1)

        if self.args.vis:
            cv2.imwrite(str(self.frame_dir / f"{self.video_path.stem}_{frame_idx:06d}.jpg"), annotated)

        if self.args.save_video:
            if self.vw is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                h, w = annotated.shape[:2]
                self.vw = cv2.VideoWriter(
                    str(self.video_out.parent / f"{self.video_out.name}_annotated.mp4"),
                    fourcc, self.args.fps, (w, h)
                )
            self.vw.write(annotated)

    def release(self):
        if self.vw is not None:
            self.vw.release()
            self.vw = None

# -----------------------------------------------------------------------------
# Per‑video processing
# -----------------------------------------------------------------------------

def process_video(video_path: Path, out_root: Path, pose_model: YOLO, args, depth_pack):
    """Process a video file and save the results."""

    # initialize tracker. see top of script for tracker options
    local_tracker.reset()
    tid_to_cls: Dict[int, str] = {}

    depth_estimator, preprocess_depth, intrinsics = depth_pack

    sam2_predictor = None
    if args.sam2:
        # Resolve the repo root (project_root/external/sam2)
        project_root = Path(__file__).resolve().parents[1]  # Go up two levels to project root
        default_root = project_root / "external" / "sam2"
        sam2_root = Path(args.sam2_root).resolve() if args.sam2_root else default_root

        # Make sure Python can import the 'sam2' package from this repo
        if str(sam2_root) not in sys.path:
            sys.path.insert(0, str(sam2_root))

        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize(config_path="../external/sam2/sam2/configs", version_base=None)
        cfg = hydra.compose(config_name="sam2.1/sam2.1_hiera_b+")
        
        # Your existing flags still work
        model_cfg = args.sam2_cfg  # e.g. "configs/sam2.1/sam2.1_hiera_b+.yaml"
        ckpt = args.sam2_ckpt or str(sam2_root / "checkpoints" / "sam2.1_hiera_base_plus.pt")

        sam2_model = build_sam2(
            model_cfg,
            ckpt,
            device=args.device if "cuda" in str(args.device).lower() else "cpu",
        )
        sam2_predictor = SAM2ImagePredictor(sam2_model)

    # output dir for this video
    video_out = out_root / video_path.stem
    print(f"\nProcessing video: {video_path}")
    writer = FrameWriter(video_out, video_path, args)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[!] Cannot open video: {video_path}")
        return None

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, int(round(source_fps / max(1, args.fps))))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    vw = None

    tracking_data: Dict[Tuple[int, int], List[Tuple]] = defaultdict(list)

    pbar = tqdm(total=(n_frames // step + 1), desc=f"infer:{video_path.stem}")
    frame_idx = -1  # Counting frame index in the video, final value is total number of frames
    while True:
        frame_idx += 1
        ret, frame = cap.read()
        if not ret:  # Arriving at the end of the video, cap.read() returns False
            break
        if frame_idx % step != 0:  # Skip frames to match the target FPS
            continue

        # --- Tracking / association entry point ---
        # 1) YOLO forward and tracker update
        if args.tracker == "botsort":
            result = pose_model.track(frame, conf=args.conf, iou=args.iou_thres,
                                      device=args.device, verbose=False, persist=True,
                                      tracker='pose_analytics/trackers/botsort.yaml')[0]

            xyxy_ul = result.boxes.xyxy.detach().cpu().float()  # (N,4)
            ids_t = getattr(result.boxes, "id", None)
            if ids_t is None:
                # No IDs returned; treat as zero tracks
                tid_arr = np.empty((0,), dtype=int)
                xyxy_trk = torch.empty((0, 4), dtype=torch.float32)
                M, N = 0, int(xyxy_ul.shape[0])
            else:
                ids_np = ids_t.detach().cpu().numpy().astype(float)  # may contain NaNs
                valid = ~np.isnan(ids_np)
                tid_arr = ids_np[valid].astype(int)
                # Select only the detections that have valid IDs to form the tracked set
                xyxy_trk = xyxy_ul[torch.from_numpy(valid)]
                M, N = int(xyxy_trk.shape[0]), int(xyxy_ul.shape[0])
            using_external_ids = False

        elif args.tracker == "bytetrack":
            result = pose_model.track(frame, conf=args.conf, iou=args.iou_thres,
                                      device=args.device, verbose=False, persist=True,
                                      tracker='pose_analytics/trackers/bytetrack.yaml')[0]

            xyxy_ul = result.boxes.xyxy.detach().cpu().float()  # (N,4)
            ids_t = getattr(result.boxes, "id", None)
            if ids_t is None:
                # No IDs returned; treat as zero tracks
                tid_arr = np.empty((0,), dtype=int)
                xyxy_trk = torch.empty((0, 4), dtype=torch.float32)
                M, N = 0, int(xyxy_ul.shape[0])
            else:
                ids_np = ids_t.detach().cpu().numpy().astype(float)  # may contain NaNs
                valid = ~np.isnan(ids_np)
                tid_arr = ids_np[valid].astype(int)
                # Select only the detections that have valid IDs to form the tracked set
                xyxy_trk = xyxy_ul[torch.from_numpy(valid)]
                M, N = int(xyxy_trk.shape[0]), int(xyxy_ul.shape[0])
            using_external_ids = False
        
        else: # args.tracker == "sort": use external tracker for IDs
            result = pose_model.predict(frame, conf=args.conf, iou=args.iou_thres,
                                        device=args.device, verbose=False)[0]

            dets_sv  = sv.Detections.from_ultralytics(result)
            tracked  = local_tracker.update(dets_sv)    # tracked.xyxy, tracked.class_id, tracked.tracker_id
            tid_arr  = getattr(tracked, "tracker_id", None)
            tid_arr  = tid_arr if tid_arr is not None else np.empty((0,), dtype=int)

            xyxy_trk = torch.from_numpy(tracked.xyxy).float()   # (M,4)
            xyxy_ul  = result.boxes.xyxy.detach().cpu().float() # (N,4)
            M, N     = int(xyxy_trk.shape[0]), int(xyxy_ul.shape[0])
            using_external_ids = True

        # 2) Depth once per frame
        depth_map_np = compute_depth_map_once(depth_estimator, frame, intrinsics, args.depth_scale)
        depth_map_np = depth_map_np if depth_map_np is not None else None

        # 3) Prepare outputs shared by all cases
        visible_tids: set[int] = set()
        dets_for_viz: List[Dict] = []
        vw_state = {"vw": vw}   # capture writer so we can update it

        # 4) Case A: no tracks → just write raw frame (no viz overlays)
        if M == 0:
            writer.write(frame, dets_for_viz, frame_idx, depth_map_np)
            vw = vw_state["vw"]; pbar.update(1); continue

        # 5) Case B: tracks exist but YOLO has 0 detections → draw tracker boxes only
        if N == 0:
            for m in range(M):
                if m >= len(tid_arr): continue
                tid     = int(tid_arr[m])
                cls_idx = int(tracked.class_id[m]) if tracked.class_id is not None else 0
                visible_tids.add(tid)
                dets_for_viz.append(build_viz_item_from_xyxy(tracked.xyxy[m], cls_idx, tid, conf=1.0))
            writer.write(frame, dets_for_viz, frame_idx, depth_map_np)
            vw = vw_state["vw"]; pbar.update(1); continue

        # 6) Case C: both have content → associate once, then reuse
        IoU    = box_iou(xyxy_trk, xyxy_ul)                     # (M,N)
        assign = IoU.argmax(dim=1)                              # best YOLO det for each track
        valid  = (IoU.max(dim=1).values >= 0.20)

        # (optional) SAM2 masks once, keyed by tracker tid
        if sam2_predictor is not None and M > 0:
            sam_masks = sam2_segment_tid_bbox(sam2_predictor, frame, xyxy_trk)
        else:
            sam_masks = {}

        # 7) Iterate associated pairs (no shape errors; no repeated depth calls)
        for m in range(M):
            if not bool(valid[m]) or m >= len(tid_arr):
                continue
            det_i = int(assign[m]); tid = int(tid_arr[m]); visible_tids.add(tid)

            # class & conf
            cls_idx = int(result.boxes.cls[det_i]) if result.boxes.cls is not None else (
                    int(tracked.class_id[m]) if using_external_ids and tracked.class_id is not None else 0)
            conf    = float(result.boxes.conf[det_i]) if result.boxes.conf is not None else 1.0

            # keypoints
            if result.keypoints is not None and len(result.keypoints) > det_i:
                kpts = result.keypoints.data[det_i].detach().cpu().numpy()  # (K,3)
            else:
                kpts = np.zeros((0,3), dtype=float)

            # snap KPs to SAM2 mask (once per det) if available
            if tid in sam_masks and kpts.size:
                inv = (1 - sam_masks[tid].astype(np.uint8))
                dist_transform = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
                for kp_idx, (kx, ky, kc) in enumerate(kpts):
                    if kc < args.kpt_thres: continue
                    xi, yi = int(round(kx)), int(round(ky))
                    if 0 <= yi < dist_transform.shape[0] and 0 <= xi < dist_transform.shape[1] and sam_masks[tid][yi, xi] == 0:
                        win = 25
                        y0, y1 = max(0, yi - win), min(dist_transform.shape[0], yi + win + 1)
                        x0, x1 = max(0, xi - win), min(dist_transform.shape[1], xi + win + 1)
                        sub = dist_transform[y0:y1, x0:x1]
                        my, mx = np.unravel_index(int(np.argmin(sub)), sub.shape)
                        kpts[kp_idx, 0] = float(x0 + mx); kpts[kp_idx, 1] = float(y0 + my)

            # per-kp filter + depth sampling using the single depth_map_np
            dt = 1.0 / max(1, args.fps)
            for kp_idx, (kx, ky, kc) in enumerate(kpts):
                if kc < args.kpt_thres: continue
                fx, fy = update_kf(tid * 10 + kp_idx, kp_idx, (kx, ky), float(kc), dt=dt)[0]
                depth_val = float('nan')
                if depth_map_np is not None:
                    xi, yi = int(round(kx)), int(round(ky))
                    if 0 <= yi < depth_map_np.shape[0] and 0 <= xi < depth_map_np.shape[1]:
                        depth_val = float(depth_map_np[yi, xi])

                vx = vy = vz = float('nan')
                hist = tracking_data.get((tid, kp_idx))
                if hist:
                    _, px, py, pd, *_ = hist[-1]
                    vx = (fx - px) / dt; vy = (fy - py) / dt
                    if not np.isnan(depth_val) and not np.isnan(pd): vz = (depth_val - pd) / dt

                # Store tracking data with FK-smoothed keypoints
                tracking_data[(tid, kp_idx)].append((frame_idx, float(fx), float(fy), depth_val, float(kc), vx, vy, vz))

            # Prepare FK-smoothed keypoints for visualization
            kpts_fk = np.array([
                [float(update_kf(tid * 10 + kp_idx, kp_idx, (kx, ky), float(kc), dt=dt)[0][0]),
                 float(update_kf(tid * 10 + kp_idx, kp_idx, (kx, ky), float(kc), dt=dt)[0][1]),
                 float(kc)]
                for kp_idx, (kx, ky, kc) in enumerate(kpts)
            ]) if kpts.size else np.zeros((0, 3), dtype=float)

            box_xywh = result.boxes.xywh[det_i].detach().cpu().numpy()
            dets_for_viz.append({
                "bbox": box_xywh,
                "tid": tid,
                "cls": CLS_NAMES[cls_idx] if 0 <= cls_idx < len(CLS_NAMES) else "unknown",
                "conf": conf,
                "keypoints": kpts,
                "keypoints_smooth": kpts_fk
            })

        # 8) Final draw/write for this frame (single place)
        writer.write(frame, dets_for_viz, frame_idx, depth_map_np, sam_masks)
        vw = vw_state["vw"]
        
        pbar.update(1)

    pbar.close()
    cap.release()
    if vw is not None:
        vw.release()

    # CSV save
    csv_path = video_out.parent / f"{video_out.name}_tracking_report.csv"
    print(f"Writing CSV report to: {csv_path}")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["track_id", "instrument", "kp_idx", "frame_id", "x", "y", "depth", "confidence", "vx", "vy", "vz", "track_length"])
        for (tid, kp_idx), items in tracking_data.items():
            if len(items) < args.min_track_frames:
                continue
            instr = tid_to_cls.get(tid, "unknown")
            track_len = len(items)
            for frame_id, x, y, depth, conf, vx, vy, vz in items:
                writer.writerow([tid, instr, kp_idx, frame_id, x, y, depth, conf, vx, vy, vz, track_len])
    print(f"✓ {video_path.name}: CSV → {csv_path}")
    if args.save_video:
        print(f"✓ {video_path.name}: video → {video_out/'annotated.mp4'}")
    return video_out

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")

    model_path = Path(args.model).resolve()
    out_root = Path(args.out_root) if args.out_root else model_path.parent.parent / "test_results"
    out_root.mkdir(parents=True, exist_ok=True)

    # Load models once
    pose_model = YOLO(str(model_path))
    pose_model.fuse()

    depth_estimator = None
    preprocess_depth = False
    intrinsic = None
    if args.depth:
        from scene3d.metric3d_utils import DepthEstimator
        # Initialize depth using DepthEstimator
        model_type = 'vit' if 'vit' in str(args.depth_model).lower() else 'convnext'
        depth_estimator = DepthEstimator(
            model_type=model_type,
            checkpoint_path=(args.depth_ckpt or None),
            device=args.device if 'cuda' in str(args.device).lower() or str(args.device).isdigit() else 'cpu'
        )
        # depth_model = depth_estimator.model
        try:
            intrinsics = read_intrinsics(args.intrinsics) if args.intrinsics else [1.0, 1.0, 0.0, 0.0]
            intrinsics = [float(x) for x in intrinsics] # fx, fy, cx, cy
            print(f"✓ Read intrinsics: fx={intrinsics[0]}, fy={intrinsics[1]}, cx={intrinsics[2]}, cy={intrinsics[3]}")
        except Exception as e:
            print(f"[!] Failed to read intrinsics: {e}")
        print("✓ depth model loaded (preprocess=" + str(preprocess_depth) + ")")

    videos = discover_videos(args.input)
    if not videos:
        print("[!] No videos found in inputs.")
        return

    print(f"Found {len(videos)} video(s). Outputs will be written under: {out_root}")

    for v in videos:
        process_video(v, out_root, pose_model, args, (depth_estimator, preprocess_depth, intrinsics))

    print("✓ Batch complete.")

if __name__ == "__main__":

    main()
