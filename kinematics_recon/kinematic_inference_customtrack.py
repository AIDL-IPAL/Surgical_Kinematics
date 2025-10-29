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

Examples (Windows CMD)
--------
python pose_analytics/kinematic_inference.py ^
    --model pose_analytics/models/yolo11s_0907/weights/best.pt ^
    --input datasets/Northwell_acq/cholec_08_06_2025/camera0_20250806_171939_chunk.avi ^
    --save-video --out-root datasets/SurgVU/surgvu_049_inference ^
    --sam2
    --depth --depth-model metric3d_vit_small --depth-ckpt ckpt.pth --intrinsics data/intrinsics.txt
"""
from __future__ import annotations
import argparse, csv, os, shutil, sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

# ------------------------------- optional Metric3D imports --------------------
try:
    from scene3d.utils.dataprocess import (
        metric3d_prep_img,
        metric3d_unpad_and_scale,
        postprocess_depth,
    )
    from scene3d.model_training.utils.hamlyn_intrinsics import read_intrinsics
except (ModuleNotFoundError, ImportError):
    print("Metric3D not found, depth will be disabled")
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
    # ---- Metric3D
    repo = repo or "yvanyin/metric3d"
    model = torch.hub.load(repo, framework, pretrain=(ckpt is None))
    if ckpt:
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True), strict=False)
    return model.to(device).eval(), True


def infer_depth(model, frame_bgr: np.ndarray, preprocess: bool, intrinsic, global_scale: float | None = None):
    with torch.no_grad():
        if preprocess:
            rgb_t, pad_info = metric3d_prep_img(frame_bgr, intrinsic, device=device)  # type: ignore
            pred, *_ = model({"input": rgb_t})
            pred = metric3d_unpad_and_scale(pred, pad_info, intrinsic, frame_bgr.shape[:2])  # type: ignore
        else:
            rgb_np = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)[None].astype(np.float32) / 255.0
            rgb_t = torch.from_numpy(rgb_np).permute(0, 3, 1, 2).to(device)
            pred = model(rgb_t)
        pred = pred.squeeze(0)
        if pred.ndim == 3:
            pred = pred.squeeze(0)
        if global_scale is not None and postprocess_depth is not None:
            pred = postprocess_depth(pred.unsqueeze(0), scale=global_scale).squeeze(0)  # type: ignore
        return pred

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
    p.add_argument("--conf", type=float, default=0.30, help="score threshold") # TODO
    p.add_argument("--kpt-thres", type=float, default=0.15, help="key‑point confidence thr")
    p.add_argument("--iou-thres", type=float, default=0.35, help="NMS IoU threshold")
    p.add_argument("--min-track-frames", type=int, default=15, help="minimum frames to keep a track in CSV")

    # masking, depth
    p.add_argument("--sam2", action="store_true", help="enable SAM2 segmentation from track-ID bbox prompts")
    p.add_argument("--sam2-root", default=None,
              help="Path to SAM2 repo root (the folder that contains the 'sam2/' package and 'checkpoints/').")
    p.add_argument("--sam2-cfg", default="sam2.1/sam2.1_hiera_b+.yaml", help="SAM2 yaml under the hydra config path")
    p.add_argument("--sam2-ckpt", default="external/sam2/checkpoints/sam2.1_hiera_base_plus.pt", help="SAM2 checkpoint path")
    p.add_argument("--depth", action="store_true", help="enable monocular depth inference")
    p.add_argument("--depth-model", default="metric3d_vit_small", help="metric3d_vit_small | depth_anything_v2_vits | …")
    p.add_argument("--depth-ckpt", default=None, help="optional fine‑tuned checkpoint")
    p.add_argument("--depth-repo", default="yvanyin/metric3d", help="hub repo (Metric3D)")
    p.add_argument("--intrinsics", default=None, help="fx fy cx cy text‑file (Metric3D)")
    p.add_argument("--depth-scale", type=float, default=None, help="global scaling factor")
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

        for kp_idx, (kx, ky, kc) in enumerate(det["keypoints"]):
            if kc < thres:
                continue
            kpt_pos = (int(kx), int(ky))
            color = KPT_COLOURS[kp_idx % len(KPT_COLOURS)]
            cv2.circle(img, kpt_pos, 2, color, -1)
            cv2.putText(img, str(kp_idx), (kpt_pos[0] + 3, kpt_pos[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return img

def sam2_segment_from_tids(predictor: "SAM2ImagePredictor",
                           frame_bgr: np.ndarray,
                           tid2xywh: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    """
    Segment instruments with SAM2 using bbox prompts from current track IDs.

    tid2xywh: {tid: [cx, cy, w, h] in pixels}
    returns:  {tid: (H,W) uint8 mask in {0,1}}
    """
    if not tid2xywh:
        return {}

    H, W = frame_bgr.shape[:2]
    # BGR -> RGB, SAM2 expects RGB
    frame_rgb = frame_bgr[:, :, ::-1].copy()
    predictor.set_image(frame_rgb)

    tids, boxes_xyxy = [], []
    for tid, xywh in tid2xywh.items():
        cx, cy, w, h = map(float, xywh)
        x1 = max(0, int(round(cx - w / 2.0)))
        y1 = max(0, int(round(cy - h / 2.0)))
        x2 = min(W - 1, int(round(cx + w / 2.0)))
        y2 = min(H - 1, int(round(cy + h / 2.0)))
        if x2 <= x1 or y2 <= y1:
            continue
        tids.append(tid)
        boxes_xyxy.append([x1, y1, x2, y2])

    if not boxes_xyxy:
        return {}

    box_prompts = np.asarray(boxes_xyxy, dtype=np.int32)

    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box_prompts,
        multimask_output=False,
    )
    # Normalize to shape (N, H, W)
    if masks.ndim == 4:        # (N, 1, H, W)
        masks = masks[:, 0, :, :]
    elif masks.ndim == 2:      # (H, W)
        masks = masks[None, ...]
    elif masks.ndim == 3 and masks.shape[0] == H and masks.shape[1] == W:
        masks = masks[None, ...]  # (H,W,N) is unlikely; handle conservatively

    masks_bin = (masks > 0).astype(np.uint8)

    return {tid: masks_bin[i] for i, tid in enumerate(tids)}

# -----------------------------------------------------------------------------
# Per‑video processing
# -----------------------------------------------------------------------------

def process_video(video_path: Path, out_root: Path, pose_model: YOLO, args, depth_pack):
    depth_model, preprocess_depth, intrinsic = depth_pack

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
    # ensure unique suffix if exists
    suffix = 1
    while video_out.exists():
        cand = out_root / f"{video_out.name}_{suffix}"
        if not cand.exists():
            video_out = cand
            break
        suffix += 1
    (video_out / "frames_annot").mkdir(parents=True, exist_ok=True) if args.vis else None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[!] Cannot open video: {video_path}")
        return None

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, int(round(source_fps / max(1, args.fps))))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    vw = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") if args.save_video else None

    # tracking state (reset per video)
    prev_boxes = torch.empty((0, 4), device=device)
    prev_ids: List[int] = []
    track_state: Dict[int, Dict] = {}
    tid_to_cls: Dict[int, str] = {}
    next_tid = 0

    def new_tid() -> int:
        nonlocal next_tid
        tid = next_tid
        next_tid += 1
        track_state[tid] = {"age": 0}
        return tid

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
        h, w = frame.shape[:2]

        # Only use YOLO predict here as tracking is handled by robust_match later
        result = pose_model.predict(frame, conf=args.conf, iou=args.iou_thres, 
                                    device=args.device, verbose=False)[0]

        # Association
        det_boxes = result.boxes.xywh.to(device)  # (N,4)
        matches = robust_match(det_boxes, prev_boxes, prev_ids, track_state, iou_thr=0.35, centre_thr=350, max_miss=15)
        prev_boxes = det_boxes.clone()
        prev_ids = [matches.get(di, new_tid()) for di in range(len(det_boxes))]
        
        # --- annotate instrument masks (place this BEFORE you build `dets` and call draw_dets) ---
        if sam2_predictor is not None:
            # det_boxes are xywh; prev_ids are the matched tids for each detection index
            tid2xywh = {
                int(tid): result.boxes.xywh[det_i].detach().cpu().numpy()
                for det_i, tid in enumerate(prev_ids)
            }
            sam_masks = sam2_segment_from_tids(sam2_predictor, frame, tid2xywh)

            if sam_masks:  # dict truthiness -> False if empty
                # 1) make a single color layer for all masks
                mask_viz = np.zeros_like(frame)  # BGR
                for tid, mask in sam_masks.items():
                    color = KPT_COLOURS[tid % len(KPT_COLOURS)]  # BGR tuple
                    mask_viz[mask.astype(bool)] = color

                # 2) alpha blend once (avoid double-tinting)
                frame = cv2.addWeighted(frame, 1.0, mask_viz, 0.35, 0.0)

                # (optional) crisp outlines to improve visibility
                for tid, mask in sam_masks.items():
                    color = KPT_COLOURS[tid % len(KPT_COLOURS)]
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(frame, contours, -1, color, 1)

                # (optional) HUD badge
                cv2.putText(frame, "Segmentation Active", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Constrain keypoints to their corresponding instrument masks
                for det_i, tid in enumerate(prev_ids):
                    if tid in sam_masks:  # Check if this track has a mask
                        mask = sam_masks[tid]
                        # For each keypoint, assign to nearest mask pixel if outside mask
                        kpts = result.keypoints.data[det_i]
                        for kp_idx in range(kpts.shape[0]):
                            kx, ky, kc = kpts[kp_idx].cpu().numpy()
                            if kc >= args.kpt_thres:
                                # Check if keypoint falls outside the mask
                                if 0 <= int(ky) < mask.shape[0] and 0 <= int(kx) < mask.shape[1]:
                                    if mask[int(ky), int(kx)] == 0:  # Outside mask
                                        # Find nearest mask point
                                        # Calculate distance transform
                                        dist_transform = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 3)
                                        # Create a mask with keypoint location
                                        kpt_mask = np.zeros_like(mask)
                                        kpt_mask[int(ky), int(kx)] = 1
                                        # Find minimum distance point (set non-mask points to max value)
                                        inverted_mask = np.ones_like(mask) * np.inf
                                        inverted_mask[mask == 1] = dist_transform[mask == 1]
                                        min_dist_val = np.min(inverted_mask[inverted_mask < np.inf])
                                        min_dist_loc = np.where((dist_transform == min_dist_val) & (mask == 1))
                                        if len(min_dist_loc[0]) > 0:
                                            # Take the first nearest point
                                            new_y, new_x = min_dist_loc[0][0], min_dist_loc[1][0]
                                            
                                            # Clone keypoints before modifying to avoid inference mode error
                                            modified_kpts = result.keypoints.data.clone()
                                            modified_kpts[det_i, kp_idx, 0] = float(new_x)
                                            modified_kpts[det_i, kp_idx, 1] = float(new_y)
                                            result.keypoints.data = modified_kpts

        # Depth (per frame)
        depth_map_np = None
        if depth_model is not None:
            depth_map = infer_depth(depth_model, frame, preprocess_depth, intrinsic, args.depth_scale)
            depth_map_np = depth_map.squeeze().detach().cpu().numpy()

        # Per‑kp update
        for det_i, tid in enumerate(prev_ids):
            cls_name = CLS_NAMES[int(result.boxes.cls[det_i])] if len(result.boxes.cls) > 0 else "unknown"
            tid_to_cls[tid] = cls_name
            kpts = result.keypoints.data[det_i]
            for kp_idx, (kx, ky, kc) in enumerate(kpts.cpu().numpy()):
                if kc < args.kpt_thres:
                    continue
                dt_sec = 1.0 / max(1, args.fps)
                filt_xy, _ = update_kf(tid * 10 + kp_idx, kp_idx, (kx, ky), float(kc), dt=dt_sec)
                
                # Calculate velocity
                vx, vy, vz = float('nan'), float('nan'), float('nan')
                depth_val = float(depth_map_np[int(ky), int(kx)]) if depth_map_np is not None and 0 <= int(ky) < h and 0 <= int(kx) < w else float('nan')
                track_history = tracking_data.get((tid, kp_idx))
                if track_history:
                    # prev_frame_idx, prev_x, prev_y, prev_depth, *_
                    _, prev_x, prev_y, prev_depth, *_ = track_history[-1]
                    vx = (filt_xy[0] - prev_x) / dt_sec
                    vy = (filt_xy[1] - prev_y) / dt_sec
                    if not np.isnan(depth_val) and not np.isnan(prev_depth):
                        vz = (depth_val - prev_depth) / dt_sec

                tracking_data[(tid, kp_idx)].append((frame_idx, float(filt_xy[0]), float(filt_xy[1]), depth_val, float(kc), vx, vy, vz))

        # Save visuals
        if args.vis or args.save_video:
            dets = []
            for det_i, tid in enumerate(prev_ids):
                bbox = result.boxes.xywh[det_i].detach().cpu().numpy()
                conf = float(result.boxes.conf[det_i]) if hasattr(result.boxes, 'conf') else 0.0
                cls_idx = int(result.boxes.cls[det_i]) if hasattr(result.boxes, 'cls') else 0
                kpts = result.keypoints.data[det_i].detach().cpu().numpy()
                dets.append({"bbox": bbox, "tid": tid, "cls": CLS_NAMES[cls_idx], "conf": conf, "keypoints": kpts})
            annotated = draw_dets(frame.copy(), dets, thres=args.kpt_thres)
            
            # Draw Kalman filtered trajectories
            for (tid, kp_idx), points in tracking_data.items():
                if tid in prev_ids:  # Only draw for visible tracks
                    # Get last N points to avoid cluttering
                    trajectory_points = points[-30:]  # last 30 points
                    if len(trajectory_points) > 1:
                        color = KPT_COLOURS[kp_idx % len(KPT_COLOURS)]
                        # Draw trajectory line
                        for i in range(len(trajectory_points) - 1):
                            pt1 = (int(trajectory_points[i][1]), int(trajectory_points[i][2]))
                            pt2 = (int(trajectory_points[i+1][1]), int(trajectory_points[i+1][2]))
                            cv2.line(annotated, pt1, pt2, color, 1)
                        # Draw current filtered point
                        last_pt = (int(trajectory_points[-1][1]), int(trajectory_points[-1][2]))
                        cv2.circle(annotated, last_pt, 3, color, -1)
                
            if args.vis:
                cv2.imwrite(str(video_out / "frames_annot" / f"{video_path.stem}_{frame_idx:06d}.jpg"), annotated)
            if args.save_video:
                if vw is None:
                    vw = cv2.VideoWriter(str(video_out.parent / f"{video_out.name}_annotated.mp4"), fourcc, args.fps, (w, h))
                vw.write(annotated)
        pbar.update(1)

    pbar.close()
    cap.release()
    if vw is not None:
        vw.release()

    # CSV save
    csv_path = video_out.parent / f"{video_out.name}_tracking_report.csv"
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

    depth_model = None
    preprocess_depth = False
    intrinsic = None
    if args.depth:
        depth_model, preprocess_depth = load_depth_model(args.depth_model, args.depth_repo, args.depth_ckpt)
        if preprocess_depth:
            if read_intrinsics is None:
                raise RuntimeError("Metric3D utilities not available. Ensure utils & intrinsics are installed.")
            intrinsic = read_intrinsics(args.intrinsics) if args.intrinsics else [1, 1, 0, 0]
        print("✓ depth model loaded (preprocess=" + str(preprocess_depth) + ")")

    videos = discover_videos(args.input)
    if not videos:
        print("[!] No videos found in inputs.")
        return

    print(f"Found {len(videos)} video(s). Outputs will be written under: {out_root}")

    for v in videos:
        process_video(v, out_root, pose_model, args, (depth_model, preprocess_depth, intrinsic))

    print("✓ Batch complete.")

if __name__ == "__main__":
    main()
