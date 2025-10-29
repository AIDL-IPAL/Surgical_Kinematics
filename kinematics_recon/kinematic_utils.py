from __future__ import annotations

# utils/kinematics.py
import itertools, numpy as np
from filterpy.kalman import KalmanFilter

# -----------------  tuning parameters  ----------------
DT          = 1 / 15.0          # default Δt, overwritten per frame
SIGMA_A2    = 2000.0             # px²/s⁴ — Higher → More responsive, less smooth. Lower → Smoother, increased lag
PX_SIG_MIN  = 1.0               # px  (when kp conf == 1)
PX_SIG_MAX  = 10.0              # px  (when kp conf == 0)
CONF_THR    = 0.05              # drop updates below this

# --------------  internal globals  --------------------
_id_gen      = itertools.count()          # monotonically increasing IDs
_kf_bank     = {}                         # (track_id, kp_idx) → KF instance

# ------------------------------------------------------
def new_track_id() -> int:
    """Return a fresh integer track id."""
    return next(_id_gen)

def _make_kf(dt, x0, y0):
    """Create a 2-D constant-velocity Kalman filter."""
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1,0,dt,0],
                     [0,1,0,dt],
                     [0,0,1,0 ],
                     [0,0,0,1 ]], dtype=float)

    sa2 = SIGMA_A2
    q = sa2 * np.array([[dt**4/4,0,dt**3/2,0],
                        [0,dt**4/4,0,dt**3/2],
                        [dt**3/2,0,dt**2,0 ],
                        [0,dt**3/2,0,dt**2]], dtype=float)
    kf.Q = q
    kf.H = np.array([[1,0,0,0],
                     [0,1,0,0]], dtype=float)
    kf.P *= 1_000.
    kf.x = np.array([x0, y0, 0, 0], dtype=float)
    return kf

# ------------------------------------------------------
def update_kf(track_id:int, kp_idx:int, meas_xy, conf:float, dt:float):
    """
    Predict & (optionally) update the Kalman filter belonging to (track, kp).
    Returns the filtered position and covariance (2×2).
    """
    key = (track_id, kp_idx)
    if key not in _kf_bank:
        _kf_bank[key] = _make_kf(dt, *meas_xy)

    kf = _kf_bank[key]
    # adjust Δt if you sub-sample frames
    kf.F[0,2] = kf.F[1,3] = dt
    kf.predict()

    if conf >= CONF_THR:
        # confidence-dependent measurement noise
        px_sigma = PX_SIG_MIN + (1-conf)*(PX_SIG_MAX-PX_SIG_MIN)
        kf.R[:]  = np.diag([px_sigma**2]*2)
        kf.update(np.asarray(meas_xy, float))

    return kf.x[:2].copy(), kf.P[:2,:2].copy()

import torch
from torchvision.ops import box_iou      # ships with ultralytics' Torch stack

def _to_xyxy(boxes):
    """
    Accept either (x,y,w,h) -or- (x1,y1,x2,y2) and return *xyxy* tensor.
    """
    if boxes.numel() == 0:
        return boxes
    if (boxes[:,2] < boxes[:,0]).any():          # already xyxy
        return boxes
    out = boxes.clone()
    out[:,2] += out[:,0]   # x+w
    out[:,3] += out[:,1]   # y+h
    return out

def simple_iou_tracker(curr, prev, prev_ids, iou_thr=0.3, top_k=3):
    curr = _to_xyxy(curr);  prev = _to_xyxy(prev)
    matches, taken_prev = {}, torch.zeros(len(prev), dtype=bool)

    if curr.numel() and prev.numel():
        ious = box_iou(curr, prev)                     # (N,M)
        # For each curr det, look at its K best previous boxes
        topk_vals, topk_idx = torch.topk(ious, min(top_k, prev.shape[0]), dim=1)
        for i in range(len(curr)):
            for j in topk_idx[i]:
                j = j.item()
                if taken_prev[j]:          # already matched
                    continue
                if ious[i, j] >= iou_thr:
                    matches[i] = int(prev_ids[j])
                    taken_prev[j] = True
                    break
    return matches

# ---------------------------------------------------------
# robust_matcher.py  – IoU + centre-dist + keep-alive TTL
# ---------------------------------------------------------
import torch
from torchvision.ops import box_iou
# ---------------------------------------------------------
# robust_matcher.py – motion-aware Hungarian assignment with geometry/class gating
# ---------------------------------------------------------
import math
import numpy as np
import torch
from torchvision.ops import box_iou

try:
    from scipy.optimize import linear_sum_assignment
    import pandas as pd
    from typing import List, Tuple
except Exception:
    linear_sum_assignment = None  # fallback to greedy if SciPy absent


def xywh2xyxy(b: torch.Tensor) -> torch.Tensor:
    out = b.clone()
    out[:, 2] = out[:, 0] + out[:, 2]
    out[:, 3] = out[:, 1] + out[:, 3]
    return out


def _centers_and_diag(xywh: torch.Tensor):
    """Return centers (N,2) and box diagonals (N,) in pixels."""
    c = xywh[:, :2] + xywh[:, 2:] / 2.0
    d = torch.sqrt((xywh[:, 2] ** 2) + (xywh[:, 3] ** 2))
    return c, d


def robust_match( # TODO LOOK at other systems & consider bringing together.
    curr_xywh: torch.Tensor,
    prev_xywh: torch.Tensor,
    prev_ids: list[int],
    prev_state: dict,
    iou_thr: float = 0.25,
    centre_thr: float | None = None,   # if None, use dynamic (diag-based) gating
    max_miss: int = 30,
    # optional context for better persistence:
    curr_classes: np.ndarray | None = None,      # (N,)
    curr_keypoints: np.ndarray | None = None,    # (N,K,3) with conf in [:,:,2]
    kpt_conf_thres: float = 0.15,
    # cost weights:
    iou_coef: float = 1.0,
    dist_coef: float = 0.8,
    kpt_coef: float = 0.3,
    class_penalty: float = 1e3,
    diag_coef: float = 1.5,   # dynamic gate: allow ~1.5×box-diagonal for motion
    vel_ema: float = 0.8,     # velocity smoothing
):
    """
    Returns dict: det_index -> track_id
    Also updates prev_state in-place: age/hits/cls/velocity/last_kpts
    """
    matches: dict[int, int] = {}
    N = int(curr_xywh.shape[0])
    M = int(prev_xywh.shape[0])
    if N == 0 and M == 0:
        return matches

    # matrices
    ious = box_iou(xywh2xyxy(curr_xywh), xywh2xyxy(prev_xywh)) if (N and M) else torch.empty((N, M))
    c_curr, _ = _centers_and_diag(curr_xywh) if N else (torch.empty((0, 2)), None)
    c_prev, d_prev = _centers_and_diag(prev_xywh) if M else (torch.empty((0, 2)), None)

    # predict prev centers with simple velocity prior
    # prev_state[tid] may have 'v' (vx,vy); otherwise use 0
    pred_prev = torch.empty_like(c_prev)
    for j, tid in enumerate(prev_ids):
        vx, vy = prev_state.get(tid, {}).get("v", (0.0, 0.0))
        pred_prev[j, 0] = c_prev[j, 0] + vx
        pred_prev[j, 1] = c_prev[j, 1] + vy

    # center distances normalized by dynamic diagonal gate
    if N and M:
        dists = torch.cdist(c_curr, pred_prev)  # (N,M)
        # diagonal gate per column (prev track)
        diag_gate = (diag_coef * d_prev).clamp(min=1.0).unsqueeze(0)  # (1,M)
        dist_norm = (dists / diag_gate).cpu().numpy()
        iou_np = ious.cpu().numpy()
    else:
        dist_norm = np.zeros((N, M), dtype=float)
        iou_np = np.zeros((N, M), dtype=float)

    # optional keypoint geometry (avg L2 over shared confident kps), normalized
    if curr_keypoints is not None and N and M:
        K = curr_keypoints.shape[1]
        kp_cost = np.zeros((N, M), dtype=float)
        for j, tid in enumerate(prev_ids):
            last_kpts = prev_state.get(tid, {}).get("last_kpts", None)  # (K,2) or None
            if last_kpts is None:
                kp_cost[:, j] = 0.0
                continue
            last = np.asarray(last_kpts, dtype=float)
            last_mask = np.isfinite(last[:, 0]) & np.isfinite(last[:, 1])

            # normalization by predicted diag
            norm = float(max(1.0, diag_coef * float(d_prev[j].item()))) if M else 1.0

            for i in range(N):
                cur = curr_keypoints[i, :, :2]
                conf = curr_keypoints[i, :, 2] >= kpt_conf_thres
                both = last_mask & conf
                if not np.any(both):
                    kp_cost[i, j] = 0.0
                else:
                    diff = cur[both] - last[both]
                    kp_cost[i, j] = float(np.nanmean(np.linalg.norm(diff, axis=1)) / norm)
    else:
        kp_cost = np.zeros((N, M), dtype=float)

    # class mismatch penalty
    cls_cost = np.zeros((N, M), dtype=float)
    if curr_classes is not None and N and M:
        for j, tid in enumerate(prev_ids):
            trk_cls = prev_state.get(tid, {}).get("cls", None)
            if trk_cls is None:
                continue
            mask = (curr_classes.astype(int) != int(trk_cls))
            cls_cost[mask, j] = class_penalty

    # build cost: lower is better
    # gating: if both (IoU < thr) and (dist_norm > 1.0), set to big
    BIG = 1e6
    cost = (iou_coef * (1.0 - iou_np)) + (dist_coef * dist_norm) + (kpt_coef * kp_cost) + cls_cost
    bad = (iou_np < iou_thr) & (dist_norm > 1.0)
    cost[bad] = BIG

    # if a fixed centre_thr is provided, also gate by that absolute distance
    if centre_thr is not None and N and M:
        abs_d = torch.cdist(c_curr, c_prev).cpu().numpy()
        cost[abs_d > float(centre_thr)] = BIG

    # solve assignment
    if N and M:
        if linear_sum_assignment is not None:
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols):
                if cost[i, j] < BIG:
                    matches[int(i)] = int(prev_ids[j])
        else:
            # greedy fallback
            order = np.argsort(cost, axis=None)
            taken_i = set(); taken_j = set()
            for flat in order:
                i, j = np.unravel_index(flat, cost.shape)
                if i in taken_i or j in taken_j:
                    continue
                if cost[i, j] >= BIG:
                    break
                matches[int(i)] = int(prev_ids[j])
                taken_i.add(i); taken_j.add(j)

    # update track ages + remove stale
    matched_tids = set(matches.values())
    for tid in list(prev_state.keys()):
        if tid not in matched_tids:
            st = prev_state[tid]
            st["age"] = st.get("age", 0) + 1
            if st["age"] > max_miss:
                del prev_state[tid]

    # update matched track state (age=0, hits, cls, velocity, last_kpts)
    for i, tid in matches.items():
        j = prev_ids.index(tid)
        # age/hits
        st = prev_state.setdefault(tid, {})
        st["age"] = 0
        st["hits"] = st.get("hits", 0) + 1
        # class
        if curr_classes is not None:
            st["cls"] = int(curr_classes[i])
        # velocity update (EMA of Δcenter)
        cx_prev, cy_prev = float(c_prev[j, 0].item()), float(c_prev[j, 1].item())
        cx_curr, cy_curr = float(c_curr[i, 0].item()), float(c_curr[i, 1].item())
        vx_new, vy_new = (cx_curr - cx_prev), (cy_curr - cy_prev)
        vx_old, vy_old = st.get("v", (0.0, 0.0))
        st["v"] = (vel_ema * vx_old + (1.0 - vel_ema) * vx_new,
                   vel_ema * vy_old + (1.0 - vel_ema) * vy_new)
        # last keypoints (store (K,2); NaN for low-conf)
        if curr_keypoints is not None:
            kps = curr_keypoints[i]
            K = kps.shape[0]
            arr = np.full((K, 2), np.nan, dtype=float)
            mask = kps[:, 2] >= kpt_conf_thres
            arr[mask, :] = kps[mask, :2]
            st["last_kpts"] = arr

    return matches


# ---------------------------------------------------------
# ANALYTICS FUNCTIONS
# ---------------------------------------------------------

def find_idle_periods(
    df_track: pd.DataFrame, 
    keypoint_idx: int,
    min_track_len: int,
    smoothing_window: int,
    velocity_threshold: float,
    min_idle_frames: int
) -> List[Tuple[int, int]]:
    """
    Identifies idle periods in a single instrument track for a specific keypoint.
    An idle period is a segment where smoothed velocity is below a threshold.
    """
    if len(df_track) < min_track_len:
        return []

    # Use the specified keypoint index for analysis
    df_kp = df_track[df_track['kp_idx'] == keypoint_idx].sort_values('frame_id').copy()

    # If the keypoint is not present or has too few points, return no periods
    if len(df_kp) < min_track_len:
        return []

    # Interpolate to fill missing frames
    full_range = pd.DataFrame({'frame_id': np.arange(df_kp['frame_id'].min(), df_kp['frame_id'].max() + 1)})
    df_kp = pd.merge(full_range, df_kp, on='frame_id', how='left')
    # Infer object types to avoid FutureWarning, then interpolate numeric columns
    df_kp = df_kp.infer_objects(copy=False)
    df_kp = df_kp.interpolate()


    # Smooth position data
    df_kp['x_smooth'] = df_kp['x'].rolling(window=smoothing_window, center=True, min_periods=1).mean()
    df_kp['y_smooth'] = df_kp['y'].rolling(window=smoothing_window, center=True, min_periods=1).mean()

    # Calculate velocity
    dx = df_kp['x_smooth'].diff().fillna(0)
    dy = df_kp['y_smooth'].diff().fillna(0)
    df_kp['velocity'] = np.sqrt(dx**2 + dy**2)

    # Detect idle frames
    df_kp['is_idle'] = df_kp['velocity'] < velocity_threshold

    # Find contiguous blocks of idle frames
    idle_periods = []
    is_currently_idle = False
    start_frame = 0
    for _, row in df_kp.iterrows():
        if row['is_idle'] and not is_currently_idle:
            is_currently_idle = True
            start_frame = int(row['frame_id'])
        elif not row['is_idle'] and is_currently_idle:
            is_currently_idle = False
            end_frame = int(row['frame_id']) - 1
            if (end_frame - start_frame + 1) >= min_idle_frames:
                idle_periods.append((start_frame, end_frame))

    # Check for an idle period that extends to the end of the track
    if is_currently_idle:
        end_frame = int(df_kp['frame_id'].max())
        if (end_frame - start_frame + 1) >= min_idle_frames:
            idle_periods.append((start_frame, end_frame))

    return idle_periods
