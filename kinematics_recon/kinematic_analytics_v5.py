"""Kinematic analytics v5 – multi‑keypoint *samples*, multi‑scale windows, multi‑file/videos.

This version treats a **sample** as a multi‑KP time window from a single instrument track
(so clustering uses all keypoints jointly). It also supports per‑instrument analysis, multiple
CSV/video pairs, and exporting representative **clips per cluster** with knobs for length,
count, and selection strategy.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
import umap
import cv2

plt.rcParams.update({"figure.autolayout": True})

# ------------------------------- Types ---------------------------------------
TrackKey = Tuple[str, int, int]  # (instrument, source_idx, track_id)
KpIdx = int

@dataclass(frozen=True, slots=True)
class Sample:
    key: TrackKey
    kp_list: Tuple[KpIdx, ...]
    start_f: int
    end_f: int
    fps: int

# ------------------------------- IO ------------------------------------------

def _expand_csvs(paths: List[Path]) -> List[Path]:
    out: List[Path] = []
    for p in paths:
        if p.is_dir():
            out.extend(sorted(p.glob("*.csv")))
        else:
            out.append(p)
    return out


def load_tracks_multi(csv_paths: List[Path]) -> Dict[Tuple[str,int,int,int], pd.DataFrame]:
    """Return dict of (instrument, source, track_id, kp_idx) → DataFrame."""
    tracks: Dict[Tuple[str,int,int,int], pd.DataFrame] = {}
    csvs = _expand_csvs(csv_paths)
    for src_idx, csvp in enumerate(csvs):
        df = pd.read_csv(csvp)
        if 'instrument' not in df.columns:
            df['instrument'] = 'unknown'
        df = df.sort_values(['instrument','track_id','kp_idx','frame_id']).reset_index(drop=True)
        for (inst, tid, kp), g in df.groupby(['instrument','track_id','kp_idx'], sort=False):
            g = g.copy(); g['source'] = src_idx
            tracks[(str(inst), src_idx, int(tid), int(kp))] = g
    return tracks

# ---------------------------- preprocessing ----------------------------------

def interpolate_track(df: pd.DataFrame) -> pd.DataFrame:
    full = pd.DataFrame({"frame_id": np.arange(int(df.frame_id.min()), int(df.frame_id.max())+1)})
    m = full.merge(df, on="frame_id", how="left")
    for col in ["x","y","depth"]:
        if col in m.columns:
            m[col] = m[col].interpolate(limit_direction="both").ffill().bfill()
    for col in ['instrument','source','track_id','kp_idx','confidence']:
        if col in m.columns:
            m[col] = m[col].ffill().bfill()
    return m

# --------------------------- sample assembly ---------------------------------

def assemble_samples(tracks_kp: Dict[Tuple[str,int,int,int], pd.DataFrame],
                     fps: int,
                     window_mode: str,
                     win_sec: float,
                     stride_sec: float,
                     rand_mean_sec: float,
                     rand_std_sec: float,
                     rand_per_track: int) -> Tuple[List[Sample], Dict[Sample, pd.DataFrame]]:
    """Create multi‑kp windows per (instrument, source, track_id).
    Returns list[Sample] and dict[Sample→(T×C) DataFrame] with columns like x_0,y_0,(depth_0),x_1,....
    """
    # group by (inst,src,tid)
    by_track: Dict[TrackKey, Dict[int, pd.DataFrame]] = {}
    for (inst, src, tid, kp), df in tracks_kp.items():
        key = (inst, src, tid)
        by_track.setdefault(key, {})[kp] = interpolate_track(df)

    samples: List[Sample] = []
    data: Dict[Sample, pd.DataFrame] = {}

    for key, kp_dict in by_track.items():
        kp_list = tuple(sorted(kp_dict.keys()))
        fmin = min(int(df.frame_id.min()) for df in kp_dict.values())
        fmax = max(int(df.frame_id.max()) for df in kp_dict.values())
        frames = np.arange(fmin, fmax+1)
        wide = pd.DataFrame({"frame_id": frames})
        for kp in kp_list:
            df = kp_dict[kp][['frame_id','x','y']].copy()
            df.columns = ['frame_id', f'x_{kp}', f'y_{kp}']
            wide = wide.merge(df, on='frame_id', how='left')
            if 'depth' in kp_dict[kp].columns:
                dfd = kp_dict[kp][['frame_id','depth']].copy(); dfd.columns = ['frame_id', f'depth_{kp}']
                wide = wide.merge(dfd, on='frame_id', how='left')
        wide = wide.interpolate(limit_direction='both').ffill().bfill()

        if window_mode == 'full':
            s = Sample(key, kp_list, int(wide.frame_id.min()), int(wide.frame_id.max()), fps)
            samples.append(s); data[s] = wide.copy()
        elif window_mode == 'sliding':
            L = max(1, int(round(win_sec*fps)))
            S = max(1, int(round(stride_sec*fps)))
            for s0 in range(0, len(wide)-L+1, S):
                sub = wide.iloc[s0:s0+L]
                s = Sample(key, kp_list, int(sub.frame_id.iloc[0]), int(sub.frame_id.iloc[-1]), fps)
                samples.append(s); data[s] = sub.reset_index(drop=True)
        else:  # random
            # seed per track for reproducibility but variety
            seed = (key[1] << 32) ^ key[2]
            rng = np.random.default_rng(seed & 0xFFFFFFFF)
            for _ in range(rand_per_track):
                L = int(np.clip(rng.normal(rand_mean_sec*fps, rand_std_sec*fps), 5, len(wide)))
                s0 = int(rng.integers(0, max(1, len(wide)-L)))
                sub = wide.iloc[s0:s0+L]
                s = Sample(key, kp_list, int(sub.frame_id.iloc[0]), int(sub.frame_id.iloc[-1]), fps)
                samples.append(s); data[s] = sub.reset_index(drop=True)

    return samples, data

# ----------------------------- featurization ---------------------------------
def sample_lengths_sec(samples: List[Sample], indices=None) -> np.ndarray:
    arr = np.array([(s.end_f - s.start_f + 1) / s.fps for s in samples], dtype=float)
    if indices is None:
        return arr
    idx = np.asarray(indices, dtype=int)
    return arr[idx]

def _per_series_feats(v: np.ndarray, t: np.ndarray) -> List[float]:
    if len(v) < 4: return [np.nan]*6
    dv = np.gradient(v, t)
    acc = np.gradient(dv, t)
    jerk = np.gradient(acc, t)
    return [float(np.nanstd(v)), float(np.nanstd(dv)), float(np.nanstd(acc)),
            float(np.nanmedian(np.abs(dv))), float(np.nanpercentile(np.abs(dv),90)),
            float(np.nanmean(np.abs(jerk)))]


def extract_multi_kp_features(sample: Sample, mat: pd.DataFrame, canonical_kps: Tuple[int, ...], include_depth: bool=True) -> np.ndarray:
    """Fixed-width feature vector across samples:
       • Iterate over a canonical set of keypoint indices for the instrument.
       • For missing keypoints (or missing depth), insert NaN feature slots so all samples align.
       • Cross-kp relation summary stays fixed-size via mean/std over all pairs.
    """
    t = (mat.frame_id.values - mat.frame_id.values[0]) / sample.fps
    feats: List[float] = []
    # per‑kp univariate (x, y, depth slots)
    for kp in canonical_kps:
        if f'x_{kp}' in mat.columns:
            feats += _per_series_feats(mat[f'x_{kp}'].values, t)
        else:
            feats += [np.nan] * 6
        if f'y_{kp}' in mat.columns:
            feats += _per_series_feats(mat[f'y_{kp}'].values, t)
        else:
            feats += [np.nan] * 6
        if include_depth:
            if f'depth_{kp}' in mat.columns:
                feats += _per_series_feats(mat[f'depth_{kp}'].values, t)
            else:
                feats += [np.nan] * 6
    # summarize pairwise distances across present kps
    present_kps = [kp for kp in canonical_kps if f'x_{kp}' in mat.columns and f'y_{kp}' in mat.columns]
    pair_stats = []
    for i, kpi in enumerate(present_kps):
        for kpj in present_kps[i+1:]:
            dx = mat[f'x_{kpj}'].values - mat[f'x_{kpi}'].values
            dy = mat[f'y_{kpj}'].values - mat[f'y_{kpi}'].values
            dist = np.hypot(dx, dy)
            pair_stats.append([np.nanmean(dist), np.nanstd(dist), np.nanpercentile(dist, 90)])
    if pair_stats:
        pair_stats = np.array(pair_stats)
        feats += list(np.nanmean(pair_stats, axis=0)) + list(np.nanstd(pair_stats, axis=0))
    else:
        feats += [np.nan]*6
    return np.array(feats, dtype=float)

# ------------------------------- clustering ----------------------------------

def eps_from_kdist(X: np.ndarray, k: int = 8, pct: float = 95.0) -> float:
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    dists, _ = nn.kneighbors(X)
    kdist = np.sort(dists[:, -1])
    return float(np.percentile(kdist, pct))

# ------------------------------- plotting ------------------------------------

def plot_umap(X: np.ndarray, labels: np.ndarray, out: Path, title: str, lengths_sec: np.ndarray | None = None):
    """
    Produce both 2D UMAP (file umap_clusters.png) and a 3D view that uses UMAP coords
    for X/Y and clip length in seconds for Z (file umap_clusters_3d.png).

    lengths_sec should be a 1D array aligned with rows of X; if omitted a zero Z-plane is used.
    Returns an (N,3) array with columns [UMAP-1, UMAP-2, length_sec].
    """
    reducer = umap.UMAP(n_neighbors=12, min_dist=0.1, metric="euclidean", random_state=42)
    emb2 = reducer.fit_transform(X)

    n = emb2.shape[0]
    if lengths_sec is None or len(lengths_sec) != n:
        z = np.zeros(n, dtype=float)
    else:
        z = np.asarray(lengths_sec, dtype=float)

    emb3 = np.column_stack([emb2, z])

    # 2D plot (compat)
    plt.figure(figsize=(6.5, 5.2))
    sc = plt.scatter(emb2[:, 0], emb2[:, 1], c=labels, cmap="tab10", s=20, alpha=0.9)
    plt.title(title)
    plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
    plt.colorbar(sc, label="cluster")
    plt.tight_layout()
    plt.savefig(out / "umap_clusters.png", dpi=300)
    plt.close()

    # 3D plot: UMAP1, UMAP2, clip length (s)
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    # orient camera closer to the x-y plane (flatter view)
    ax.view_init(elev=10, azim=-60)
    p = ax.scatter(emb3[:, 0], emb3[:, 2], emb3[:, 1], c=labels, cmap="tab10", s=30, alpha=0.9)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("clip length (s)")
    ax.set_zlabel("UMAP-2")
    ax.set_title(f"{title} — 3D (Z=UMAP-2)")
    fig.colorbar(p, ax=ax, pad=0.1, label="cluster")
    plt.tight_layout()
    plt.savefig(out / "umap_clusters_3d.png", dpi=300)
    plt.pause(10)
    plt.close()

    return emb3

def plot_exemplar_sample(sample: Sample, mat: pd.DataFrame, out: Path):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    t = (mat.frame_id - mat.frame_id.min()) / sample.fps
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection='3d')
    for kp in sample.kp_list:
        x = mat[f'x_{kp}']
        y = mat[f'y_{kp}']
        ax.plot(x, t, y, label=f'KP {kp}', alpha=0.9)
    ax.set_xlabel('x')
    ax.set_ylabel('time (s)')
    ax.set_zlabel('y')
    ax.set_title(f"3D Noodle – {sample.key} frames {sample.start_f}-{sample.end_f}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()

# ---------------------------- clip export/iterator ----------------------------

def _write_video_segment(cap: cv2.VideoCapture, start_f: int, end_f: int, name: Path, fps_fallback: int, show: bool = False):
    if cap is None or not cap.isOpened(): return
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_v = cap.get(cv2.CAP_PROP_FPS) or fps_fallback
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(str(name), fourcc, fps_v, (width, height))
    while cap.get(cv2.CAP_PROP_POS_FRAMES) <= end_f:
        ret, frame = cap.read()
        if not ret: break
        vw.write(frame)
        if show:
            cv2.imshow('cluster-clip', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    vw.release()

def _centered_bounds(s: Sample, secs: float | None) -> Tuple[int,int]:
    if secs is None:
        return s.start_f, s.end_f
    half = int(round((secs * s.fps) / 2))
    mid = (s.start_f + s.end_f) // 2
    return max(s.start_f, mid - half), min(s.end_f, mid + half)


def export_cluster_clips(
    video_paths: List[Path],
    samples: List[Sample],
    labels: np.ndarray,
    mats: Dict[Sample, pd.DataFrame],
    canonical_kps: Tuple[int, ...],
    out: Path,
    secs: float = 5.0,
    per_cluster: int = 2,
    mode: str = 'medoid',
    show: bool = False,
    seed: int = 0,
):
    out_dir = out/"clips"; out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # group sample indices by cluster id
    clusters: Dict[int, List[int]] = {}
    for i, lab in enumerate(labels):
        if lab == -1: continue
        clusters.setdefault(int(lab), []).append(i)

    # cache caps by source
    caps: Dict[int, cv2.VideoCapture] = {}
    def get_cap(src: int):
        if src in caps: return caps[src]
        if src >= len(video_paths) or video_paths[src] is None: return None
        cap = cv2.VideoCapture(str(video_paths[src]))
        if not cap.isOpened(): return None
        caps[src] = cap; return cap

    for cid, idxs in clusters.items():
        chosen: List[int] = []
        if mode == 'random':
            chosen = rng.choice(idxs, size=min(per_cluster, len(idxs)), replace=False).tolist()
        elif mode == 'longest':
            lengths = sorted(idxs, key=lambda i: samples[i].end_f - samples[i].start_f, reverse=True)
            chosen = lengths[:per_cluster]
        else:  # medoid then random fill
            feats = [extract_multi_kp_features(samples[i], mats[samples[i]], canonical_kps) for i in idxs]
            F = np.vstack(feats)
            col_mask = ~np.all(np.isnan(F), axis=0)
            F = F[:, col_mask]
            Z = StandardScaler().fit_transform(F)
            centroid = Z.mean(axis=0, keepdims=True)
            med_local = int(np.argmin(((Z - centroid)**2).sum(axis=1)))
            chosen = [idxs[med_local]]
            others = [i for i in idxs if i != chosen[0]]
            if others:
                extra = rng.choice(others, size=min(per_cluster-1, len(others)), replace=False).tolist()
                chosen += extra

        for i in chosen:
            s = samples[i]
            src = s.key[1]
            cap = get_cap(src)
            if cap is None: continue
            start_f, end_f = _centered_bounds(s, secs)
            name = out_dir / f"c{cid}_{s.key[0]}_src{src}_tid{s.key[2]}_{start_f}-{end_f}.mp4"
            _write_video_segment(cap, start_f, end_f, name, fps_fallback=s.fps, show=show)

    for cap in caps.values(): cap.release()
    if show: cv2.destroyAllWindows()

# --------------------------------- CLI ---------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=Path, nargs='+', default=[Path("datasets/SurgVU/surgvu_049_inference")])
    ap.add_argument('--video', type=Path, nargs='*', default=None)
    ap.add_argument('--fps', type=int, default=15)
    ap.add_argument('--method', choices=['dbscan','kmeans'], default='kmeans')
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--eps', default='auto')
    ap.add_argument('--min-samples', type=int, default=8)
    ap.add_argument('--min-segments', type=int, default=30)
    ap.add_argument('--out', type=Path, default=None)
    # windowing 
    ap.add_argument('--window-mode', choices=['full','sliding','random'], default='sliding')
    ap.add_argument('--length-as-feature', default=True, action='store_false',
                help='append window length (sec) as a feature before scaling')
    ap.add_argument('--win-sec', type=float, default=2.5) 
    ap.add_argument('--stride-sec', type=float, default=0.5)
    ap.add_argument('--rand-mean-sec', type=float, default=2.5)
    ap.add_argument('--rand-std-sec', type=float, default=1.5)
    ap.add_argument('--rand-per-track', type=int, default=5)
    ap.add_argument('--min-kps', type=int, default=1, help='minimum number of present keypoints required in a window')
    # clips
    ap.add_argument('--export-clips', action='store_true')
    ap.add_argument('--secs', type=float, default=5.0)
    ap.add_argument('--per-cluster', type=int, default=5)
    ap.add_argument('--clip-mode', choices=['medoid','random','longest'], default='random')
    ap.add_argument('--show-clips', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    out_root = args.out or (args.csv[0] / 'analytics_multiKP')
    out_root.mkdir(parents=True, exist_ok=True)

    csv_files = _expand_csvs(list(args.csv))
    video_list = list(args.video) if args.video is not None else []

    # load per‑kp tracks
    tracks_kp = load_tracks_multi(csv_files)

    # filter instruments by unique (source, track_id) count
    from collections import defaultdict
    uniq_tracks_by_inst = defaultdict(set)
    for (inst, src, tid, kp) in tracks_kp.keys():
        uniq_tracks_by_inst[inst].add((src, tid))
    instruments = [inst for inst, S in uniq_tracks_by_inst.items() if len(S) >= args.min_segments]
    if not instruments:
        print('[!] No instrument meets min‑segments; adjust threshold or add more CSVs.')
        return

    for inst in instruments:
        subset = {k:v for k,v in tracks_kp.items() if k[0]==inst}
        if not subset: continue
        samples, mats = assemble_samples(subset, args.fps, args.window_mode, args.win_sec,
                                         args.stride_sec, args.rand_mean_sec, args.rand_std_sec,
                                         args.rand_per_track)
        if not samples:
            print(f"[skip] {inst}: no windows produced"); continue

        canonical_kps = tuple(sorted({kp for s in samples for kp in s.kp_list}))
        feat_list = []
        valid_idx = []
        for i, s in enumerate(samples):
            mat = mats[s]
            present = [kp for kp in canonical_kps if f'x_{kp}' in mat.columns and f'y_{kp}' in mat.columns]
            if len(present) < args.min_kps:
                continue
            feat_list.append(extract_multi_kp_features(s, mat, canonical_kps))
            valid_idx.append(i)
        if not feat_list:
            print(f"[skip] {inst}: no samples pass min-kps={args.min_kps}")
            continue
        F = np.vstack(feat_list)
        col_mask = ~np.all(np.isnan(F), axis=0)
        F = F[:, col_mask]
        row_mask = ~np.any(~np.isfinite(F), axis=1)
        X = F[row_mask]
        kept_idx = np.array(valid_idx)[row_mask]
        if len(X) < 2:
            print(f"[skip] {inst}: not enough valid windows after cleaning"); continue
        lengths_valid = sample_lengths_sec(samples, kept_idx)
        if args.length_as_feature:
            X = np.hstack([X, lengths_valid.reshape(-1, 1)])
        Xs = StandardScaler().fit_transform(X)

        out_dir = out_root / inst.replace(' ','_')
        out_dir.mkdir(parents=True, exist_ok=True)

        labels_all = np.full(len(samples), -1)
        if args.method=='dbscan':
            eps = float(eps_from_kdist(Xs, k=max(3,args.min_samples), pct=95.0)) if str(args.eps).lower()=='auto' else float(args.eps)
            db = DBSCAN(eps=eps, min_samples=args.min_samples).fit(Xs)
            labels_valid = db.labels_
            labels_all[kept_idx] = labels_valid
            emb = plot_umap(Xs, labels_valid, out_dir,
              f'UMAP – DBSCAN [{inst}] (eps={eps:.3f})',
              lengths_sec=lengths_valid)
        else:
            km = KMeans(n_clusters=args.k, n_init='auto', random_state=0).fit(Xs)
            labels_valid = km.labels_
            labels_all[kept_idx] = labels_valid
            emb = plot_umap(Xs, labels_valid, out_dir,
              f'UMAP – KMeans (k={args.k}) [{inst}]',
              lengths_sec=lengths_valid)
            
            # plot exemplars and create stitched clips for every cluster
            unique_clusters = [cid for cid in np.unique(labels_valid) if cid != -1]
            rng = np.random.default_rng(args.seed)
            
            # Cache video captures by source index
            caps = {}
            def get_cap(src_idx):
                if src_idx in caps: return caps[src_idx]
                video_path = None
                if src_idx < len(video_list) and video_list[src_idx] is not None:
                    video_path = video_list[src_idx]
                elif src_idx < len(csv_files):
                    csv_path = csv_files[src_idx]
                    vid_name = csv_path.name.replace('_tracking_report.csv', '_annotated.mp4')
                    potential_vid_path = csv_path.with_name(vid_name)
                    if potential_vid_path.exists():
                        video_path = potential_vid_path
                
                if video_path:
                    cap = cv2.VideoCapture(str(video_path))
                    if cap.isOpened():
                        caps[src_idx] = cap
                        return cap
                    else:
                        print(f"[warn] Could not open video for exemplar: {video_path}")
                else:
                    print(f"[info] No video found for exemplar from source {src_idx}")
                caps[src_idx] = None
                return None

            for cid in unique_clusters:
                cluster_sample_indices = np.where(labels_all == cid)[0]
                if len(cluster_sample_indices) == 0: continue

                # Select N samples from the cluster based on clip-mode
                chosen_indices = []
                if args.clip_mode == 'random':
                    chosen_indices = rng.choice(cluster_sample_indices, size=min(args.per_cluster, len(cluster_sample_indices)), replace=False).tolist()
                elif args.clip_mode == 'longest':
                    lengths = sorted(cluster_sample_indices, key=lambda i: samples[i].end_f - samples[i].start_f, reverse=True)
                    chosen_indices = lengths[:args.per_cluster]
                else: # medoid
                    cluster_feats_indices = np.where(np.isin(kept_idx, cluster_sample_indices))[0]
                    cluster_feats = Xs[cluster_feats_indices]
                    centroid = cluster_feats.mean(0, keepdims=True)
                    medoid_local_idx = int(np.argmin(((cluster_feats - centroid) ** 2).sum(1)))
                    medoid_global_idx = kept_idx[cluster_feats_indices[medoid_local_idx]]
                    chosen_indices = [medoid_global_idx]
                    others = [i for i in cluster_sample_indices if i != medoid_global_idx]
                    if others:
                        extra = rng.choice(others, size=min(args.per_cluster - 1, len(others)), replace=False).tolist()
                        chosen_indices.extend(extra)

                # Plot trajectory for the first chosen sample (e.g., the medoid)
                if chosen_indices:
                    first_samp = samples[chosen_indices[0]]
                    plot_path = out_dir / f"exemplar_window_cluster{cid}.png"
                    plot_exemplar_sample(first_samp, mats[first_samp], plot_path)

                # Stitch clips for the chosen samples
                vw = None
                for i, sample_idx in enumerate(chosen_indices):
                    samp = samples[sample_idx]
                    cap = get_cap(samp.key[1])
                    if not cap: continue

                    start_f, end_f = _centered_bounds(samp, args.secs)
                    
                    if vw is None: # Initialize video writer on first valid clip
                        clip_path = out_dir / f"exemplar_stitched_cluster{cid}.mp4"
                        fps_v = cap.get(cv2.CAP_PROP_FPS) or samp.fps
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        vw = cv2.VideoWriter(str(clip_path), fourcc, fps_v, (width, height))
                    
                    # Add freeze frame before the new segment (except for the first one)
                    if i > 0:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
                        ret, first_frame = cap.read()
                        if ret:
                            freeze_frames = int(round(0.5 * (cap.get(cv2.CAP_PROP_FPS) or samp.fps)))
                            for _ in range(freeze_frames):
                                vw.write(first_frame)

                    # Write the actual clip segment
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
                    while cap.get(cv2.CAP_PROP_POS_FRAMES) <= end_f:
                        ret, frame = cap.read()
                        if not ret: break
                        vw.write(frame)
                
                if vw:
                    vw.release()
            
            # Release all cached video captures
            for cap in caps.values():
                if cap: cap.release()


        # mapping CSV
        rows = []
        for i, s in enumerate(samples):
            rows.append({
                'instrument': s.key[0], 'source': s.key[1], 'track_id': s.key[2],
                'start_frame': s.start_f, 'end_frame': s.end_f, 'label': int(labels_all[i])
            })
        pd.DataFrame(rows).to_csv(out_dir/'window_labels.csv', index=False)

        # optional clips per cluster
        if args.export_clips and len(video_list) > 0:
            vids = [video_list[i] if i < len(video_list) else None for i in range(len(csv_files))]
            export_cluster_clips(
                vids, samples, labels_all, mats, canonical_kps, out_dir,
                secs=args.secs, per_cluster=args.per_cluster,
                mode=args.clip_mode, show=args.show_clips, seed=args.seed
            )

    print('✓ analytics v5 complete →', out_root)

if __name__ == '__main__':
    main()
