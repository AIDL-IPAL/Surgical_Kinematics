#!/usr/bin/env python3
"""
Plot a schematic of the kinematic analytics v5 pipeline.

Conventions:
- Solid arrows/boxes = required stages
- Dotted arrows/boxes = optional branches (UMAP, exemplar plotting, clip export, UMAP→KMeans)
"""

import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def box(ax, x, y, w, h, text, dotted=False, fontsize=11):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.015",
        linewidth=1.8,
        linestyle=":" if dotted else "-",
        fill=False,
    )
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize)
    # Return box coordinates: center, left, right, top, bottom
    return {
        'center': (x + w/2, y + h/2),
        'left': (x, y + h/2),
        'right': (x + w, y + h/2),
        'top': (x + w/2, y + h),
        'bottom': (x + w/2, y)
    }

def get_connection_points(box1, box2):
    """Determine the best connection points between two boxes"""
    # Calculate the direction vector between centers
    dx = box2['center'][0] - box1['center'][0]
    dy = box2['center'][1] - box1['center'][1]
    
    # Determine which edges to connect based on relative positions
    if abs(dx) > abs(dy):  # Boxes are primarily horizontal to each other
        if dx > 0:  # box2 is to the right of box1
            return box1['right'], box2['left']
        else:  # box2 is to the left of box1
            return box1['left'], box2['right']
    else:  # Boxes are primarily vertical to each other
        if dy > 0:  # box2 is above box1
            return box1['top'], box2['bottom']
        else:  # box2 is below box1
            return box1['bottom'], box2['top']

def arrow(ax, p1, p2, dotted=False):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle="->", mutation_scale=12, lw=1.8,
        linestyle=":" if dotted else "-"
    ))

def build(fig_w=13, fig_h=8, style="light"):
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    if style == "dark":
        fig.patch.set_facecolor("#111")
        ax.set_facecolor("#111")

    # Increased spacing between boxes
    # ───────────────────── Row 1: Inputs ─────────────────────
    csv_box = box(ax, 0.02, 0.82, 0.28, 0.12,
                  "Inputs (per-video CSVs)\ntrack_id, kp_idx, frame_id, x, y, depth, conf")
    vid_box = box(ax, 0.40, 0.82, 0.28, 0.12,
                  "Optional Videos\nfor clip sampling / export", dotted=True)
    
    # Connect csv_box to vid_box (dotted)
    p1, p2 = get_connection_points(csv_box, vid_box)
    arrow(ax, p1, p2, dotted=True)

    # ─────────────────── Row 2: Load & filter ─────────────────
    load_box = box(ax, 0.02, 0.62, 0.28, 0.12,
                   "Load per-keypoint tracks\n(group by (instrument, source, track_id, kp_idx))")
    
    # Connect csv_box to load_box
    p1, p2 = csv_box['bottom'], load_box['top']
    arrow(ax, p1, p2)

    filt_box = box(ax, 0.40, 0.62, 0.28, 0.12,
                   "Per-instrument filter\n≥ min_segments unique tracks")
    
    # Connect load_box to filt_box
    p1, p2 = get_connection_points(load_box, filt_box)
    arrow(ax, p1, p2)

    # ─────────── Row 3: Windowing & sample assembly ───────────
    win_box = box(ax, 0.02, 0.42, 0.28, 0.12,
                  "Assemble multi-KP windows\nwindow_mode: full | sliding | random\n(T×C matrix per sample)")
    
    # Connect filt_box to win_box
    p1, p2 = filt_box['bottom'], win_box['top']
    arrow(ax, p1, p2)

    clean_box = box(ax, 0.40, 0.42, 0.28, 0.12,
                    "Quality gates\nmin_kps per window\ninterpolate & align frames")
    
    # Connect win_box to clean_box
    p1, p2 = get_connection_points(win_box, clean_box)
    arrow(ax, p1, p2)

    # ───────── Row 4: Features & preprocessing ─────────
    feat_box = box(ax, 0.02, 0.22, 0.28, 0.12,
                   "Feature extraction (fixed-width)\n• per-KP (x,y[,depth]): std, vel, acc, jerk stats\n• cross-KP distances: mean/std/p90")
    
    # Connect clean_box to feat_box
    p1, p2 = clean_box['bottom'], feat_box['top']
    arrow(ax, p1, p2)

    prep_box = box(ax, 0.40, 0.22, 0.28, 0.12,
                   "Feature cleaning\nDrop all-NaN columns\nKeep finite rows\nStandardScaler (z-score)")
    
    # Connect feat_box to prep_box
    p1, p2 = get_connection_points(feat_box, prep_box)
    arrow(ax, p1, p2)

    # ───────────── Row 5: Clustering & UMAP ─────────────
    db_box = box(ax, 0.78, 0.42, 0.28, 0.12,
                "DBSCAN (optional)\n• eps=auto via k-distance\n• min_samples", dotted=True)
    
    # Connect clean_box to db_box
    p1, p2 = get_connection_points(clean_box, db_box)
    arrow(ax, p1, p2, dotted=True)

    km_box = box(ax, 0.78, 0.22, 0.28, 0.12,
                "KMeans (optional)\n• k clusters\n• n_init", dotted=True)
    
    # Connect prep_box to km_box
    p1, p2 = get_connection_points(prep_box, km_box)
    arrow(ax, p1, p2, dotted=True)

    umap_box = box(ax, 0.78, 0.02, 0.28, 0.12,
                  "UMAP embedding (optional)\nvisualization; optional 2nd-stage KMeans", dotted=True)
    
    # Connect prep_box to umap_box
    p1, p2 = prep_box['bottom'], umap_box['top']
    arrow(ax, p1, p2, dotted=True)

    # ───────── Row 6: Exemplars & clip export ─────────
    ex_box = box(ax, 0.02, 0.02, 0.28, 0.12,
                "Per-cluster exemplar (optional)\nmedoid in feature space\n→ exemplar plots", dotted=True)
    
    # Connect from clusters to ex_box
    p1, p2 = db_box['bottom'], ex_box['top']
    arrow(ax, p1, p2, dotted=True)
    
    p1, p2 = km_box['left'], ex_box['right']
    arrow(ax, p1, p2, dotted=True)
    
    p1, p2 = umap_box['left'], ex_box['right']
    arrow(ax, p1, p2, dotted=True)

    clip_box = box(ax, 0.40, 0.02, 0.28, 0.12,
                  "Clip export (optional)\nper_cluster, secs, mode=medoid|random|longest\nalign by (source, frames)", dotted=True)
    
    # Connect clusters to clip_box
    p1, p2 = db_box['bottom'], clip_box['top']
    arrow(ax, p1, p2, dotted=True)
    
    p1, p2 = km_box['bottom'], clip_box['top']
    arrow(ax, p1, p2, dotted=True)

    # minor styling
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return fig, ax

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="analytics_schematic.png",
                    help="Output path (.png/.svg/.pdf).")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--style", choices=["light","dark"], default="light")
    ap.add_argument("--size", type=float, nargs=2, default=[13, 8],
                    help="Figure size in inches (W H).")
    ap.add_argument("--transparent", action="store_true")
    args = ap.parse_args()

    fig, _ = build(fig_w=args.size[0], fig_h=args.size[1], style=args.style)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", transparent=args.transparent)
    print(f"✓ saved: {args.out}")

if __name__ == "__main__":
    main()
