#!/usr/bin/env python3
"""
COCO-Keypoints  ➜  YOLO-v8 Pose
-------------------------------------------------------------
 • Forces every object to have 7 key-points (pads with zeros)
 • Writes Ultralytics-style txt labels  (one object / line):
       cls  xc  yc  bw  bh   x1 y1 v1 … x7 y7 v7
   where coordinates are normalized to [0,1] and   v ∈ {0,1,2}.
 • Generates the dataset-yaml with kpt_shape=[7,3]

Example
-------
python coco2yolo_pose.py \
        --img-dir datasets/SurgPose/images \
        --coco    datasets/SurgPose_COCO/annotations/train.json \
        --out     datasets/SurgPoseYOLO/pose

python coco2yolo_pose.py \
        --img-dir datasets/SurgPose/images \
        --coco-train datasets/SurgPose_COCO/annotations/train.json \
        --coco-val   datasets/SurgPose_COCO/annotations/val.json \
        --out   datasets/SurgPoseYOLO/pose
"""
import json, argparse, random, shutil
from pathlib import Path
import os
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import yaml

# ------------------------------------------------- user editable
TOOLS     = ['grasper', 'scissors', 'clip_applier', 'cautery_hook']  # 4 classes
N_KPTS    = 7                                                        # fixed
OUT_YAML  = 'surgpose_pose.yaml'                                     # written inside --out
# --------------------------------------------------------------

CLS2ID = {n:i for i,n in enumerate(TOOLS)}            # str → 0-based id
PAD    = [0.0, 0.0, 0.0] * N_KPTS                      # [x y v] * 7 zeros

# ---------------------------------------------------------------------------
def load_coco(ann_path: Path):
    """return (images list, anns by image_id dict)"""
    with ann_path.open() as f:
        coco = json.load(f)
    imgs = {img['id']: img for img in coco['images']}
    anns = defaultdict(list)
    for a in coco['annotations']:
        anns[a['image_id']].append(a)
    # category id ➜ name
    cid2name = {c['id']: c['name'].lower() for c in coco['categories']}
    return imgs, anns, cid2name

def coco_bbox_to_yolo(b, w, h):
    """[x,y,w,h] (abs) → xc,yc,bw,bh (relative)"""
    x,y,bw,bh = b
    return (x + bw/2)/w, (y + bh/2)/h, bw/w, bh/h

# --- put this near the top, right after N_KPTS ----------------------
# Mapping from SurgPose index (1-7) -> Northwell seven-slot order (0-6)
ORDER = [4, 5, 3, 2, 1, 6, 7]          # <- 1-based indices

# -------------------------------------------------------------------
def kpts_pad7_norm(flat, w, h):
    """
    `flat` is the list read directly from COCO (x1,y1,v1,…).
    We (a) pad to 7x3, (b) re-order with ORDER, (c) normalise x,y.
    """
    # --------------- pad -------------------------------------------
    k = len(flat) // 3
    if k < N_KPTS:
        flat += PAD[: (N_KPTS - k) * 3]

    # --------------- reshape & reorder -----------------------------
    arr = np.asarray(flat, np.float32).reshape(-1, 3)          # (7,3)
    arr = arr[[i - 1 for i in ORDER], :]                       # reorder

    # --------------- normalise x, y -------------------------------
    arr[:, 0] /= w
    arr[:, 1] /= h
    return arr.reshape(-1).tolist()


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def convert_split(img_ids, imgs, anns, cid2name,
                  img_root: Path, out_img: Path, out_lbl: Path):
    """
    Write YOLO-Pose images and label-txt files for one split, fixing
    double “images/…” and duplicate “d1_” prefixes in the source path.
    """
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    root_name_lc = img_root.name.lower()

    for iid in tqdm(img_ids, desc=f'write {out_img.parent.name}'):
        rec = imgs[iid]

        # ----------- sanitize relative image path --------------------------
        rel = Path(rec['file_name'])
        parts = list(rel.parts)
        while parts and parts[0].lower() in {'images', root_name_lc}:
            parts.pop(0)                               # drop leading dir
        rel = Path(*parts) if parts else rel.name

        src = img_root / rel                           # fixed source path
        if not src.exists():
            tqdm.write(f'⚠ missing: {src} – skipped')
            continue

        # ----------- build unique output filename --------------------------
        base = rel.name
        # remove *second* “d1_” if present (d1_d1_xxx → d1_xxx)
        if base.startswith('d1_d1_'):
            base = 'd1_' + base[len('d1_d1_'):]
        tag  = out_img.parent.name + '_'               # e.g. train_ / val_
        stem = (tag + base) if not base.startswith(tag) else base
        stem = Path(stem).stem                         # drop extension

        # ----------- link/copy image --------------------------------------
        dst_img = out_img / f'{stem}.jpg'
        if not dst_img.exists():
            try:
                os.link(src, dst_img)                  # hard-link (fast)
            except OSError:
                shutil.copy2(src, dst_img)             # cross-FS fallback

        # ----------- write label file -------------------------------------
        w, h = rec['width'], rec['height']
        with (out_lbl / f'{stem}.txt').open('w') as lf:
            for a in anns[iid]:
                cls = cid2name[a['category_id']]
                if cls not in CLS2ID:
                    continue

                xc, yc, bw, bh = coco_bbox_to_yolo(a['bbox'], w, h)
                kps            = kpts_pad7_norm(a['keypoints'], w, h)
                line = [CLS2ID[cls], xc, yc, bw, bh] + kps
                lf.write(' '.join(f'{v:.6f}' if j else str(int(v))
                                  for j, v in enumerate(line)) + '\n')

# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img-dir', required=True, type=Path,
                    help='directory that contains all source images')
    ag = ap.add_mutually_exclusive_group(required=True)
    ag.add_argument('--coco',       type=Path,
                    help='single COCO json (train/val split done here)')
    ag.add_argument('--coco-train', type=Path, help='COCO train json')
    ap.add_argument('--coco-val',   type=Path, help='COCO val json (when using --coco-train)')
    ap.add_argument('--val-split',  type=float, default=.1,
                    help='val-fraction if using a single COCO file')
    ap.add_argument('--out',        required=True, type=Path)
    args = ap.parse_args()

    # create folders
    for sub in ('images/train','images/val','labels/train','labels/val'):
        (args.out/sub).mkdir(parents=True, exist_ok=True)

    if args.coco:
        imgs, anns, cid2name = load_coco(args.coco)
        img_ids = list(imgs)
        random.seed(0); random.shuffle(img_ids)
        n_val   = int(len(img_ids)*args.val_split)
        val_ids = set(img_ids[:n_val])
        tr_ids  = img_ids[n_val:]
        convert_split(tr_ids, imgs, anns, cid2name,
                      args.img_dir, args.out/'images/train', args.out/'labels/train')
        convert_split(val_ids, imgs, anns, cid2name,
                      args.img_dir, args.out/'images/val',   args.out/'labels/val')
    else:  # train+val json given
        if not args.coco_val:
            ap.error('--coco-val is required when using --coco-train')
        imgs_tr, anns_tr, cid2name = load_coco(args.coco_train)
        imgs_va, anns_va, _        = load_coco(args.coco_val)   # same cat mapping
        convert_split(imgs_tr, imgs_tr, anns_tr, cid2name,
                      args.img_dir, args.out/'images/train', args.out/'labels/train')
        convert_split(imgs_va, imgs_va, anns_va, cid2name,
                      args.img_dir, args.out/'images/val',   args.out/'labels/val')

    # ---------- dataset YAML -------------------------------------------------
    yaml.safe_dump(dict(
        path       = str(args.out.resolve()),
        train      = 'images/train',
        val        = 'images/val',
        nc         = len(TOOLS),
        names      = TOOLS,
        kpt_shape  = [N_KPTS, 3]
    ), (args.out/OUT_YAML).open('w'))

    print('✓ YOLO-Pose dataset written →', args.out)

# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()
