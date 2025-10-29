#!/usr/bin/env python3
"""
SurgPose ➜ YOLO-Pose dataset
----------------------------
 • Left camera only (`regular/left_video.mp4`)
 • 7 key-points / instrument (IDs 1-7 = left arm, 8-14 = right arm)
 • 4 tool classes: grasper | scissors | clip_applier | cautery_hook
"""

import argparse, json, yaml, random, itertools, csv
from pathlib import Path
import cv2, numpy as np
from tqdm import tqdm

TOOLS        = ['grasper', 'scissors', 'clip_applier', 'cautery_hook']
NAME2ID      = {n:i for i,n in enumerate(TOOLS)}         # 0-based for YOLO
KPT_SHAPE    = [7, 3]                                    # n_kpts, xyz(v)
OUT_YAML     = 'surgpose_pose.yaml'                      # written at --out

# ---------- helpers -----------------------------------------------------------------
TOK2CANON = {
    # grasper-family
    'lnd': 'grasper', 'mnd': 'grasper', 'prograsp': 'grasper',
    'forceps': 'grasper', 'cadiere': 'grasper',
    # scissors-family
    'scissor': 'scissors', 'scissors': 'scissors', 'maryland': 'scissors',
    # clip-applier
    'clip applier': 'clip_applier', 'large clip': 'clip_applier',
    'hem-o': 'clip_applier', 'hemolock': 'clip_applier',
    # hook
    'hook': 'cautery_hook', 'spatula': 'cautery_hook',
    'cautery hook': 'cautery_hook',
}

def canonical(raw: str) -> str | None:
    """free-text → canonical class or None"""
    s = raw.strip().lower()
    for tok, canon in TOK2CANON.items():
        if tok in s:
            return canon
    return None                        # unknown → drop row

def kp14_to_side7(kp14: dict, side: str):
    idxs = range(1, 8) if side == 'left' else range(8, 15)
    pts = []
    valid_count = 0
    
    for i in idxs:
        pt = kp14.get(i)
        if pt is not None:
            pts.append(pt)
            valid_count += 1
        else:
            pts.append([0, 0])  # Zero pad missing points
    
    if valid_count >= 5:  # Return if at least 5 valid keypoints exist
        return np.asarray(pts, np.float32)
    else:
        return None

def load_traj_table(csv_path: Path):
    """
    traj_tools.csv  →  {'000007': ('grasper','scissors'), …}
    rows whose tools cannot be mapped are ignored.
    """
    table = {}
    with csv_path.open() as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            tid = f'{int(r["traj_id"]):06d}'
            l = canonical(r['left_tool'])
            r_ = canonical(r['right_tool'])
            if l and r_:
                table[tid] = (l, r_)
    return table

# ------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--surgroot', default=Path('datasets/SurgPose'), help='SurgPose root directory')
    ap.add_argument('--traj_csv', default=Path('datasets/SurgPose/traj_tools.csv'), help='traj_id ↔ tool CSV')
    ap.add_argument('--out',      type=Path, default='datasets/SurgPoseYOLO/surgpose_yolo_05')
    ap.add_argument('--fps',      type=float,  default=0.5, help='FPS for YOLO-Pose dataset')
    ap.add_argument('--val-split',type=float,default=0.20, help='validation split ratio')
    args = ap.parse_args()

    random.seed(0)
    TRAJ2CLS = load_traj_table(args.traj_csv)
    
    # Determine video splits first
    all_tids = [p.name.zfill(6) for p in args.surgroot.iterdir() if p.is_dir() and p.name.zfill(6) in TRAJ2CLS]
    random.shuffle(all_tids)
    val_size = int(len(all_tids) * args.val_split)
    val_tids = set(all_tids[:val_size])
    
    (args.out/'images/train').mkdir(parents=True, exist_ok=True)
    (args.out/'images/val').mkdir(exist_ok=True)
    (args.out/'labels/train').mkdir(parents=True, exist_ok=True)
    (args.out/'labels/val').mkdir(exist_ok=True)

    for case in tqdm(sorted(p for p in args.surgroot.iterdir() if p.is_dir()),
                     desc='cases'):
        tid  = case.name.zfill(6)
        if tid not in TRAJ2CLS:
            continue
        cls_left, cls_right = TRAJ2CLS[tid]
        cid_left, cid_right = NAME2ID[cls_left], NAME2ID[cls_right]

        # Determine split based on video ID
        split = 'val' if tid in val_tids else 'train'
        print(f'\n Processing {tid} ({split})')
        
        vid = case/'regular'/'left_video.mp4'
        kp  = case/'keypoints_left.yaml'
        bb  = case/'bbox_left.json'
        if not (vid.exists() and kp.exists() and bb.exists()):
            continue

        # --- load annotations -------------------------------------------------
        kp_yaml = yaml.safe_load(kp.read_text())
        kp_yaml = {int(k):v for k,v in kp_yaml.items()}
        bb_json = {int(k):v for k,v in json.loads(bb.read_text()).items()}

        cap = cv2.VideoCapture(str(vid))
        every_n = max(int(cap.get(cv2.CAP_PROP_FPS)//args.fps),1)
        fidx = -1
        while True:
            ok, frame = cap.read(); fidx += 1
            if not ok: break
            if fidx % every_n: continue
            if fidx not in kp_yaml or fidx not in bb_json: continue

            img_dir, lbl_dir = args.out/f'images/{split}', args.out/f'labels/{split}'
            stem   = f'{tid}_{fidx:06d}'
            cv2.imwrite(str(img_dir/f'{stem}.jpg'), frame)
            h,w = frame.shape[:2]

            with open(lbl_dir/f'{stem}.txt','w') as lf:
                for side,(cid,obj_key) in (('left', (cid_left, 'obj1')),
                                           ('right',(cid_right,'obj2'))):
                    if obj_key not in bb_json[fidx]: continue
                    box   = bb_json[fidx][obj_key]     # [x,y,w,h]
                    kps7  = kp14_to_side7(kp_yaml[fidx], side)
                    if kps7 is None: continue
                    xc,yc = (box[0]+box[2]/2)/w, (box[1]+box[3]/2)/h
                    bw,bh = box[2]/w, box[3]/h
                    # YOLO-Pose: class xc yc bw bh  x1 y1 v1 x2 y2 v2 ...
                    line  = [cid, xc, yc, bw, bh]
                    for x,y in kps7:
                        visibility = 0 if (x == 0 and y == 0) else 2
                        line += [x/w, y/h, visibility]
                    lf.write(' '.join(f'{v:.6f}' if i else str(int(v))
                                      for i,v in enumerate(line))+'\n')
        cap.release()
    
    # --- write dataset YAML ------------------------------------------------
    out_yaml = args.out/OUT_YAML
    with out_yaml.open('w') as f:
        yaml.dump({
            'path': str(args.out),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(TOOLS),
            'names': TOOLS,
            'kpt_shape': KPT_SHAPE,
            'kpt_names': [f'k{i+1}' for i in range(KPT_SHAPE[0])],
        }, f, sort_keys=False, default_flow_style=False)

if __name__ == '__main__':
    main()
