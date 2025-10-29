#!/usr/bin/env python3
"""
Turn SurgVU tool-usage CSVs into **presence-only** COCO JSON.

• For every video frame whose timestamp falls inside an
  (install_time  … uninstall_time) interval we create **one
  annotation per tool** with bbox==full-frame (0,0,W,H).

Later you can fine-tune with your precise boxes.  The detector will
already know the class logits, so convergence is much faster.

Example:
    python pose_estimation/surgvu_csv2coco.py  \
        --video-root  datasets/SurgVU/videos \
        --label-root  datasets/SurgVU/labels \
        --out-dir     datasets/SurgVU/SurgVU_presence
"""
import csv, json, cv2, pathlib, argparse, itertools
from collections import defaultdict
from tqdm import tqdm
import configparser, ast
import numpy as np

# ---------------------- helpers ---------------------------------------------
def find_parts(case_dir: pathlib.Path):
    """Return [(part_idx:int, video_path:Path, n_frames:int)] sorted by part."""
    parts = []
    for p in case_dir.glob(f"{case_dir.name}_video_part_*.mp4"):
        part_idx = int(p.stem.rsplit("_part_")[-1])          # 001 → 1
        cap = cv2.VideoCapture(str(p)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
        parts.append((part_idx, p, n))
    return sorted(parts)

def split_interval(st, et, s_part, e_part, part_lens, fps):
    """Yield (part_idx, start_sec, end_sec) possibly split across parts."""
    if s_part == e_part:
        yield s_part, st, et
    else:                           # spans two consecutive parts only (rare)
        yield s_part, st, part_lens[s_part]/fps             # until end of first
        yield e_part , 0.0, et                              # from 0 in next
# -----------------------------------------------------------------------------

# --------------------------------------------------------------------- classes
cfg = configparser.ConfigParser()
cfg.read("pose_estimation/instrumentation.cfg")                       # same file you already use
TOOL_TYPES = ast.literal_eval(cfg["tools"]["types"])
CLASSES = ['grasper', 'scissors', 'clip_applier', 'cautery_hook']
CID_BY_NAME = {name: i + 1 for i, name in enumerate(CLASSES)}

# ------------------------------------------------------------------  name map
# tokens must be lower-case and trimmed
MAP_TOKENS = {
    'grasper'      : ['prograsp', 'grasper', 'cadiere'],
    'scissors'     : ['scissor', 'scissors', 'maryland'],
    'clip_applier' : ['clip applier', 'large clip', 'hem-o-lok', 'hemolock'],
    'cautery_hook'         : ['cautery hook', 'hook', 'spatula'],
}

# reverse lookup  token → canonical   (only for the 4 canonical names)
CANON_BY_TOKEN = {tok.strip(): canon
                  for canon, toks in MAP_TOKENS.items()
                  for tok in toks}

def hms_to_sec(t: str) -> float:
    """
    Convert timestamp like '07:24.8' or '0:41:58' to seconds.
    Works for mm:ss.s  or  hh:mm:ss
    """
    parts = t.split(":")
    parts = [float(p) for p in parts]
    if len(parts)==2:
        m,s = parts
        return m*60+s
    elif len(parts)==3:
        h,m,s = parts
        return h*3600+m*60+s
    raise ValueError(f"Bad time string: {t}")

# --------------------------------------------------------------------- main
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-root", required=True, type=pathlib.Path)
    ap.add_argument("--label-root", required=True, type=pathlib.Path)
    ap.add_argument("--out-dir", required=True,  type=pathlib.Path)
    ap.add_argument("--fps", type=float, default=25.0,
                    help="real frame-rate of source videos (default 25 fps)")
    ap.add_argument("--split", type=float, default=.8,
                    help="train split fraction")
    ap.add_argument("--sample-fps", type=float, default=0.1,
                help="output sampling rate (frames / sec), default 1")

    return ap.parse_args()

def main():
    args       = parse_args()
    out_root   = args.out_dir
    (out_root/"train").mkdir(parents=True, exist_ok=True)
    (out_root/"val").mkdir(exist_ok=True)
    (out_root/"annotations").mkdir(exist_ok=True)

    coco_train = {"images": [], "annotations": [], "categories": []}
    coco_val   = {"images": [], "annotations": [], "categories": []}
    for cid,name in enumerate(CLASSES,1):
        for c in (coco_train, coco_val):
            c["categories"].append({"id":cid, "name":name, "supercategory":"tool"})

    img_id = ann_id = 0
    rng = np.random.default_rng(0)

    csv_files = sorted(args.label_root.glob("case_*/tools*.csv"))
    if not csv_files:
        raise SystemExit("❌ No tools.csv found – check --label-root")

    for csv_path in tqdm(csv_files, desc="CSV"):
        case_dir = csv_path.parent
        case     = case_dir.name
        parts    = find_parts(args.video_root/case)             # all mp4s

        if not parts:
            tqdm.write(f"⚠ no mp4 for {case}")
            continue

        part_len = {idx: n for idx,_,n in parts}                # frames per part

        # ── read CSV → {part → {cid:[(t0,t1),…]}} ─────────────
        part_intervals = defaultdict(lambda: defaultdict(list))
        with open(csv_path, newline="") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                raw = (row["groundtruth_toolname"] or
                       row["commercial_toolname"]).lower()

                tool = next((CANON_BY_TOKEN[tok] for tok in CANON_BY_TOKEN if tok in raw),
                            None)                       # None → ignore this row
                if tool is None:
                    continue

                cid = CID_BY_NAME[tool]                 # 1…4
                
                # --- timestamps --------------------------------------------------
                s_t = hms_to_sec(row["install_case_time"])
                e_t = hms_to_sec(row["uninstall_case_time"])

                # --- part numbers (robust to "1" or "1.0") -----------------------
                try:
                    s_p = int(float(row["install_case_part"]))
                    e_p = int(float(row["uninstall_case_part"]))
                except (ValueError, TypeError):
                    continue        # skip malformed row

                # --- split across video parts if needed --------------------------
                for p, t0, t1 in split_interval(s_t, e_t, s_p, e_p, part_len, args.fps):
                    part_intervals[p][cid].append((t0, t1))

        # ── iterate over every video-part separately ────────────────────────────────
        for part_idx, vid_path, n_frames in parts:

            ivals = part_intervals.get(part_idx, {})
            if not ivals:
                continue                          # nothing active in this part

            cap  = cv2.VideoCapture(str(vid_path))

            native_fps   = cap.get(cv2.CAP_PROP_FPS) or args.fps   # fallback
            sample_fps   = getattr(args, 'sample_fps', 1)          # 1 fps default
            step         = max(int(round(native_fps / sample_fps)), 1)

            # indices we will **actually decode**
            sample_idx   = set(range(0, n_frames, step))

            n_train      = int(len(sample_idx) * args.split)
            train_mask   = np.full(len(sample_idx), False)
            train_mask[:n_train] = True
            rng.shuffle(train_mask)

            sample_iter  = enumerate(sorted(sample_idx))
            train_iter   = iter(train_mask)

            for fidx, orig_idx in sample_iter:
                cap.set(cv2.CAP_PROP_POS_FRAMES, orig_idx)
                ok, frame = cap.read()
                if not ok:
                    break

                sec = orig_idx / native_fps                     # timestamp (s)
                active = [cid for cid, intervals in ivals.items()
                          if any(a <= sec <= b for a, b in intervals)]
                if not active:
                    continue                                    # no tool visible

                sub   = "train" if next(train_iter) else "val"
                fname = f"{case}_p{part_idx:02d}_{orig_idx:06d}.jpg"
                cv2.imwrite(str(out_root / sub / fname), frame)

                h, w  = frame.shape[:2]
                coco  = coco_train if sub == "train" else coco_val
                coco["images"].append(dict(id=img_id,
                                           file_name=fname,
                                           height=h, width=w))

                for cid in active:                             # one ann / tool
                    coco["annotations"].append(dict(
                        id=ann_id, image_id=img_id, category_id=cid,
                        bbox=[0, 0, w, h], area=float(w * h), iscrowd=0))
                    ann_id += 1
                img_id += 1
            cap.release()

    # save JSON
    with open(out_root/"annotations"/"train_coco_det.json","w") as f:
        json.dump(coco_train, f)
    with open(out_root/"annotations"/"val_coco_det.json","w") as f:
        json.dump(coco_val, f)
    print(f"✓ Pseudo-COCO written   "
          f"train imgs: {len(coco_train['images'])}   "
          f"val imgs: {len(coco_val['images'])}")

if __name__ == "__main__":
    import json, csv
    main()
