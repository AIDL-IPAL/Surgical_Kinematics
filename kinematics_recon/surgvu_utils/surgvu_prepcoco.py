#!/usr/bin/env python3
# prep_surgvu_coco.py
"""
Convert the full SurgVU (videos + *_gc.json) into COCO-Detection.

Result:
datasets/SurgVU/
   ├── train/               ← JPGs
   ├── val/
   └── annotations/
         ├── train_coco_det.json
         └── val_coco_det.json
"""

import json, cv2, numpy as np, argparse, itertools
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import configparser, ast

# ------------------------------- categories (reuse instrumentation.cfg)
cfg = configparser.ConfigParser()
cfg.read("instrumentation.cfg")
TOOL_TYPES = ast.literal_eval(cfg["tools"]["types"])
CLASSES    = [TOOL_TYPES[k] for k in sorted(TOOL_TYPES)]

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="datasets/SurgVU")
    ap.add_argument("--train-split", type=float, default=0.8)
    return ap.parse_args()

# ------------------------------- main
def main():
    args          = parse_args()
    src_root      = args.root
    dst_root      = src_root         # keep outputs inside same tree
    (dst_root/"train").mkdir(parents=True, exist_ok=True)
    (dst_root/"val").mkdir(parents=True, exist_ok=True)
    (dst_root/"annotations").mkdir(exist_ok=True)

    coco_train = {"images": [], "annotations": [], "categories": []}
    coco_val   = {"images": [], "annotations": [], "categories": []}
    for cid, name in enumerate(CLASSES, 1):
        coco_train["categories"].append(dict(id=cid, name=name, supercategory="tool"))
    coco_val["categories"] = coco_train["categories"]

    img_id_train = img_id_val = ann_id_train = ann_id_val = 0
    rng = np.random.default_rng(0)

    # crawl every case_***/ sub-folder
    gc_files = sorted(src_root.glob("case_*/*_gc.json"))
    if not gc_files:
        raise SystemExit("❌  No *_gc.json files found – check folder structure")

    for gc_path in tqdm(gc_files, desc="Parsing SurgVU"):
        video_name = gc_path.stem.rsplit("_gc", 1)[0]
        video_path = gc_path.with_suffix(".mp4")
        if not video_path.exists():
            tqdm.tqdm.write(f"⚠  video missing → {video_path.name}")
            continue

        with open(gc_path) as f:
            js = json.load(f)
        frame2boxes = defaultdict(list)
        for box in js["boxes"]:
            parts = box["name"].split("_")
            if len(parts) < 4 or parts[0]!="slice" or parts[1]!="nr":
                continue
            frame_idx = int(parts[2])
            tool      = "_".join(parts[3:]).lower().strip()
            frame2boxes[frame_idx].append((tool, box["corners"]))

        # pick train/val frames
        frames = sorted(frame2boxes)
        rng.shuffle(frames)
        split = int(len(frames)*args.train_split)
        train_frames = set(frames[:split])

        cap = cv2.VideoCapture(str(video_path))
        for fidx in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ok, frame = cap.read()
            if not ok:
                continue
            h,w = frame.shape[:2]
            subdir, img_id, ann_id, coco = (
                ("train", img_id_train, ann_id_train, coco_train)
                if fidx in train_frames else
                ("val"  , img_id_val  , ann_id_val  , coco_val)
            )
            # write jpg
            fname = f"{video_name}_f{fidx:06d}.jpg"
            cv2.imwrite(str(dst_root/subdir/fname), frame)

            coco["images"].append(dict(id=img_id, file_name=fname, height=h, width=w))

            for tool_name, corners in frame2boxes[fidx]:
                try:
                    cat_id = CLASSES.index(tool_name)+1
                except ValueError:
                    continue
                xs, ys = zip(*corners)
                x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
                coco["annotations"].append(dict(
                    id       = ann_id,
                    image_id = img_id,
                    category_id = cat_id,
                    bbox     = [x0, y0, x1-x0, y1-y0],
                    area     = float((x1-x0)*(y1-y0)),
                    iscrowd  = 0
                ))
                ann_id += 1

            if fidx in train_frames:
                img_id_train = img_id + 1
                ann_id_train = ann_id
            else:
                img_id_val   = img_id + 1
                ann_id_val   = ann_id

        cap.release()

    # save JSON
    with open(dst_root/"annotations"/"train_coco_det.json","w") as f:
        json.dump(coco_train, f)
    with open(dst_root/"annotations"/"val_coco_det.json","w") as f:
        json.dump(coco_val, f)

    print(f"\n✓ COCO export done: "
          f"{len(coco_train['images'])} train imgs / "
          f"{len(coco_val['images'])} val imgs")

if __name__ == "__main__":
    import json
    main()
