#!/usr/bin/env python3
"""
Convert a (subset of) SurgPose to COCO-Keypoints 1.0

Folder layout that is expected:

    000000/
        regular/left_video.mp4
        bbox_left.json
        keypoints_left.yaml
    000001/
        …

Only the *left* camera is used here.

Key-points  1-7  → left instrument
Key-points  8-14 → right instrument   (seven key-points each)

The bounding–box JSON already stores [x, y, w, h] in pixel units.
"""

import argparse, csv, json, yaml, random, shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import cv2, numpy as np
import configparser, ast
import itertools

# ──────────────────────────────────────────────────────────────────────────────
# read the shared instrumentation.cfg                                          
cfg = configparser.ConfigParser()
cfg.read("pose_estimation/instrumentation.cfg")          # relative to repo root

KEYPOINT_LABELS = ast.literal_eval(cfg["keypoints"]["labels"])
N_KP            = len(KEYPOINT_LABELS)                   # should be 7

# one single COCO category “instrument”
CATEGORIES = [dict(
    id        = 1,
    name      = "instrument",
    keypoints = KEYPOINT_LABELS,
    skeleton  = [[i, i + 1] for i in range(N_KP - 1)],
)]

# ──────────────────────────────────────────────────────────────────────────────
def load_keypoints_yaml(path: Path) -> dict[int, dict[str, np.ndarray | None]]:
    """
    YAML → {frame_idx: {'left': (7,2) | None , 'right': (7,2) | None}}
    Valid only if *all* 7 key-points are present (no gaps / NaNs).
    """
    with open(path) as f:
        raw = yaml.safe_load(f)

    out = {}
    for fidx, kp_dict in raw.items():
        fidx = int(fidx)

        left  = [kp_dict.get(i) for i in range(1,  N_KP + 1)]
        right = [kp_dict.get(i) for i in range(N_KP + 1, 2 * N_KP + 1)]

        def _ok(arr):          # exactly 7 entries and none missing
            return None not in arr and len(arr) == N_KP

        out[fidx] = dict(
            left  = np.asarray(left , np.float32) if _ok(left)  else None,
            right = np.asarray(right, np.float32) if _ok(right) else None,
        )
    return out



def load_bbox_json(path: Path) -> dict[int, dict[str, list]]:
    """
    JSON → {frame_idx: {'obj1': [x,y,w,h] | None, 'obj2': […] | None}}
    """
    with open(path) as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}


def write_coco(dst_root: Path, split: str,
               images: list[dict], annotations: list[dict]) -> None:
    (dst_root / "annotations").mkdir(parents=True, exist_ok=True)
    coco = dict(
        info        = dict(version="surgpose-coco"),
        images      = images,
        annotations = annotations,
        categories  = CATEGORIES,
    )
    with open(dst_root / "annotations" / f"{split}.json", "w") as f:
        json.dump(coco, f, indent=2)


# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="datasets/SurgPose", type=Path)
    ap.add_argument("--dst", default="datasets/SurgPose_COCO", type=Path)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--fps", type=int, default=1,
                    help="sampling rate in frames / second (default 1)")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    (args.dst / "train").mkdir(parents=True, exist_ok=True)
    (args.dst / "val").mkdir(exist_ok=True)

    img_id = ann_id = 0
    imgs_train, imgs_val = [], []
    anns_train, anns_val = [], []

    folders = sorted([p for p in args.src.iterdir() if p.is_dir() and p.name.isdigit()])

    for f in tqdm(folders, desc="folders"):
        vid = f / "regular" / "left_video.mp4"
        kp_yaml = f / "keypoints_left.yaml"
        bbox_js = f / "bbox_left.json"
        if not (vid.exists() and kp_yaml.exists() and bbox_js.exists()):
            tqdm.write(f"⚠ {f.name} missing one of the required files – skipped")
            continue

        kps = load_keypoints_yaml(kp_yaml)
        bbs = load_bbox_json(bbox_js)

        cap = cv2.VideoCapture(str(vid))
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        every_n = max(int(src_fps // args.fps), 1)

        frame_idx = -1
        while True:
            ok, frame = cap.read()
            frame_idx += 1
            if not ok:
                break
            if frame_idx % every_n:
                continue            # down-sample to requested fps
            if frame_idx not in kps or frame_idx not in bbs:
                continue

            for side, kp_arr, bb_key in (("left", kps[frame_idx]["left"], "obj1"),
                                         ("right", kps[frame_idx]["right"], "obj2")):

                if (kp_arr is None or bbs[frame_idx].get(bb_key) is None):
                    continue
                # train/val split – uniform random per annotation
                split = "val" if rng.random() < args.val_split else "train"
                dst_split = args.dst / split

                fname = f"{f.name}_{side[0]}_{frame_idx:06d}.jpg"
                cv2.imwrite(str(dst_split / fname), frame)
                h, w = frame.shape[:2]

                # --- COCO image record ---------------------------------------
                img_rec = dict(
                    id=img_id, file_name=f"{split}/{fname}", width=w, height=h
                )
                if split == "train":
                    imgs_train.append(img_rec)
                else:
                    imgs_val.append(img_rec)

                # --- annotation ----------------------------------------------
                x, y, bw, bh = bbs[frame_idx][bb_key]
                if bw <= 0 or bh <= 0 or kp_arr.shape[0] != N_KP:
                    continue
                flat_kps = list(itertools.chain.from_iterable(
                    ([float(px), float(py), 2.0] for px, py in kp_arr)
                ))
                ann_rec = dict(
                    id=ann_id, image_id=img_id, category_id=1,
                    keypoints=flat_kps,
                    num_keypoints=N_KP,
                    bbox=[x, y, bw, bh],
                    area=float(bw * bh),
                    iscrowd=0,
                )
                if split == "train":
                    anns_train.append(ann_rec)
                else:
                    anns_val.append(ann_rec)

                img_id += 1
                ann_id += 1

        cap.release()

    # ---------- write out ----------------------------------------------------
    write_coco(args.dst, "train", imgs_train, anns_train)
    write_coco(args.dst, "val",   imgs_val,   anns_val)
    print(f"✓ done – train imgs {len(imgs_train)}, val imgs {len(imgs_val)}")

def unzip_folder(src: Path, dst: Path) -> None:
    """
    Unzip all .zip files in the source directory to the destination directory.
    """
    import zipfile
    for zip_file in src.glob("*.zip"):
        with zipfile.ZipFile(zip_file, 'r') as zf:
            zf.extractall(dst)
            print(f"Unzipped {zip_file.name} to {dst}")

if __name__ == "__main__":
    # for zip_file in Path("datasets/SurgPose").glob("*.zip"):
    #     unzip_folder(Path("datasets/SurgPose"), Path("datasets/SurgPose"))
    main()
