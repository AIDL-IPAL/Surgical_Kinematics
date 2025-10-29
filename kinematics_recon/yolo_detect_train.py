#!/usr/bin/env python
"""
Train a YOLOv8 Detector (boxes + class IDs) on a YOLO-format dataset.

Usage
-----
# train from a small checkpoint (fast, good baseline)
python pose_analytics/yolo_detect_train.py \
    --data  datasets/SurgPoseYOLO/detect/surgpose_det.yaml \
    --model yolo11s.pt \
    --epochs 150 --batch 16 --img 640

# or train a larger model for higher accuracy
python pose_analytics/yolo_detect_train.py \
    --data  datasets/SurgPoseYOLO/detect/surgpose_det.yaml \
    --model yolov12s.pt \
    --epochs 200 --batch 8 --img 640

in windows cmd:
python pose_analytics/yolo_detect_train.py --data datasets/SurgVU_YOLO/detect/encord_det.yaml --model yolo11s.pt

for finetune:
python pose_analytics/yolo_detect_train.py --data datasets/SurgVU_YOLO/detect/encord_det.yaml --model yolo11s.pt
"""
import argparse
import os
from datetime import datetime
from ultralytics import YOLO


def train(
    data: str,
    model: str = 'yolo11s.pt',
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    project: str | None = None,
    name: str | None = None,
    workers: int = 8,
):
    """
    Fire-and-forget wrapper around `YOLO.train` for detection.

    Parameters
    ----------
    data   : path to the dataset YAML (YOLO detect format)
    model  : checkpoint (.pt) or model YAML
    epochs : total epochs to train
    imgsz  : square image size
    batch  : global batch size
    project, name : optional custom run directory
    workers : DataLoader workers (set ≤ 4 on Windows laptops)
    """
    # Auto-generate project and name if not provided
    if project is None:
        project = 'pose_analytics/models'

    # Extract a clean run name from the model filename
    base = os.path.basename(model)
    model_name = base.replace('.pt', '').replace('.yaml', '')

    # Generate name with MMDD format
    date_str = datetime.now().strftime('%m%d')
    if name is None:
        name = f'{model_name}_{date_str}'
    else:
        name = f'{model_name}_{date_str}_{name}'

    model_obj = YOLO(model)  # load weights or build from .yaml

    model_obj.train(
        # core dataset / run params ------------------------------------------------
        data=data,                 task='detect',  pretrained=True,
        epochs=epochs,             imgsz=imgsz,    batch=batch,
        project=project,           name=name,      workers=workers,
        device=0,                  verbose=True,

        # ----   augmentation knobs (kept where applicable for detection)   ------
        mosaic=0.10,               # keep low during fine-tune
        close_mosaic=10,           # disable for final 10 epochs
        hsv_h=0.05,                # ±5 % hue
        hsv_s=0.40,                # ±40 % saturation
        hsv_v=0.50,                # ±50 % value/brightness
        # blur=0.20,                # 20 % chance Gaussian blur
        # noise=0.20,               # 20 % chance additive Gaussian noise
        erasing=0.20,              # random erasing can help robustness
        flipud=0.0,                # avoid vertical flip for surgical scenes
        fliplr=0.50,               # horizontal flip
        degrees=10,                # small rotation
        translate=0.10,            # slight x/y shift
        scale=0.35,                # multiscale
        multi_scale=True,          # allow multi-scale
        shear=0.0,  perspective=0.001,  # slight shear and perspective
        # mixup=0.05,              # optional, enable if beneficial
        copy_paste=0.0             # off by default here
    )


if __name__ == '__main__':
    # Windows / multiprocessing safety ─────────────────────────────────────
    import multiprocessing as mp
    import platform
    if platform.system() == 'Windows':
        mp.freeze_support()

    ap = argparse.ArgumentParser(description='YOLOv8 Detection training helper')
    ap.add_argument('--data',  default='datasets/SurgVU_YOLO/detect/encord_det.yaml',
                    help='dataset YAML (YOLO detect format)')
    ap.add_argument('--model', default='yolo11s.pt',
                    help='model checkpoint (.pt) or arch (.yaml)')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--img',    type=int, default=640, help='image size')
    ap.add_argument('--batch',  type=int, default=16)
    ap.add_argument('--workers',type=int, default=8)
    ap.add_argument('--project',default=None)
    ap.add_argument('--name',   default=None,
                    help='run name inside project/ (default auto)')
    args = ap.parse_args()

    train(
        data=args.data,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.img,
        batch=args.batch,
        project=args.project,
        name=args.name,
        workers=args.workers
    )
