#!/usr/bin/env python
"""
Train a YOLOv8-Pose model on the SurgPose-YOLO dataset.

Usage
-----
# train from the Nano-Pose checkpoint (fastest)
python yolo_pose_train.py \
    --data  datasets/SurgPoseYOLO/pose/surgpose_pose.yaml \
    --model yolov12s-pose.pt \
    --epochs 150 --batch 16 --img 640

# or train the 'small' pose model for higher accuracy
python yolo_pose_train.py \
    --data  datasets/SurgPoseYOLO/pose/surgpose_pose.yaml \
    --model yolov12s-pose.pt \
    --epochs 200 --batch 8 --img 640

in windows cmd:
python pose_analytics/yolo_pose_train.py --data datasets/SurgPoseYOLO/pose.yaml --model yolo11l-pose.pt

for finetune:
python pose_analytics/yolo_pose_train.py --data datasets/SurgVU_YOLO/pose/encord_pose.yaml --model yolo11s-pose.pt --epochs 2500
"""
import argparse, os
from datetime import datetime
from ultralytics import YOLO

def train(data: str,
          model: str = 'yolov11s-pose.pt',
          epochs: int = 100,
          imgsz: int = 640,
          batch: int = 16,
          project: str | None = None,
          name: str | None = None,
          workers: int = 8,
          ):
    """
    Fire-and-forget wrapper around `YOLO.train`.

    Parameters
    ----------
    data   : path to the dataset YAML (created by the converter)
    model  : checkpoint or YAML defining the pose backbone
    epochs : total epochs to train
    imgsz  : square image size (will be auto-scaled inside)
    batch  : global batch size
    project, name : optional custom run directory, otherwise YOLO defaults
    workers : DataLoader workers (set ≤ 4 on Windows laptops)
    class_weights : whether to use automatic class weighting for imbalance
    """
    # Auto-generate project and name if not provided
    if project is None:
        project = 'pose_analytics/models'
    
    # Extract model size from model string (n, s, m, l)
    model_name = os.path.basename(model).replace('-pose.pt', '').replace('.pt', '')

    # Generate name with MMDD format
    date_str = datetime.now().strftime('%m%d')
    if name is None:
        name = f'{model_name}_{date_str}'
    else:
        name = f'{model_name}_{date_str}_{name}'
    
    model = YOLO(model)                  # loads weights (or build from .yaml)

    model.train(
        # core dataset / run params ------------------------------------------------
        data=data,                 task='pose',   pretrained=True,
        epochs=epochs,             imgsz=imgsz,   batch=batch,
        project=project,           name=name,     workers=workers,
        device=0,                  verbose=True,  patience=500,

        # ----   augmentation knobs   --------------------------------------------
        mosaic=0.10,               # keep low during fine-tune
        close_mosaic=10,           # disable for final 10 epochs
        hsv_h=0.05,               # ±5 % hue
        hsv_s=0.40,                # ±40 % saturation
        hsv_v=0.50,                # ±50 % value/brightness
        # blur=0.20,                 # 20 % chance Gaussian blur
        # noise=0.20,                # 20 % chance additive Gaussian noise
        erasing=0.20,              # small random cut-outs (specular / smoke)
        flipud=0.0,                # no vertical flip
        fliplr=0.50,               # keep horizontal flip
        degrees=10,                # small roll-rotation
        translate=0.10,            # slight x/y shift
        scale=0.5,                # 0.5-1.5 multiscale (default)
        multi_scale=True,          # allow multi-scale TODO NEW
        shear=0.0,  perspective=0.0005,   # slight shear and perspective
        # mixup=0.05,                # light MixUp
        copy_paste=0.0             # keep off for pose
    )


if __name__ == '__main__':
    # Windows / multiprocessing safety ─────────────────────────────────────
    import multiprocessing as mp, platform
    if platform.system() == 'Windows':
        mp.freeze_support()

    ap = argparse.ArgumentParser(description='YOLOv8-Pose training helper')
    ap.add_argument('--data',  default='datasets/SurgVU_YOLO/pose/encord_pose.yaml',
                    help='dataset YAML (yolo format)')
    ap.add_argument('--model', default='yolo11l-pose.pt',
                    help='model checkpoint (.pt) or arch (.yaml)')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--img',    type=int, default=640, help='image size')
    ap.add_argument('--batch',  type=int, default=16)
    ap.add_argument('--workers',type=int, default=8)
    ap.add_argument('--project',default=None)
    ap.add_argument('--name',   default=None,
                    help='run name inside project/ (default auto)')
    args = ap.parse_args()


    train(data=args.data,
          model=args.model,
          epochs=args.epochs,
          imgsz=args.img,
          batch=args.batch,
          project=args.project,
          name=args.name,
          workers=args.workers)
