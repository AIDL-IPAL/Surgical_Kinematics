#!/usr/bin/env python3
"""
Fine-tune **RTMPose-S** on SurgPose-COCO (7 key-points / instrument).

Example
-------
python train_sp_rtmpose.py \
    --data-root  datasets/SurgPose_COCO \
    --work-dir   work_dirs/rtmpose_surgpose
"""
import argparse
from pathlib import Path
from importlib.resources import files

import torch
from mmengine.config import Config
from mmengine.runner import Runner
import mmpose, os
from mmpose.utils import register_all_modules

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 7-point template used in the converter ─────────────────────────────
KP_LABELS = [
    "tip", "jaw_left", "jaw_right",
    "wrist", "shaft_mid", "trocar", "shaft_base"
]
NUM_KP = len(KP_LABELS)                            # 7

def build_cfg(data_root: str, work_dir: str) -> Config:
    """Create an MMPose config tailored to SurgPose."""

    MMPOSEROOT = Path(__file__).resolve().parent.parent / 'mmpose'

    cfg_file = MMPOSEROOT / 'configs/body_2d_keypoint/rtmpose/coco' \
                           / 'rtmpose-s_8xb256-420e_coco-256x192.py'

    base = Config.fromfile(str(cfg_file))

    # ──────── metainfo ──────────────────────────────────────────────
    metainfo = dict(
        dataset_name   = 'surgpose',
        num_keypoints  = NUM_KP,
        keypoint_info  = {
            i: dict(id=i, name=n, color=[255, 0, 0], type='', swap='')
            for i, n in enumerate(KP_LABELS)
        },
        skeleton_info  = {
            i: dict(id=i, link=(KP_LABELS[i], KP_LABELS[i + 1]),
                    color=[0, 255, 0])
            for i in range(NUM_KP - 1)
        },
        flip_pairs     = [(1, 2)],              # jaw_left ↔ jaw_right
        flip_indices   = [0, 2, 1, 3, 4, 5, 6], # 7 elements!
        joint_weights  = [1.] * NUM_KP,
        sigmas         = [0.05] * NUM_KP
    )

    def dl_cfg(split: str):
        return dict(
            type       = 'CocoDataset',
            data_root  = data_root,
            ann_file   = f'annotations/{split}.json',
            data_prefix= dict(img=f'{split}/'),
            metainfo   = metainfo
        )

    # replace the datasets ----------------------------------------------------
    base.train_dataloader.update(batch_size=32, num_workers=4,
                                 dataset=dl_cfg('train'))
    base.val_dataloader.update(batch_size=32, num_workers=4,
                               dataset=dl_cfg('val'))
    base.test_dataloader.dataset = dl_cfg('val')     # reuse for testing
    base.val_evaluator.ann_file  = (Path(data_root) /
                                    'annotations/val.json').as_posix()

    # model head / test cfg ---------------------------------------------------
    base.model.head.out_channels      = NUM_KP
    base.model.test_cfg.flip_test     = False        # faster evaluation

    # training schedule -------------------------------------------------------
    base.max_epochs = base.train_cfg.max_epochs = 120     # ~30 min on 1×A100
    # shorten the cosine lr schedule
    for sch in base.param_scheduler:
        if getattr(sch, 'end', None) is not None:
            sch.end = base.max_epochs

    # misc --------------------------------------------------------------------
    base.work_dir          = work_dir
    base.default_scope     = 'mmpose'
    base.randomness        = dict(seed=42, deterministic=False)
    base.visualizer.enable = False                     # no GUI pop-ups

    # you can optionally warm-start from the COCO weights shipped in the base
    # config (faster convergence); otherwise leave it as is
    # base.load_from = 'https://download.openmmlab.com/mmpose/v1/.../xxx.pth'

    return base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True,
                    help='Folder created by the SurgPose → COCO script')
    ap.add_argument('--work-dir',  required=True,
                    help='Where checkpoints & logs will be written')
    args = ap.parse_args()

    cfg = build_cfg(args.data_root, args.work_dir)
    register_all_modules()
    Runner.from_cfg(cfg).train()


if __name__ == '__main__':
    main()
