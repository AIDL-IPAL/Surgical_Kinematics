# ---------------------------------------------------------------------
# dlc_build_and_train.py
# ---------------------------------------------------------------------
"""
Turn your existing JSON annotations into a DeepLabCut project and train
a network (DLC-ResNet-50) for N-key-point laparoscopic instruments.

• Reads:
      instrumentation.cfg      (key-point names / colours)
      annotations/<time>/*.json  +  images/             (your exporter)

• Creates / writes:
      dlc_project/   (config.yaml, labeled-data, training-datasets, …)
      dlc_project/training-datasets/iteration-0/UnaugmentedDataSet_*/ ...
"""

import json, ast, configparser, shutil, datetime
from pathlib import Path
from itertools import chain
from typing   import List, Dict
import pandas as pd
from tqdm import tqdm
import deeplabcut as dlc

# ------------------------------------------------------------------ paths
ANN_DIR = Path("annotations/20250610_121622")         # <-- adjust
CFG_INI = Path("instrumentation.cfg")
PROJECT_ROOT = Path("dlc_project")                    # will be (re)created
VIDEO_OR_IMG_ROOT = ANN_DIR / "images"               # images were copied here

# ------------------------------------------------------------------ 1/4  read key-point names
cp = configparser.ConfigParser(); cp.read(CFG_INI)
KP_LABELS: List[str] = ast.literal_eval(cp["keypoints"]["labels"])
NUM_KP = len(KP_LABELS)

# ------------------------------------------------------------------ 2/4  convert COCO JSONs → DLC dataframe(s)
def coco_to_dlc_df(json_dir: Path, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_rows = []
    header = ["scorer"] + list(chain(*[(lbl+"_x", lbl+"_y", lbl+"_likelihood")
                                       for lbl in KP_LABELS]))
    header.insert(0, "filename")

    for js_file in tqdm(sorted(json_dir.glob("*.json")), desc="Convert"):
        frame = Path(js_file.stem + ".png")  # your exporter uses .png
        with open(js_file) as f:
            insts = json.load(f)             # list of instruments (we use 1st)

        if not insts:   # empty frame
            continue
        xy = insts[0]["keypoints"]           # [[x,y], …]
        vis= insts[0].get("visibility", [2]*NUM_KP)
        # DLC likelihood = 1.0 if v==2 else 0.01 (barely visible)
        lk = [1.0 if v == 2 else 0.01 for v in vis]

        row = [frame.as_posix()]
        for (x,y),l in zip(xy, lk):
            row += [x, y, l]
        csv_rows.append(row)

    df = pd.DataFrame(csv_rows, columns=header)
    csv_path = out_dir / "CollectedData_user.csv"
    df.to_csv(csv_path, index=False)
    return df

df = coco_to_dlc_df(ANN_DIR, PROJECT_ROOT / "labeled-data" / "frames")
print("DataFrame shape:", df.shape)

# ------------------------------------------------------------------ 3/4  create / update the DLC project
if PROJECT_ROOT.exists():
    print("Project exists – will re-use")
    config_path = next(PROJECT_ROOT.glob("config.yaml"))
else:
    config_path = dlc.create_new_project(
        project="LapInstrumentPose",
        experimenter="user",
        videofile_path=None,            # we’ll add images not video
        working_directory=str(PROJECT_ROOT),
        copy_videos=False)

# --- inject bodyparts & skeleton into config.yaml
import ruamel.yaml as yaml
yaml = yaml.YAML()
cfg = yaml.load(open(config_path))

cfg["bodyparts"] = KP_LABELS

# DLC “skeleton” is a list of lists [[bpA,bpB], …]
cfg["skeleton"] = [[KP_LABELS[i], KP_LABELS[i+1]]
                   for i in range(NUM_KP-1)]

cfg["project_path"] = str(PROJECT_ROOT.resolve())
cfg["numframes2pick"] = 0                   # we already labelled frames
cfg["TrainingFraction"] = [0.95]            # 95 % train / 5 % val
cfg["net_type"] = "resnet_50"
cfg["batch_size"] = 4
cfg["maxiters"]   = 40000                   # small set – quick train

yaml.dump(cfg, open(config_path, "w"))
print("✓ updated config.yaml")

# ------------------------------------------------------------------ 4/4  create dataset + train
dlc.convertcsv2h5(config_path)
dlc.create_training_dataset(config_path, net_type=cfg["net_type"])
dlc.train_network(config_path,
                  shuffle=1,
                  displayiters=100,
                  saveiters=1000,
                  maxiters=cfg["maxiters"],
                  gputouse=0)          # set -1 for CPU

print("Training done – model in dlc_project/dlc-models/")
