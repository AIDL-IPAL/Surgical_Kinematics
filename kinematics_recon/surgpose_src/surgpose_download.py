#!/usr/bin/env python3
"""
Download & unpack the public SurgPose‐80 dataset from Zenodo.
Default target:  datasets/SurgPose
"""

import argparse, pathlib, tarfile, urllib.request, io, sys, shutil, time
import requests, math
from tqdm.auto import tqdm

URL   = ("https://zenodo.org/api/records/15278516/files-archive"
         "?filename=SurgPose80_dataset.tar.gz")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", type=pathlib.Path,
                    default=pathlib.Path("ProjectSurgeryHernia/datasets/SurgPose"),
                    help="destination root directory")
    return ap.parse_args()

def download_with_progress(url: str, dst: pathlib.Path, chunk=1024 * 1024):
    """Download *url* → *dst* with a tqdm progress bar (1 MiB chunks)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        bar   = tqdm(total=total, unit="B", unit_scale=True,
                     unit_divisor=1024, desc=f"↓ {dst.name}", ncols=80)
        with open(dst, "wb") as f:
            for chunk_bytes in r.iter_content(chunk_size=chunk):
                if chunk_bytes:                # filter out keep-alive chunks
                    f.write(chunk_bytes)
                    bar.update(len(chunk_bytes))
        bar.close()
    return dst



# ---------------------------------------------------------------------
def main():
    args = parse_args()
    dst  = args.dst.expanduser()
    dst.mkdir(parents=True, exist_ok=True)

    tmp_tar = dst / "tmp_surgpose.tar.gz"
    if tmp_tar.exists() and tmp_tar.stat().st_size > 0:
        print("✔ archive already present – skipping download")
    else:
        print("⬇  downloading (one-off ≈ 21 GB)")
        download_with_progress(URL, tmp_tar)
        
    import zipfile
    print("✂  unpacking …")

    # Read the first 2 bytes to sniff the format
    with open(tmp_tar, "rb") as f:
        magic = f.read(2)

    if magic == b"\x1f\x8b":
        # It's gzip-compressed → tar.gz
        with tarfile.open(tmp_tar, "r:gz") as tar:
            for m in tqdm(tar.getmembers(), desc="extract", unit="file", ncols=80):
                tar.extract(m, dst)

    elif magic == b"PK":
        # It's a zip file
        with zipfile.ZipFile(tmp_tar, "r") as zip_ref:
            for m in tqdm(zip_ref.infolist(), desc="extract", unit="file", ncols=80):
                zip_ref.extract(m, dst)

    else:
        raise ValueError("Unknown archive format")

    print(f"✓ dataset ready at  {dst.resolve()}")
    tmp_tar.unlink()

def unzip_subpacks(dst: pathlib.Path):
    """
    Unzip all sub-packages in the given directory.
    This is useful if the dataset contains multiple zip files.
    """
    import zipfile
    for subpack in dst.glob("*.zip"):
        print(f"Unzipping {subpack.name} ...")
        with zipfile.ZipFile(subpack, "r") as zip_ref:
            zip_ref.extractall(dst)
        subpack.unlink()  # Remove the zip file after extraction

if __name__ == "__main__":
    # main()
    unzip_subpacks(pathlib.Path("datasets/SurgPose"))