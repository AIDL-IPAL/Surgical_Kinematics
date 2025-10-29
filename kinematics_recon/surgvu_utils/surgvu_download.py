#!/usr/bin/env python3
# download_surgvu.py
"""
Grab SurgVU-24 from the public bucket and unzip it.

$ python download_surgvu.py  [/your/target/dir]
"""

import sys, shutil, zipfile, pathlib, requests, tqdm, os, hashlib

URL   = "https://storage.googleapis.com/isi-surgvu/surgvu24_videos_only.zip"
CHUNK = 1 << 20                     # 1 MiB
DEST  = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("datasets/SurgVU")

DEST.mkdir(parents=True, exist_ok=True)
zip_path = DEST / "surgvu24.zip"

# --------------------------------------------------------------------- download
if not zip_path.exists():
    print(f"⬇  Downloading SurgVU24 →  {zip_path} …")
    with requests.get(URL, stream=True, timeout=30) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with zip_path.open("wb") as f, tqdm.tqdm(total=total, unit="B", unit_scale=True, desc=zip_path.name) as bar:
            for chunk in r.iter_content(chunk_size=CHUNK):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
else:
    print(f"✓ ZIP already present – {zip_path.stat().st_size/1e6:.1f} MB")

# --------------------------------------------------------------------- unzip
extract_flag = not any((DEST / "videos").iterdir()) if (DEST / "videos").exists() else True
if extract_flag:
    print("🗜  Unzipping …")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(path=DEST)
    print("✓ Unzip done.")
else:
    print("✓ Videos already extracted – skipping unzip.")

print(f"\nRoot folder: {DEST.resolve()}")
print("Sub-folders now look like:")
for p in sorted((DEST).iterdir()):
    print("   •", p.relative_to(DEST))
