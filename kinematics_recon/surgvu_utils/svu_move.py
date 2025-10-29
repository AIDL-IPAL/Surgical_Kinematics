#!/usr/bin/env python3
"""
collect_svup2_parts.py
Move/copy all ...part_XXX.mp4 files for cases in [start, end] into:
  <root>/svup2_case_<start>_<end>

Example:
  python collect_svup2_parts.py --root /path/to/surgvu24 --start 50 --end 100
  python collect_svup2_parts.py --root "C:\datasets\surgvu24" --start 50 --end 100 --mode copy
"""

import argparse, re, shutil
from pathlib import Path

CASE_RE  = re.compile(r"case_(\d+)", re.IGNORECASE)
PART_RE  = re.compile(r"part_(\d+)", re.IGNORECASE)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path, help="Folder containing the files (e.g., surgvu24)")
    ap.add_argument("--start", required=True, type=int, help="Start case number (inclusive)")
    ap.add_argument("--end",   required=True, type=int, help="End case number (inclusive)")
    ap.add_argument("--part",  type=int, default=2, help="Part index to select (default: 2)")
    ap.add_argument("--mode",  choices=["move","copy"], default="move", help="Move or copy (default: move)")
    ap.add_argument("--dry-run", action="store_true", help="Print actions only")
    args = ap.parse_args()

    root = args.root.resolve()
    assert root.is_dir(), f"Not a directory: {root}"

    out_dir = root / f"svup2_case_{args.start}_{args.end}"
    out_dir.mkdir(parents=True, exist_ok=True)

    target_part_tag = f"part_{args.part:03d}"

    picked = 0
    for p in root.glob("*.mp4"):  # files were flattened; adjust to rglob if needed
        name = p.name

        # must contain the target part tag
        if target_part_tag not in name.lower():
            # fall back to regex in case of non-zero-padded formats
            mpart = PART_RE.search(name)
            if not (mpart and int(mpart.group(1)) == args.part):
                continue

        # extract first case number
        mcase = CASE_RE.search(name)
        if not mcase:
            continue
        case_num = int(mcase.group(1))
        if not (args.start <= case_num <= args.end):
            continue

        dest = out_dir / name
        print(f"{args.mode.upper()}: {p} -> {dest}")
        if not args.dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if args.mode == "copy":
                shutil.copy2(p, dest)
            else:
                shutil.move(str(p), str(dest))
        picked += 1

    print(f"\nDone. Selected {picked} file(s) to {out_dir}")

if __name__ == "__main__":
    main()
