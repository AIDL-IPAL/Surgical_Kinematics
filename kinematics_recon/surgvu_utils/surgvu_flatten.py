#!/usr/bin/env python3
"""
flatten_surgvu.py — Move or copy all .mp4 files from subdirectories of a root
                    directory (e.g., surgvu24/) into the root, renaming to
                    avoid collisions, and optionally removing empty folders.

Usage:
  python flatten_surgvu.py --root /path/to/surgvu24
  python flatten_surgvu.py --root "C:\datasets\surgvu24" --mode copy --dry-run
  python flatten_surgvu.py --root /path/to/surgvu24 --pattern "*.mp4" --no-delete-empty
"""

import argparse
import shutil
from pathlib import Path

def uniquify(target: Path) -> Path:
    """Append __1, __2, ... before the suffix until a free name is found."""
    if not target.exists():
        return target
    stem, suffix = target.stem, target.suffix
    i = 1
    while True:
        candidate = target.with_name(f"{stem}__{i}{suffix}")
        if not candidate.exists():
            return candidate
        i += 1

def planned_target(root: Path, src: Path, sep: str = "__", use_full_rel: bool = True) -> Path:
    """
    Build a destination filename that includes the relative parent path as a prefix,
    so collisions across cases/cameras don't overwrite.
    e.g., case_0001/camera0/video.mp4 -> case_0001__camera0__video.mp4
    """
    rel_parent = src.parent.relative_to(root)
    if use_full_rel and rel_parent.parts:
        prefix = sep.join(rel_parent.parts)
        new_name = f"{prefix}{sep}{src.name}"
    elif rel_parent.parts:
        # Only first-level dir (e.g., case_0001) as prefix
        new_name = f"{rel_parent.parts[0]}{sep}{src.name}"
    else:
        new_name = src.name
    return root / new_name

def flatten(root: Path,
            pattern: str = "*.mp4",
            mode: str = "move",
            dry_run: bool = False,
            delete_empty: bool = True,
            sep: str = "__",
            use_full_rel: bool = True) -> None:
    root = root.resolve()
    assert root.is_dir(), f"Root does not exist or is not a directory: {root}"

    src_files = [p for p in root.rglob(pattern) if p.is_file() and p.parent != root]

    print(f"Found {len(src_files)} file(s) matching {pattern} under {root}")
    moved = 0
    copied = 0
    skipped = 0

    for src in src_files:
        dest = planned_target(root, src, sep=sep, use_full_rel=use_full_rel)
        dest = uniquify(dest)  # ensure no collision
        action = "COPY" if mode == "copy" else "MOVE"
        print(f"{action}: {src} -> {dest}")
        if not dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)  # dest is the root, but safe
            try:
                if mode == "copy":
                    shutil.copy2(src, dest)
                    copied += 1
                else:
                    # Use shutil.move for cross-filesystem safety
                    shutil.move(str(src), str(dest))
                    moved += 1
            except Exception as e:
                print(f"  ! Error on {action.lower()} '{src}': {e}")
                skipped += 1

    # Optionally remove empty directories (only sensible if we moved)
    if delete_empty and mode == "move":
        # Remove deepest directories first
        dirs = sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda d: len(d.parts), reverse=True)
        for d in dirs:
            try:
                d.rmdir()
                print(f"REMOVED EMPTY DIR: {d}")
            except OSError:
                # Not empty or can't remove—skip
                pass

    print("\nSummary:")
    print(f"  Moved : {moved}")
    print(f"  Copied: {copied}")
    print(f"  Skipped/Errors: {skipped}")
    print("Done.")

def main():
    parser = argparse.ArgumentParser(description="Flatten mp4 files into the root directory (e.g., surgvu24).")
    parser.add_argument("--root", required=True, type=Path, help="Path to the root folder (e.g., surgvu24).")
    parser.add_argument("--pattern", default="*.mp4", help="Glob pattern for files to flatten (default: *.mp4).")
    parser.add_argument("--mode", choices=["move", "copy"], default="move",
                        help="Move (default) or copy files.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without making changes.")
    parser.add_argument("--no-delete-empty", action="store_true",
                        help="Do not remove empty directories afterward.")
    parser.add_argument("--sep", default="__", help="Separator used in new filenames (default: __).")
    parser.add_argument("--first-level-only", action="store_true",
                        help="Prefix with only the first-level folder (e.g., case_0001) instead of full relative path.")
    args = parser.parse_args()

    flatten(
        root=args.root,
        pattern=args.pattern,
        mode=args.mode,
        dry_run=args.dry_run,
        delete_empty=not args.no_delete_empty,
        sep=args.sep,
        use_full_rel=not args.first_level_only
    )

if __name__ == "__main__":
    main()
