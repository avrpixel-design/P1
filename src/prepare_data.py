#!/usr/bin/env python3
"""
Generate deterministic, reproducible file lists for each dataset.

IMPORTANT — Relationship to published results:
    The results reported in the paper were generated using the default
    filesystem ordering returned by glob() on the original machine. To
    reproduce those exact results, the pre-generated file lists shipped
    in data/file_lists/ should be used (these were captured on the
    original system).

    If you are running from scratch on a new machine, this script can
    generate fresh file lists in two modes:

    --mode lock    (default) Record the glob ordering on THIS machine.
                   Results will match the paper only if the filesystem
                   ordering is identical (same OS, same extraction).

    --mode shuffle Apply a seeded shuffle for platform-independent
                   determinism. Results will be numerically close but
                   not identical to the paper (different 500-sample
                   subset).

Usage:
    # Lock current filesystem ordering (recommended on original machine)
    python src/prepare_data.py --data-dir data --max-samples 500

    # Platform-independent seeded shuffle (for fresh replication)
    python src/prepare_data.py --data-dir data --max-samples 500 --mode shuffle --seed 42
"""

import argparse
import os
import random
from glob import glob
from pathlib import Path


DATASET_DIRS = {
    "RAVDESS":   {"subdir": "RAVDESS",   "dimension": "emotional"},
    "IEMOCAP":   {"subdir": "IEMOCAP",   "dimension": "emotional"},
    "L2-ARCTIC": {"subdir": "L2-ARCTIC", "dimension": "linguistic"},
    "GMU":       {"subdir": "GMU",        "dimension": "linguistic",
                  "aliases": ["GMU-Accented Speech Archive",
                              "GMU Speech Accent Archive"]},
    "UA-Speech": {"subdir": "UA-Speech",  "dimension": "pathological"},
    "MDVR-KCL":  {"subdir": "MDVR-KCL",  "dimension": "pathological"},
}

AUDIO_EXTENSIONS = ("*.wav", "*.flac", "*.mp3")


def find_audio_files(dataset_path: str) -> list:
    """Recursively find all audio files under dataset_path."""
    files = []
    for ext in AUDIO_EXTENSIONS:
        files.extend(glob(os.path.join(dataset_path, "**", ext), recursive=True))
    # Sort for deterministic baseline before shuffle
    return sorted(files)


def resolve_dataset_path(data_dir: str, info: dict) -> str | None:
    """Resolve the dataset directory, trying aliases if primary doesn't exist."""
    primary = os.path.join(data_dir, info["subdir"])
    if os.path.isdir(primary):
        return primary
    for alias in info.get("aliases", []):
        alt = os.path.join(data_dir, alias)
        if os.path.isdir(alt):
            return alt
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate deterministic file lists for reproducibility."
    )
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Root data directory (default: data)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffle mode (default: 42)")
    parser.add_argument("--max-samples", type=int, default=500,
                        help="Maximum samples per dataset (default: 500)")
    parser.add_argument("--mode", type=str, default="lock",
                        choices=["lock", "shuffle"],
                        help="'lock': record glob order as-is (matches original machine). "
                             "'shuffle': seeded shuffle for cross-platform determinism.")
    args = parser.parse_args()

    out_dir = os.path.join(args.data_dir, "file_lists")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Data directory : {os.path.abspath(args.data_dir)}")
    print(f"Mode           : {args.mode}")
    if args.mode == "shuffle":
        print(f"Seed           : {args.seed}")
    print(f"Max samples    : {args.max_samples}")
    print(f"Output         : {os.path.abspath(out_dir)}/")
    print()

    summary = []

    for name, info in DATASET_DIRS.items():
        path = resolve_dataset_path(args.data_dir, info)
        if path is None:
            print(f"  [{name}] NOT FOUND — skipping")
            continue

        all_files = find_audio_files(path)
        if not all_files:
            print(f"  [{name}] No audio files found in {path}")
            continue

        if args.mode == "shuffle":
            rng = random.Random(args.seed)
            rng.shuffle(all_files)
        # else: 'lock' mode keeps sorted glob order (filesystem-deterministic)

        # Select up to max_samples
        selected = all_files[:args.max_samples]

        # Write file list (relative paths from data_dir)
        list_path = os.path.join(out_dir, f"{name}.txt")
        with open(list_path, "w") as f:
            for fpath in selected:
                rel = os.path.relpath(fpath, args.data_dir)
                f.write(rel + "\n")

        print(f"  [{name}] {len(selected)}/{len(all_files)} files → {list_path}")
        summary.append((name, info["dimension"], len(all_files), len(selected)))

    # Write summary
    print(f"\n{'='*60}")
    print(f"{'Dataset':<15} {'Dimension':<15} {'Total':>8} {'Selected':>10}")
    print(f"{'-'*60}")
    for name, dim, total, sel in summary:
        print(f"{name:<15} {dim:<15} {total:>8} {sel:>10}")
    print(f"{'='*60}")

    if args.mode == "lock":
        print(f"\nFile lists record the glob ordering on THIS machine.")
        print(f"Commit these to the repo for exact reproducibility.")
    else:
        print(f"\nFile lists use seeded shuffle (seed={args.seed}).")
        print(f"Results will be close but not identical to paper values.")

    print(f"\nRun experiment with: python src/run_experiment.py --file-lists {out_dir}")


if __name__ == "__main__":
    main()
