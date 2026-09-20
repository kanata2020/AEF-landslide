"""Fit Sentinel-2/DEM normalization on the selected training partition."""

import argparse
from pathlib import Path

from landslide_benchmark.cross_region import fit_optical_norm
from landslide_benchmark.data import load_split
from landslide_benchmark.paths import DEFAULT_SPLIT, DEFAULT_AEF_DIR, DEFAULT_S2_DIR, DEFAULT_NORM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-json", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--aef-dir", type=Path, default=DEFAULT_AEF_DIR)
    parser.add_argument("--s2-dir", type=Path, default=DEFAULT_S2_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_NORM)
    args = parser.parse_args()
    events = load_split(args.split_json, args.aef_dir, args.s2_dir)["train"]
    fit_optical_norm(events, args.output)
    print(f"Saved training-only normalization: {args.output}")


if __name__ == "__main__":
    main()
