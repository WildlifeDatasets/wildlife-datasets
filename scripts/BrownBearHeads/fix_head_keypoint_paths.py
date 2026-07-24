"""Fix BrownBearHeads head-keypoint sidecar paths.

The Kaggle dataset metadata uses paths like:

    BrownBearHeads/2017/Aardvark/598A2507_BLR_0_Aardvark.JPG

Some generated `head_keypoints.csv` files instead keep raw head-crop paths like:

    2017/images/598A2507_BLR_0_Aardvark.JPG

This script rewrites `head_keypoints.csv["path"]` so it exactly matches
`metadata.csv["path"]`, using the image year and filename as the link key.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pandas as pd


def path_year(path: str) -> str:
    match = re.search(r"(20\d{2})", str(path))
    if match is None:
        raise ValueError(f"Could not find year in path: {path}")
    return match.group(1)


def path_filename(path: str) -> str:
    return os.path.basename(os.path.normpath(str(path)))


def path_keys(paths: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "year": paths.map(path_year),
            "filename": paths.map(path_filename),
        }
    )


def fix_paths(
    root: str | Path,
    metadata_file: str,
    keypoints_file: str,
    output_file: str | None,
    write: bool = True,
) -> pd.DataFrame:
    root = Path(root).expanduser()
    metadata_path = root / metadata_file
    keypoints_path = root / keypoints_file
    output_path = root / (output_file or keypoints_file)

    metadata = pd.read_csv(metadata_path, low_memory=False)
    keypoints = pd.read_csv(keypoints_path, low_memory=False)

    metadata_keys = path_keys(metadata["path"])
    keypoint_keys = path_keys(keypoints["path"])

    duplicate_metadata = metadata_keys.duplicated().sum()
    if duplicate_metadata:
        raise ValueError(f"metadata.csv has {duplicate_metadata} duplicate year+filename keys.")

    path_lookup = pd.Series(metadata["path"].to_numpy(), index=pd.MultiIndex.from_frame(metadata_keys))
    fixed_paths = path_lookup.reindex(pd.MultiIndex.from_frame(keypoint_keys))

    missing = fixed_paths.isna()
    if missing.any():
        examples = keypoints.loc[missing.to_numpy(), "path"].head(10).to_list()
        raise ValueError(
            f"Could not map {int(missing.sum())} head-keypoint paths to metadata.csv. Examples: {examples}"
        )

    keypoints = keypoints.copy()
    keypoints["path"] = fixed_paths.to_numpy()

    duplicate_output = keypoints["path"].duplicated().sum()
    if duplicate_output:
        raise ValueError(f"Fixed head_keypoints.csv would have {duplicate_output} duplicate paths.")

    if write:
        keypoints.to_csv(output_path, index=False)

    print(f"Metadata rows: {len(metadata):,}")
    print(f"Head-keypoint rows: {len(keypoints):,}")
    print(f"Path overlap after fix: {len(set(metadata['path']) & set(keypoints['path'])):,}")
    if write:
        print(f"Saved: {output_path}")
    return keypoints


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix BrownBearHeads head_keypoints.csv paths.")
    parser.add_argument("root", help="Kaggle BrownBearHeads root containing metadata.csv and head_keypoints.csv.")
    parser.add_argument("--metadata-file", default="metadata.csv", help="Metadata CSV filename.")
    parser.add_argument("--keypoints-file", default="head_keypoints.csv", help="Head-keypoint CSV filename.")
    parser.add_argument(
        "--output-file", default=None, help="Output filename. Defaults to overwriting head_keypoints.csv."
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print summary without writing.")
    args = parser.parse_args()

    if args.dry_run:
        fixed = fix_paths(args.root, args.metadata_file, args.keypoints_file, args.output_file, write=False)
        print(f"Dry run OK. First fixed path: {fixed.loc[0, 'path']}")
    else:
        fix_paths(args.root, args.metadata_file, args.keypoints_file, args.output_file)
