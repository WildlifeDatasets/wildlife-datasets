"""Prepare BrownBearHeads metadata from the Zenodo release.

The script reads all original split CSVs from the BrownBearHeads release,
merges them into one dataframe, deduplicates repeated images, and writes a
single metadata table that supports all dataset splits we want to expose later.

The output contains one row per image path together with:

- basic metadata such as identity, date, year, and camera
- optional face keypoints matched by filename
- the six original yearly split definitions: `split_2017` ... `split_2022`
- two derived standardized splits: `split_ood` and `split_iid`

"""

from __future__ import annotations

import argparse
import json
import os
import re
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

YEARS = [2017, 2018, 2019, 2020, 2021, 2022]


def split_csv_patterns(root: str) -> list[str]:
    """Return all known locations where split CSVs may exist."""

    return [
        os.path.join(root, "data", "reid_annotations", "test_on_*", "*.csv"),
        os.path.join(root, "data", "train_test_reid", "test_on_*", "*.csv"),
        os.path.join(root, "reid_annotations", "test_on_*", "*.csv"),
        os.path.join(root, "train_test_reid", "test_on_*", "*.csv"),
    ]


def find_split_csvs(root: str) -> list[str]:
    """Find and return all split CSV files under the release root."""

    csv_paths: list[str] = []
    for pattern in split_csv_patterns(root):
        csv_paths.extend(glob(pattern))

    csv_paths = sorted(set(csv_paths))
    if not csv_paths:
        raise FileNotFoundError(f"No BrownBearHeads split CSVs found under {root}")
    return csv_paths


def load_raw_metadata(root: str) -> pd.DataFrame:
    """Load all split CSVs and append split provenance columns."""

    frames = []
    csv_paths = find_split_csvs(root)
    for csv_path in tqdm(csv_paths, desc="Loading split CSVs", unit="file"):
        frame = pd.read_csv(csv_path, low_memory=False)
        frame["experiment"] = os.path.basename(os.path.dirname(csv_path))
        frame["split_name"] = os.path.splitext(os.path.basename(csv_path))[0]
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def join_unique(series: pd.Series):
    """Join unique non-null values into one stable pipe-separated string."""

    values = sorted(set(series.dropna().astype(str)))
    return "|".join(values) if values else pd.NA


def first_keypoints(series: pd.Series):
    """Return the first keypoint list, preserving all-NaN annotations."""

    for value in series:
        if isinstance(value, (list, tuple, np.ndarray)):
            return list(value)
    return pd.NA


def normalize_basename(path: str) -> str:
    """Normalize a path to the comparable image basename used by annotation files."""

    return os.path.basename(os.path.normpath(str(path))).lower()


def superface_source_basename(path: str) -> str:
    """Map `superface_data` filenames back to the original head-crop basename."""

    basename = os.path.basename(os.path.normpath(str(path)))
    stem, suffix = os.path.splitext(basename)
    if stem.startswith("bear_"):
        stem = stem[len("bear_") :]
    stem = re.sub(r"_\d+$", "", stem)
    return f"{stem}{suffix}"


def keypoint_json_patterns(root: str) -> list[str]:
    """Return known BrownBear face-keypoint annotation locations."""

    return [
        os.path.join(root, "data", "preprocessing_annotations", "coco_bearface_new", "annotations", "*.json"),
        os.path.join(root, "data", "GT", "coco_bearface_new", "annotations", "*.json"),
        os.path.join(root, "preprocessing_annotations", "coco_bearface_new", "annotations", "*.json"),
        os.path.join(root, "GT", "coco_bearface_new", "annotations", "*.json"),
        os.path.join(root, "data", "preprocessing_annotations", "superface_data", "annotations", "*.json"),
        os.path.join(root, "data", "GT", "superface_data", "annotations", "*.json"),
        os.path.join(root, "preprocessing_annotations", "superface_data", "annotations", "*.json"),
        os.path.join(root, "GT", "superface_data", "annotations", "*.json"),
    ]


def coco_xy(keypoints: list[float]) -> list[float]:
    """Convert COCO [x, y, v] keypoints into flattened [x, y] coordinates."""

    values = []
    for x, y, visibility in np.asarray(keypoints, dtype=float).reshape(-1, 3):
        if visibility <= 0 or x < 0 or y < 0:
            values.extend([np.nan, np.nan])
        else:
            values.extend([float(x), float(y)])
    return values


def load_keypoint_file(path: str) -> pd.DataFrame:
    """Load one COCO-style face-keypoint JSON as basename-indexed rows."""

    with open(path) as file:
        data = json.load(file)

    images = {image["id"]: image for image in data.get("images", [])}
    annotations = data.get("annotations", [])
    if not images or not annotations:
        return pd.DataFrame(columns=["basename"])

    parts = set(Path(path).parts)
    is_superface = "superface_data" in parts
    column = "keypoints_13" if is_superface else "keypoints_5"

    records = []
    for annotation in tqdm(annotations, desc=f"Reading {os.path.basename(path)}", unit="ann", leave=False):
        image = images.get(annotation["image_id"])
        if image is None:
            continue
        file_name = image.get("file_name", "")
        basename = superface_source_basename(file_name) if is_superface else os.path.basename(file_name)
        records.append(
            {
                "basename": normalize_basename(basename),
                column: coco_xy(annotation.get("keypoints", [])),
            }
        )

    return pd.DataFrame(records)


def load_keypoint_annotations(root: str) -> pd.DataFrame:
    """Load all available BrownBear face-keypoint annotations."""

    paths: list[str] = []
    for pattern in keypoint_json_patterns(root):
        paths.extend(glob(pattern))
    paths = sorted(set(paths))
    if not paths:
        return pd.DataFrame(columns=["basename"])

    frames = [load_keypoint_file(path) for path in tqdm(paths, desc="Loading keypoint JSONs", unit="file")]
    frames = [frame for frame in frames if len(frame)]
    if not frames:
        return pd.DataFrame(columns=["basename"])

    data = pd.concat(frames, ignore_index=True)
    grouped = data.groupby("basename", dropna=False)
    return grouped.apply(
        lambda frame: pd.Series(
            {
                "keypoints_5": first_keypoints(frame["keypoints_5"]) if "keypoints_5" in frame else pd.NA,
                "keypoints_13": first_keypoints(frame["keypoints_13"]) if "keypoints_13" in frame else pd.NA,
            }
        ),
        include_groups=False,
    ).reset_index()


def add_keypoint_annotations(data: pd.DataFrame, root: str) -> pd.DataFrame:
    """Attach available face keypoints to ReID rows by head-crop basename."""

    annotations = load_keypoint_annotations(root)
    if annotations.empty:
        return data

    data = data.copy()
    data["basename"] = data["image"].apply(normalize_basename)
    data = data.merge(annotations, on="basename", how="left")
    data.drop(columns=["basename"], inplace=True)

    has_keypoints_5 = data["keypoints_5"].apply(lambda value: isinstance(value, list))
    has_keypoints_13 = data["keypoints_13"].apply(lambda value: isinstance(value, list))
    data["keypoints"] = pd.NA
    data.loc[has_keypoints_13, "keypoints"] = data.loc[has_keypoints_13, "keypoints_13"]
    data.loc[has_keypoints_5, "keypoints"] = data.loc[has_keypoints_5, "keypoints_5"]
    return data.drop(columns=["keypoints_5", "keypoints_13"], errors="ignore")


def normalize_original_split(split_name: str) -> str:
    """Map original split-file names to train, val, or test."""

    mapping = {
        "train_iid": "train",
        "val_iid": "val",
        "test_iid": "test",
        "test_ood": "test",
    }
    if split_name not in mapping:
        raise ValueError(f"Unknown split name: {split_name}")
    return mapping[split_name]


def summarize_yearly_split(frame: pd.DataFrame, year: int):
    """Summarize one original yearly experiment for one image.

    Each image should have at most one role inside `test_on_<year>`.
    If the image does not appear in that experiment, the function returns NA.
    """

    experiment_name = f"test_on_{year}"
    subset = frame[frame["experiment"] == experiment_name]
    if subset.empty:
        return pd.NA

    values = sorted({normalize_original_split(name) for name in subset["split_name"]})
    if len(values) != 1:
        raise ValueError(f"Image appears in multiple split roles inside {experiment_name}: {values}")
    return values[0]


def assign_day_roles(
    days: list,
    rng: np.random.Generator,
    test_fraction: float,
    val_fraction: float,
) -> dict:
    """Assign whole days to train, val, and test without overlap."""

    days = list(days)
    n_days = len(days)
    if n_days == 0:
        return {}
    if n_days == 1:
        return {days[0]: "train"}

    shuffled_days = list(rng.permutation(days))

    n_test = min(max(int(round(test_fraction * n_days)), 1), n_days - 1)
    test_days = set(shuffled_days[:n_test])
    remaining_days = shuffled_days[n_test:]

    if len(remaining_days) <= 1:
        val_days = set()
    else:
        n_val = int(round(val_fraction * len(remaining_days)))
        n_val = min(max(n_val, 1), len(remaining_days) - 1)
        val_days = set(remaining_days[:n_val])

    mapping = {}
    for day in shuffled_days:
        if day in test_days:
            mapping[day] = "test"
        elif day in val_days:
            mapping[day] = "val"
        else:
            mapping[day] = "train"
    return mapping


def build_day_based_split(
    data: pd.DataFrame,
    forced_test_mask: pd.Series,
    seed: int,
    test_fraction: float,
    val_fraction: float,
) -> pd.Series:
    """Build a split by assigning full identity-days to subsets.

    Rows marked by `forced_test_mask` are always assigned to `test`.
    All remaining rows are split identity by identity using full days only.
    """

    split = pd.Series(index=data.index, dtype="object")
    split.loc[forced_test_mask] = "test"

    rng = np.random.default_rng(seed)
    remaining = data.loc[~forced_test_mask].copy()
    remaining["day"] = remaining["date"].dt.date

    groups = remaining.groupby("identity")
    n_groups = remaining["identity"].nunique(dropna=False)
    for identity, frame_identity in tqdm(groups, total=n_groups, desc="Building day split", unit="identity"):
        days = sorted(day for day in frame_identity["day"].dropna().unique())
        mapping = assign_day_roles(
            days=days,
            rng=rng,
            test_fraction=test_fraction,
            val_fraction=val_fraction,
        )
        split.loc[frame_identity.index] = frame_identity["day"].map(mapping).fillna("train").to_numpy()

    return split


def build_chronological_identity_split(
    data: pd.DataFrame,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> pd.Series:
    """Split each identity chronologically into train, val, and test by day.

    Unique observation days are sorted within each identity. The earliest chunk
    of days is assigned to train, the next chunk to val, and the latest chunk
    to test. All observations from the same identity-day stay together.
    """

    if not np.isclose(train_fraction + val_fraction + test_fraction, 1.0):
        raise ValueError("Train, val, and test fractions must sum to 1.")

    split = pd.Series(index=data.index, dtype="object")

    groups = data.groupby("identity")
    n_groups = data["identity"].nunique(dropna=False)
    for _, frame_identity in tqdm(groups, total=n_groups, desc="Building IID split", unit="identity"):
        frame_identity = frame_identity.copy()
        frame_identity["day"] = frame_identity["date"].dt.normalize()

        unique_days = sorted(day for day in frame_identity["day"].dropna().unique())
        n_days = len(unique_days)

        if n_days == 0:
            split.loc[frame_identity.index] = "train"
            continue
        if n_days == 1:
            split.loc[frame_identity.index] = "train"
            continue

        n_train = int(round(train_fraction * n_days))
        n_val = int(round(val_fraction * n_days))

        n_train = min(max(n_train, 1), n_days)
        remaining_after_train = n_days - n_train
        if remaining_after_train <= 0:
            n_train = n_days - 1
            remaining_after_train = 1

        n_val = min(max(n_val, 0), remaining_after_train - 1 if remaining_after_train > 1 else 0)
        n_test = n_days - n_train - n_val

        if n_test <= 0:
            if n_val > 0:
                n_val -= 1
            else:
                n_train -= 1
            n_test = n_days - n_train - n_val

        day_roles = {}
        for day in unique_days[:n_train]:
            day_roles[day] = "train"
        for day in unique_days[n_train : n_train + n_val]:
            day_roles[day] = "val"
        for day in unique_days[n_train + n_val :]:
            day_roles[day] = "test"

        split.loc[frame_identity.index] = frame_identity["day"].map(day_roles).fillna("train").to_numpy()

    return split


def add_base_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """Create normalized metadata columns needed for merging and splitting."""

    data = raw.copy()
    data["identity"] = data["id"].astype(str)
    data["date"] = pd.to_datetime(data.get("timestamp"), errors="coerce")

    if "year" in data.columns:
        data["year"] = data["year"].fillna(data["date"].dt.year)
    else:
        data["year"] = data["date"].dt.year

    return data


def add_standardized_splits(data: pd.DataFrame) -> pd.DataFrame:
    """Add the derived `split_ood` and `split_iid` columns."""

    data = data.copy()
    split_ood = pd.Series("train", index=data.index, dtype="object")
    split_ood.loc[data["year"] == 2021] = "val"
    split_ood.loc[data["year"] == 2022] = "test"
    data["split_ood"] = split_ood
    data["split_iid"] = build_chronological_identity_split(
        data=data,
        train_fraction=0.6,
        val_fraction=0.1,
        test_fraction=0.3,
    )
    return data


def merge_image_rows(data: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated split rows into one row per original image path."""

    grouped = data.groupby("image", dropna=False)

    rows = []
    n_groups = data["image"].nunique(dropna=False)
    for image, frame in tqdm(grouped, total=n_groups, desc="Merging image rows", unit="image"):
        rows.append(
            pd.Series(
                {
                    "identity": join_unique(frame["identity"]),
                    "path": image,
                    "path_original": image,
                    "date": frame["date"].min(),
                    "year": frame["year"].min(),
                    "camera": join_unique(frame["camera"]) if "camera" in frame.columns else pd.NA,
                    "keypoints": first_keypoints(frame["keypoints"]) if "keypoints" in frame else pd.NA,
                    "split_2017": summarize_yearly_split(frame, 2017),
                    "split_2018": summarize_yearly_split(frame, 2018),
                    "split_2019": summarize_yearly_split(frame, 2019),
                    "split_2020": summarize_yearly_split(frame, 2020),
                    "split_2021": summarize_yearly_split(frame, 2021),
                    "split_2022": summarize_yearly_split(frame, 2022),
                    "split_ood": join_unique(frame["split_ood"]),
                    "split_iid": join_unique(frame["split_iid"]),
                }
            )
        )

    merged = pd.DataFrame(rows).reset_index(drop=True)
    merged = merged.sort_values(["year", "identity", "path"], na_position="last").reset_index(drop=True)
    merged.insert(0, "image_id", range(len(merged)))
    return merged


def clean_metadata(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn raw split CSV rows into one merged metadata table."""

    data = add_base_columns(raw)
    data = add_standardized_splits(data)
    return merge_image_rows(data)


def has_keypoints(value) -> bool:
    """Return whether a dataframe value contains a keypoint list."""

    return isinstance(value, (list, tuple, np.ndarray))


def print_keypoint_coverage(df: pd.DataFrame) -> None:
    """Print how many rows received keypoint annotations."""

    if "keypoints" not in df.columns:
        return

    n_keypoints = int(df["keypoints"].apply(has_keypoints).sum())
    n_total = len(df)
    coverage = 100 * n_keypoints / n_total if n_total else 0
    print(f"Matched keypoints for {n_keypoints} / {n_total} rows ({coverage:.2f}%).")
    if n_keypoints < n_total:
        print("Note: Zenodo provides sparse GT/preprocessing keypoint annotations, not keypoints for every ReID image.")


def main(
    root: str = "./Public_release",
    output_csv: str = "scripts/BrownBearHeads/metadata.csv",
    include_keypoints: bool = False,
) -> pd.DataFrame:
    """Merge all split CSVs and save one clean metadata file."""

    raw = load_raw_metadata(root)
    if include_keypoints:
        raw = add_keypoint_annotations(raw, root)
    merged = clean_metadata(raw)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)

    print(f"Saved {len(merged)} rows to {output_csv}")
    if include_keypoints:
        print_keypoint_coverage(merged)
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge BrownBearHeads split CSVs into one metadata file.")
    parser.add_argument("root", nargs="?", default="~/Public_release")
    parser.add_argument("output_csv", nargs="?", default="~/BrownBearHeads/metadata.csv")
    parser.add_argument(
        "--include-keypoints",
        action="store_true",
        help="Attach sparse Zenodo ground-truth keypoints to intermediate metadata.",
    )
    args = parser.parse_args()
    main(
        root=os.path.expanduser(args.root),
        output_csv=os.path.expanduser(args.output_csv),
        include_keypoints=args.include_keypoints,
    )
