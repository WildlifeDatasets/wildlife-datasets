"""Merge BrownBearHeads split CSVs into one metadata file.

The script reads all original split CSVs from the BrownBearHeads release,
merges them into one dataframe, deduplicates repeated images, and writes a
single metadata table that supports all dataset splits we want to expose later.

The output contains one row per image path together with:

- basic metadata such as identity, date, year, and camera
- the six original yearly split definitions: `split_2017` ... `split_2022`
- two derived standardized splits: `split_ood` and `split_iid`

"""

from __future__ import annotations

import argparse
import os
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

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
    for csv_path in find_split_csvs(root):
        frame = pd.read_csv(csv_path, low_memory=False)
        frame["experiment"] = os.path.basename(os.path.dirname(csv_path))
        frame["split_name"] = os.path.splitext(os.path.basename(csv_path))[0]
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def join_unique(series: pd.Series):
    """Join unique non-null values into one stable pipe-separated string."""

    values = sorted(set(series.dropna().astype(str)))
    return "|".join(values) if values else pd.NA


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

    for identity, frame_identity in remaining.groupby("identity"):
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

    for _, frame_identity in data.groupby("identity"):
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

    merged = grouped.apply(
        lambda frame: pd.Series(
            {
                "identity": join_unique(frame["identity"]),
                "path_original": frame.name,
                "date": frame["date"].min(),
                "year": frame["year"].min(),
                "camera": join_unique(frame["camera"]) if "camera" in frame.columns else pd.NA,
                "split_2017": summarize_yearly_split(frame, 2017),
                "split_2018": summarize_yearly_split(frame, 2018),
                "split_2019": summarize_yearly_split(frame, 2019),
                "split_2020": summarize_yearly_split(frame, 2020),
                "split_2021": summarize_yearly_split(frame, 2021),
                "split_2022": summarize_yearly_split(frame, 2022),
                "split_ood": join_unique(frame["split_ood"]),
                "split_iid": join_unique(frame["split_iid"]),
            }
        ),
        include_groups=False,
    )

    merged = merged.reset_index(drop=True)
    return merged.sort_values(["year", "identity", "path_original"], na_position="last").reset_index(drop=True)


def clean_metadata(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn raw split CSV rows into one merged metadata table."""

    data = add_base_columns(raw)
    data = add_standardized_splits(data)
    return merge_image_rows(data)


def main(
    root: str = "./Public_release",
    output_csv: str = "scripts/BrownBearHears/clean_metadata.csv",
) -> pd.DataFrame:
    """Merge all split CSVs and save one clean metadata file."""

    raw = load_raw_metadata(root)
    merged = clean_metadata(raw)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)

    print(f"Saved {len(merged)} rows to {output_csv}")
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge BrownBearHeads split CSVs into one metadata file.")
    parser.add_argument("root", nargs="?", default="~/Public_release")
    parser.add_argument("output_csv", nargs="?", default="~/BrownBearHears/clean_metadata.csv")
    args = parser.parse_args()
    main(root=args.root, output_csv=args.output_csv)
