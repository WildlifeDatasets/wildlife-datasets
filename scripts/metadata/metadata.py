"""Create docs/csv/metadata.csv with the coverage of optional columns for each dataset.

The dataframes of all datasets are cached as pickles in `root_dataframes`. Each cell of
the output contains the fraction of images with a non-missing value in the column.
"""

import argparse
import os
import pickle

import pandas as pd
from tqdm import tqdm

from wildlife_datasets.datasets import names_all

ignore_datasets = [
    "BalearicLizardSegmented",
    "BirdIndividualIDSegmented",
    "SealIDSegmented",
    "AnimalCLEF2025",
    "AnimalCLEF2026",
    "WildlifeReID10k",
]
dataset_kwargs = {
    "BrownBearHeads": {"load_keypoints": True},
    "CHIRP": {"load_keypoints": True, "load_segmentation": True},
    "Melops": {"load_keypoints": True, "bbox": "head"},
}
columns_drop = ["image_id", "identity", "path", "hf_index", "age_group", "original_identity"]
columns_order = [
    "bbox",
    "segmentation",
    "split",
    "date",
    "encounter_id",
    "orientation",
    "keypoints",
    "species",
    "video",
    "frame_number",
    "camera",
    "age",
    "gender",
    "image_quality",
    "latitude/longitude",
    "location",
]


def cache_dataframes(root_data: str, root_dataframes: str, overwrite: bool) -> None:
    os.makedirs(root_dataframes, exist_ok=True)
    for dataset_class in tqdm(names_all):
        name = dataset_class.display_name()
        file_name = os.path.join(root_dataframes, f"{name}.pkl")
        if name in ignore_datasets or (os.path.exists(file_name) and not overwrite):
            continue

        dataset = dataset_class(os.path.join(root_data, name), **dataset_kwargs.get(name, {}))
        with open(file_name, "wb") as handle:
            pickle.dump(dataset.metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_dataframes(root_dataframes: str) -> dict[str, pd.DataFrame]:
    metadatas = {}
    for dataset_class in names_all:
        name = dataset_class.display_name()
        if name in ignore_datasets:
            continue
        with open(os.path.join(root_dataframes, f"{name}.pkl"), "rb") as file:
            metadatas[name] = pickle.load(file)
    return dict(sorted(metadatas.items(), key=lambda x: x[0].lower()))


def coverage(metadata: pd.DataFrame, column: str) -> float:
    if column == "split":
        return float(any(col == "split" or col.startswith("split_") for col in metadata.columns))
    elif column not in metadata.columns:
        return 0.0
    return metadata[column].notna().mean()


def create_table(metadatas: dict[str, pd.DataFrame], min_count: int = 2) -> pd.DataFrame:
    cols_list = []
    for metadata in metadatas.values():
        cols_list.extend({"split" if col.startswith("split_") else col for col in metadata.columns})
    cols = pd.Series(cols_list).value_counts()
    columns = [col for col in cols[cols >= min_count].index if col not in columns_drop]

    table = pd.DataFrame(
        {name: {column: coverage(metadata, column) for column in columns} for name, metadata in metadatas.items()}
    ).T
    if "latitude" in table and "longitude" in table:
        table["latitude/longitude"] = table[["latitude", "longitude"]].min(axis=1)
        table = table.drop(columns=["latitude", "longitude"])
    ordered = [col for col in columns_order if col in table]
    table = table[ordered + sorted(col for col in table if col not in ordered)]
    table.index.name = "dataset"
    return table.round(3).replace(0, pd.NA)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-data", default="/data/wildlife_datasets/data")
    parser.add_argument("--root-dataframes", default="/data/wildlife_datasets/dataframes")
    parser.add_argument("--output", default="docs/csv/metadata.csv")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cache_dataframes(args.root_data, args.root_dataframes, args.overwrite)
    table = create_table(load_dataframes(args.root_dataframes))
    table.to_csv(args.output)
    print(f"Saved {table.shape[0]} datasets and {table.shape[1]} columns to {args.output}")
