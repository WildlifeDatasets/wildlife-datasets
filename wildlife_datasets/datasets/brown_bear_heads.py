import os

import numpy as np
import pandas as pd

from .datasets import WildlifeDataset
from .downloads import DownloadKaggle

summary = {
    "licenses": "Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)",
    "licenses_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    "url": "https://www.kaggle.com/datasets/picekl/brown-bear-heads",
    "publication_url": "https://doi.org/10.1016/j.cub.2025.12.022",
    "cite": "rosenberg2026individual",
    "animals": {"brown bear"},
    "animals_simple": "bears",
    "real_animals": True,
    "year": 2026,
    "reported_n_total": 72939,
    "reported_n_individuals": 109,
    "wild": True,
    "clear_photos": True,
    "pose": "multiple",
    "unique_pattern": False,
    "from_video": False,
    "cropped": True,
    "span": "5.1 years",
    "size": 6740,
}


class BrownBearHeads(DownloadKaggle, WildlifeDataset):
    summary = summary
    kaggle_url = "picekl/brown-bear-heads"
    kaggle_type = "datasets"
    head_keypoints_file = "head_keypoints.csv"

    @staticmethod
    def parse_keypoints(keypoints: pd.DataFrame) -> pd.DataFrame:
        point_indices = sorted({int(column.split("_")[1]) for column in keypoints.columns if column.endswith("_x")})
        return pd.DataFrame(
            {
                "path": keypoints["path"],
                "keypoints": [
                    [
                        float(value) if not pd.isna(value) else np.nan
                        for point_index in point_indices
                        for value in (row[f"keypoint_{point_index:02d}_x"], row[f"keypoint_{point_index:02d}_y"])
                    ]
                    for _, row in keypoints.iterrows()
                ],
                "keypoint_scores": [
                    [
                        float(row[f"keypoint_{point_index:02d}_score"])
                        if not pd.isna(row[f"keypoint_{point_index:02d}_score"])
                        else np.nan
                        for point_index in point_indices
                    ]
                    for _, row in keypoints.iterrows()
                ],
                "min_keypoint_score": keypoints["min_keypoint_score"],
                "mean_keypoint_score": keypoints["mean_keypoint_score"],
                "n_out_of_bounds_keypoints": keypoints["n_out_of_bounds_keypoints"],
            }
        )

    def create_catalogue(self, load_keypoints: bool = False) -> pd.DataFrame:
        """
        Create the catalogue DataFrame for the prepared BrownBearHeads dataset.

        This loader expects the Kaggle reupload prepared for Wildlife Datasets.
        Unlike the original raw BrownBear_ReID release, the prepared version
        already provides a unified `metadata.csv`, so the catalogue can be
        loaded directly without reconstructing split membership from the
        original experiment CSV files.

        Returns:
            pd.DataFrame: A dataframe containing one row per prepared image.
            The dataframe is expected to include columns such as:

                - identity (str): Individual brown bear identity label.
                - path (str): Relative path to the prepared image file.
                - width (int): Resized image width.
                - height (int): Resized image height.
                - width_original (int): Original image width before resizing.
                - height_original (int): Original image height before resizing.
                - date (datetime): Observation timestamp.
                - year (int): Observation year.
                - camera (str): Camera or camera-model metadata.
                - split_2017 ... split_2022 (str): Original yearly split roles.
                - split_ood (str): Standardized out-of-distribution split.
                - split_iid (str): Standardized in-distribution split.
                - image_id (int): Stable image identifier.
                - species (str): Added automatically as `brown bear` if missing.
                - keypoints (list[float], optional): Face keypoints loaded from
                  `head_keypoints.csv` when `load_keypoints=True`.
                - keypoint_scores (list[float], optional): Per-keypoint model
                  scores loaded from `head_keypoints.csv` when requested.
        """

        assert self.root is not None
        df = pd.read_csv(os.path.join(self.root, "metadata.csv"), low_memory=False)
        if "image_id" not in df.columns:
            df["image_id"] = range(len(df))

        if load_keypoints:
            keypoints = pd.read_csv(os.path.join(self.root, self.head_keypoints_file), low_memory=False)
            keypoints = self.parse_keypoints(keypoints)
            df = df.merge(keypoints, on="path", how="left")

        if "species" not in df.columns:
            df["species"] = "brown bear"

        return self.finalize_catalogue(df)
