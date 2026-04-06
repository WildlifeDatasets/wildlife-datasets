import os

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
    "reported_n_total": 72940,
    "reported_n_individuals": 109,
    "wild": True,
    "clear_photos": True,
    "pose": "single",
    "unique_pattern": False,
    "from_video": False,
    "cropped": True,
    "span": "6 years",
    "size": 32900,
}


class BrownBearHeads(DownloadKaggle, WildlifeDataset):
    summary = summary
    kaggle_url = "picekl/brown-bear-heads"
    kaggle_type = "datasets"

    def create_catalogue(self) -> pd.DataFrame:
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
                - image_id (int): Added automatically if missing.
                - species (str): Added automatically as `brown bear` if missing.

        Notes:
            - The method looks for `metadata.csv` either directly in `self.root`
              or in a nested `BrownBearHeads/` folder.
            - If `image_id` is not present in the metadata, a sequential one is
              created.
            - If `species` is not present in the metadata, it is set to
              `brown bear`.
        """

        assert self.root is not None
        metadata_path = self.find_metadata_path()
        df = pd.read_csv(metadata_path, low_memory=False)

        if "image_id" not in df.columns:
            df["image_id"] = range(len(df))
        if "species" not in df.columns:
            df["species"] = "brown bear"

        return self.finalize_catalogue(df)

    def find_metadata_path(self) -> str:
        """
        Find the prepared BrownBearHeads metadata file.

        The Kaggle reupload may store `metadata.csv` either directly in the
        dataset root or inside a nested `BrownBearHeads/` directory.

        Returns:
            str: Absolute path to the discovered `metadata.csv` file.

        Raises:
            FileNotFoundError: If `metadata.csv` cannot be found in any
            supported location.
        """

        assert self.root is not None
        candidates = [
            os.path.join(self.root, "metadata.csv"),
            os.path.join(self.root, "BrownBearHeads", "metadata.csv"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError("Could not find BrownBearHeads metadata.csv in the provided root.")
