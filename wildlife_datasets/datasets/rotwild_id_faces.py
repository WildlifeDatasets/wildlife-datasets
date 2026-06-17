import os

import pandas as pd

from .datasets import WildlifeDataset
from .downloads import DownloadKaggle
from .utils import parse_bbox_mask

summary = {
    "licenses": "Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)",
    "licenses_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    "url": "https://www.kaggle.com/datasets/jonaschu/rotwildid-faces",
    "publication_url": "",
    "cite": "",
    "animals": {"red deer"},
    "animals_simple": "deers",
    "real_animals": True,
    "year": 2026,
    "reported_n_total": 1012,
    "reported_n_individuals": 59,
    "wild": True,
    "clear_photos": True,
    "pose": "single",
    "unique_pattern": False,
    "from_video": False,
    "cropped": True,
    "span": "unknown",
    "size": 183,
}


class RotwildID_Faces(DownloadKaggle, WildlifeDataset):
    summary = summary
    kaggle_url = "jonaschu/rotwildid-faces"
    kaggle_type = "datasets"

    def create_catalogue(self, image_type: str = "mask") -> pd.DataFrame:
        """
        Create the catalogue DataFrame for the RotwildID_Faces dataset.

        `image_type` must be in `("landmark_affine", "landmark_crop", "mask")`,
        it is recommended to use mask. See the documentation at the Kaggle website.

        The dataset contains keypoints for right eye, left eye and nose (in this order).
        See the keyword `keypoints` below.

        Returns:
            pd.DataFrame: A dataframe containing one row per image.
            The dataframe includes columns:

                - identity (str): Individual identity label.
                - path (str): Relative path to the image file.
                - bbox (list[float]): Bounding box. Automatic use via `img_load=bbox`.
                - segmentation (list[float]): Segmentation mask. Automatic use via `img_load=bbox_mask`.
                - keypoints (list[float]): List of keypoints (x right eye, y right eye, ...).
                - image_quality (str): Quality of the image.
        """

        assert self.root is not None

        allowed_types = ("landmark_affine", "landmark_crop", "mask")
        if image_type not in allowed_types:
            raise ValueError(f"image_type must by in {allowed_types}")

        metadata_path = os.path.join(self.root, image_type, "image_metadata.csv")
        metadata = pd.read_csv(metadata_path, index_col=0)
        metadata["path"] = image_type + os.path.sep + metadata["path"]
        metadata["bbox"] = metadata["bbox"].apply(parse_bbox_mask)
        metadata["keypoints"] = metadata["keypoints"].apply(parse_bbox_mask)
        metadata["segmentation"] = metadata["segmentation"].apply(parse_bbox_mask)

        return self.finalize_catalogue(metadata)
