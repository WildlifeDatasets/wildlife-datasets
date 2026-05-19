import io
import os

import numpy as np
import pandas as pd
from datasets import DownloadConfig, load_dataset
from datasets import Image as HFImage
from PIL import Image as PILImage

from .datasets import WildlifeDataset
from .downloads import DownloadHuggingFace

summary = {
    "licenses": "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)",
    "licenses_url": "https://creativecommons.org/licenses/by-nc/4.0/",
    "url": "https://huggingface.co/datasets/megretlab/red_bee_reID",
    "publication_url": "https://openaccess.thecvf.com/content/WACV2026/papers/Meyers_One-Shot_Fine-Grained_Re-Identification_of_Paint_Marked_Honey_Bees_using_Vision_WACV_2026_paper.pdf",
    "cite": "meyers2026one",
    "animals": {"honey bees"},
    "animals_simple": "bees",
    "real_animals": True,
    "year": 2026,
    "reported_n_total": 9495,
    "reported_n_individuals": 45,
    "wild": False,
    "clear_photos": True,
    "pose": "multiple",
    "unique_pattern": False,
    "from_video": True,
    "cropped": True,
    "span": "3 days",
    "size": 9495,
}


class RedBeeReID(DownloadHuggingFace, WildlifeDataset):
    summary = summary
    hf_url = "megretlab/red_bee_reID"
    image_columns = ("rotated_masked", "unrotated_unmasked")
    keypoint_names = ("head", "neck", "thorax", "waist", "tail")

    def __init__(self, *args, image_column: str = "rotated_masked", **kwargs):
        self._check_image_column(image_column)
        self.image_column = image_column
        super().__init__(*args, image_column=image_column, **kwargs)

    @classmethod
    def _download(cls, image_column: str = "rotated_masked", download_config: DownloadConfig | None = None, **kwargs):
        cls._check_image_column(image_column)
        if download_config is None:
            download_config = DownloadConfig(disable_tqdm=False, download_desc=cls.__name__)
        super()._download(download_config=download_config, **kwargs)

    @classmethod
    def _check_image_column(cls, image_column: str) -> None:
        if image_column not in cls.image_columns:
            raise ValueError(f"image_column must be one of {cls.image_columns}.")

    @staticmethod
    def _get_image_sizes(split, image_column: str) -> pd.DataFrame:
        split = split.cast_column(image_column, HFImage(decode=False))

        def get_size(image):
            if image["path"] is not None and os.path.exists(image["path"]):
                image_file = image["path"]
            else:
                image_file = io.BytesIO(image["bytes"])
            with PILImage.open(image_file) as img:
                return img.size

        sizes = [get_size(row[image_column]) for row in split]
        return pd.DataFrame(sizes, columns=["image_width", "image_height"])

    @staticmethod
    def _points_to_crop(df: pd.DataFrame, points: np.ndarray, image_column: str) -> np.ndarray:
        points = points - df[["cx", "cy"]].to_numpy()[:, None, :]

        if image_column == "rotated_masked":
            angle = np.deg2rad(-90 - df["angle"].to_numpy())
            cos = np.cos(angle)[:, None]
            sin = np.sin(angle)[:, None]
            x, y = points[..., 0], points[..., 1]
            points = np.stack([x * cos - y * sin, x * sin + y * cos], axis=2)

        points += df[["image_width", "image_height"]].to_numpy()[:, None, :] / 2
        return points

    @classmethod
    def _annotations_to_crop(cls, df: pd.DataFrame, image_column: str) -> None:
        keypoint_cols = [f"{name}_{axis}" for name in cls.keypoint_names for axis in ("x", "y")]
        keypoint_source = df[keypoint_cols].to_numpy()
        source_center_cols = ["bbox_x", "bbox_y", "bbox_width", "bbox_height"]
        bbox_source_center = df[source_center_cols].to_numpy()
        bbox_source = bbox_source_center.copy()
        bbox_source[:, :2] -= bbox_source[:, 2:] / 2

        keypoints = cls._points_to_crop(df, keypoint_source.reshape(len(df), -1, 2), image_column).reshape(len(df), -1)

        offsets = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        corners = bbox_source[:, None, :2] + bbox_source[:, None, 2:] * offsets
        corners = cls._points_to_crop(df, corners, image_column)

        image_size = df[["image_width", "image_height"]].to_numpy()
        top_left = np.clip(corners.min(axis=1), 0, image_size)
        bottom_right = np.clip(corners.max(axis=1), 0, image_size)
        bbox = np.column_stack([top_left, bottom_right - top_left])

        df["keypoints_source"] = pd.Series(list(keypoint_source))
        df["bbox_source_center"] = pd.Series(list(bbox_source_center))
        df["bbox_source"] = pd.Series(list(bbox_source))
        df["keypoints"] = pd.Series(list(keypoints))
        df["bbox"] = pd.Series(list(bbox))

    def create_catalogue(
        self,
        image_column: str = "rotated_masked",
        download_config: DownloadConfig | None = None,
    ) -> pd.DataFrame:
        self._check_image_column(image_column)
        self.image_column = image_column
        if download_config is None:
            download_config = DownloadConfig(disable_tqdm=False, download_desc=self.__class__.__name__)
        self.dataset = load_dataset(self.hf_url, download_config=download_config)

        split = self.dataset["train"]
        metadata = split.remove_columns(list(self.image_columns)).to_pandas()
        metadata = pd.concat([metadata, self._get_image_sizes(split, image_column)], axis=1)
        self._annotations_to_crop(metadata, image_column)
        metadata["image_id"] = np.arange(len(metadata))
        metadata["identity"] = metadata["bee_id"].apply(
            lambda identity: self.unknown_name if pd.isna(identity) else str(int(identity))
        )
        metadata["path"] = np.nan
        metadata["hf_index"] = metadata["image_id"]
        metadata["split_original"] = "train"
        metadata["species"] = "honey bee"
        metadata["video"] = metadata["video_key"].astype(int)

        return self.finalize_catalogue(metadata)

    def get_image(self, idx):
        if not hasattr(self, "dataset"):
            self.dataset = load_dataset(self.hf_url)
        hf_index = int(self.df["hf_index"].iloc[idx])
        return self.dataset["train"][hf_index][self.image_column]
