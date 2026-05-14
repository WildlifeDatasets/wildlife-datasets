import numpy as np
import pandas as pd
from datasets import DownloadConfig, load_dataset

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
        keypoint_cols = [
            "head_x",
            "head_y",
            "neck_x",
            "neck_y",
            "thorax_x",
            "thorax_y",
            "waist_x",
            "waist_y",
            "tail_x",
            "tail_y",
        ]
        bbox_cols = ["bbox_x", "bbox_y", "bbox_width", "bbox_height"]
        metadata["keypoints"] = pd.Series(list(metadata[keypoint_cols].to_numpy()))
        metadata["bbox"] = pd.Series(list(metadata[bbox_cols].to_numpy()))
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
