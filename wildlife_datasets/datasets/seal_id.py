import logging
import os
import shutil

import pandas as pd

from . import utils
from .datasets import WildlifeDataset

logger = logging.getLogger(__name__)

summary = {
    "licenses": "Attribution 4.0 International (CC BY 4.0)",
    "licenses_url": "https://creativecommons.org/licenses/by/4.0/",
    "url": "https://doi.org/10.23729/0f4a3296-3b10-40c8-9ad3-0cf00a5a4a53",
    "publication_url": "https://www.mdpi.com/1424-8220/22/19/7602",
    "cite": "nepovinnykh2022sealid",
    "animals": {"ringed seal"},
    "animals_simple": "seals",
    "real_animals": True,
    "original_dataset": True,
    "year": 2022,
    "n_images": 2080,
    "n_images_identified": 2080,
    "n_individuals": 57,
    "wild": True,
    "clear_photos": False,
    "pose": "multiple",
    "unique_pattern": True,
    "from_video": False,
    "cropped": False,
    "span": "10 years",
    "size": 1816,
    "description": "SealID is a Saimaa ringed seals database from the Lake Saimaa in Finland. The data were collected annually during the Saimaa ringed seal molting season from 2010 to 2019 by both ordinary digital cameras during boat surveys and game camera traps. The GPS coordinates, the observation times, and the numbers of the seals were noted.  Seal images were matched by an expert using individually characteristic fur patterns. The dataset contains patches and syandard images. Patches show small patterned body parts which are sufficient for seal identification. Standard images are presented both as full images and their segmented version with seal only and black background.",
    "processing": "Download with single use token",
}


class SealID(WildlifeDataset):
    summary = summary
    prefix = "source_"
    archive = "22b5191e-f24b-4457-93d3-95797c900fc0_ui65zipk.zip"

    @classmethod
    def _download(cls, url=None):
        if url is None:
            raise ValueError(
                "URL must be provided for SealID.\nCheck https://wildlifedatasets.github.io/wildlife-datasets/preprocessing/#sealid"
            )
        utils.download_url(url, cls.archive)

    @classmethod
    def _extract(cls, **kwargs):
        utils.extract_archive(cls.archive, delete=True)
        utils.extract_archive(os.path.join("SealID", "full images.zip"), delete=True)
        utils.extract_archive(os.path.join("SealID", "patches.zip"), delete=True)

        # Create new folder for segmented images
        folder_new = os.getcwd() + "Segmented"
        if not os.path.exists(folder_new):
            os.makedirs(folder_new)

        # Move segmented images to new folder
        folder_move = os.path.join("patches", "segmented")
        shutil.move(folder_move, os.path.join(folder_new, folder_move))
        folder_move = os.path.join("full images", "segmented_database")
        shutil.move(folder_move, os.path.join(folder_new, folder_move))
        folder_move = os.path.join("full images", "segmented_query")
        shutil.move(folder_move, os.path.join(folder_new, folder_move))
        file_copy = os.path.join("patches", "annotation.csv")
        shutil.copy(file_copy, os.path.join(folder_new, file_copy))
        file_copy = os.path.join("full images", "annotation.csv")
        shutil.copy(file_copy, os.path.join(folder_new, file_copy))

    def create_catalogue(self) -> pd.DataFrame:
        # Load information about the dataset
        root = self.get_root()
        data = pd.read_csv(os.path.join(root, "full images", "annotation.csv"))

        # Finalize the dataframe
        df = pd.DataFrame(
            {
                "image_id": data["file"].str.split(".", expand=True)[0],
                "path": "full images" + os.path.sep + self.prefix + data["reid_split"] + os.path.sep + data["file"],
                "identity": data["class_id"].astype(int),
                "original_split": data["segmentation_split"].replace(
                    {"training": "train", "testing": "test", "validation": "val"}
                ),
                "original_split_reid": data["reid_split"],
            }
        )
        return self.finalize_catalogue(df)


class SealIDSegmented(SealID):
    prefix = "segmented_"
    warning = """You are trying to download or extract a segmented dataset.
        It is already included in its non-segmented version.
        Skipping."""

    @classmethod
    def get_data(cls, *args, **kwargs):
        logger.warning(cls.warning)

    @classmethod
    def _download(cls, *args, **kwargs):
        logger.warning(cls.warning)

    @classmethod
    def _extract(cls, *args, **kwargs):
        logger.warning(cls.warning)
