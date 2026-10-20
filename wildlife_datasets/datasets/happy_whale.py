import os

import numpy as np
import pandas as pd

from . import utils
from .datasets import WildlifeDataset
from .downloads import DownloadKaggle

summary = {
    "licenses": "Other",
    "licenses_url": "https://www.kaggle.com/competitions/happy-whale-and-dolphin/rules/#7.-competition-data.",
    "url": "https://www.kaggle.com/competitions/happy-whale-and-dolphin",
    "publication_url": "https://link.springer.com/article/10.1007/s42991-021-00180-9",
    "cite": "cheeseman2021advanced",
    "animals": {"dolphin", "whale"},
    "animals_simple": "dolphins+whales",
    "real_animals": True,
    "original_dataset": True,
    "year": 2022,
    "reported_n_total": None,
    "reported_n_individuals": None,
    "wild": True,
    "clear_photos": True,
    "pose": "multiple",
    "unique_pattern": True,
    "from_video": False,
    "cropped": False,
    "span": "very long",
    "size": 61912,
    "description": "HappyWhale, HumpbackWhale and NOAARightWhale are datasets of various whale species. They are a product of multi-year collaboration of multiple research institutions and citizen scientists. All these datasets were released as Kaggle competitions to make it easy and rewarding for the public to participate in science by building innovative tools to engage anyone interested in marine mammals. The whales were photographed during aerial surveys. HumpbackWhale is the most uniform dataset with a clear view on the whale tail above water. NOAARightWhale contains images of submerging whales. HappyWhale is the most diverse dataset with images of dorsal fins. Some image contain only the dorsal fin, while others contain a significant part of the whale body.",
    "processing": "Kaggle required + terms",
    "modifications": "Species fixed",
}


class HappyWhale(DownloadKaggle, WildlifeDataset):
    summary = summary
    kaggle_url = "happy-whale-and-dolphin"
    kaggle_type = "competitions"

    def create_catalogue(self) -> pd.DataFrame:
        # Load the training data
        root = self.get_root()
        data = pd.read_csv(os.path.join(root, "train.csv"))
        df1 = pd.DataFrame(
            {
                "image_id": data["image"].str.split(".", expand=True)[0],
                "path": "train_images" + os.path.sep + data["image"],
                "identity": data["individual_id"],
                "species": data["species"],
                "original_split": "train",
            }
        )

        test_files = utils.find_images(os.path.join(root, "test_images"))
        test_files = list(test_files["file"])
        test_files = pd.Series(np.sort(test_files))

        # Load the testing data
        df2 = pd.DataFrame(
            {
                "image_id": test_files.str.split(".", expand=True)[0],
                "path": "test_images" + os.path.sep + test_files,
                "identity": self.unknown_name,
                "species": np.nan,
                "original_split": "test",
            }
        )

        # Finalize the dataframe
        df = pd.concat([df1, df2])
        return self.finalize_catalogue(df)

    def fix_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        # Replace the wrong species names
        replace_identity = [
            ("bottlenose_dolpin", "bottlenose_dolphin"),
            ("kiler_whale", "killer_whale"),
        ]
        return self.fix_labels_replace_identity(df, replace_identity, col="species")
