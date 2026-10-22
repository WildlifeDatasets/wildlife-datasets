import os

import pandas as pd

from .datasets import WildlifeDataset
from .downloads import DownloadKaggle

summary = {
    "licenses": "Other",
    "licenses_url": "https://www.kaggle.com/datasets/wildlifedatasets/wildlifereid-10k",
    "url": "https://www.kaggle.com/datasets/wildlifedatasets/wildlifereid-10k",
    "publication_url": "https://openaccess.thecvf.com/content/CVPR2025W/FGVC/html/Adam_WildlifeReID-10k_Wildlife_re-identification_dataset_with_10k_individual_animals_CVPRW_2025_paper.html",
    "cite": "adam2025wildlifereid",
    "animals": {"multiple"},
    "animals_simple": "multiple",
    "real_animals": True,
    "original_dataset": False,
    "year": 2025,
    "n_images": 140488,
    "n_individuals": 10772,
    "wild": True,
    "clear_photos": None,
    "pose": "multiple",
    "unique_pattern": None,
    "from_video": False,
    "cropped": True,
    "span": "very long",
    "size": 24760,
    "description": "WildlifeReID-10k is a wildlife re-identification dataset with more than 140k images of 10k individual animals. It is a collection of 37 existing wildlife re-identification datasets with additional processing steps. WildlifeReID-10k contains diverse animals such as marine turtles, primates, birds, African herbivores, marine mammals, and domestic animals.",
    "processing": "Kaggle required",
}


class WildlifeReID10k(DownloadKaggle, WildlifeDataset):
    summary = summary
    kaggle_url = "wildlifedatasets/wildlifereid-10k"
    kaggle_type = "datasets"

    def create_catalogue(self) -> pd.DataFrame:
        root = self.get_root()
        df = pd.read_csv(os.path.join(root, "metadata.csv"), low_memory=False)
        df["image_id"] = range(len(df))
        return self.finalize_catalogue(df)
