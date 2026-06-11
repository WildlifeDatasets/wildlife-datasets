import os
import re

import pandas as pd

from .datasets import WildlifeDataset
from .downloads import DownloadURL

summary = {
    "licenses": "Attribution 4.0 International (CC BY 4.0)",
    "licenses_url": "https://creativecommons.org/licenses/by/4.0/",
    "url": "https://zenodo.org/records/20026776",
    "publication_url": "https://arxiv.org/abs/2601.08798",
    "cite": "yesharim2026near",
    "animals": {"Hula painted frog"},
    "animals_simple": "frogs",
    "real_animals": True,
    "year": 2026,
    "reported_n_total": 1233,
    "reported_n_individuals": 191,
    "wild": False,
    "clear_photos": True,
    "pose": "single",
    "unique_pattern": True,
    "from_video": False,
    "cropped": True,
    "span": "8 years",
    "size": 7000,
}

def get_date(x):
    x_split = x.split('-')   
    assert len(x_split) == 2
    year, month_day = x_split

    # Check whether it ends with a letter
    m = re.fullmatch(r'(\d+)([a-z]?)', month_day)
    assert m is not None

    # Convert the letter into day (no -> 1, a -> 2, b -> 3, ...)
    month = int(m.group(1))
    day_letter = m.group(2)
    day = ord(day_letter) - ord("a") + 2 if day_letter else 1

    return f"{year}-{month:02}-{day:02}"

class HulaPaintedFrogs(DownloadURL, WildlifeDataset):
    summary = summary
    downloads = [
        ("https://zenodo.org/records/20026776/files/extra.csv?download=1", "extra.csv"),
        ("https://zenodo.org/records/20026776/files/extra.zip?download=1", "extra.zip"),
        ("https://zenodo.org/records/20026776/files/labeled.csv?download=1", "labeled.csv"),
        ("https://zenodo.org/records/20026776/files/labeled.zip?download=1", "labeled.zip"),
        ("https://zenodo.org/records/20026776/files/unlabeled.csv?download=1", "unlabeled.csv"),
        ("https://zenodo.org/records/20026776/files/unlabeled.zip?download=1", "unlabeled.zip"),
    ]

    def create_catalogue(self) -> pd.DataFrame:
        labeled = pd.read_csv(f"{self.root}/labeled.csv")
        labeled["path"] = "labeled" + os.path.sep + labeled["rel_path"].str.replace("/", os.path.sep)
        unlabeled = pd.read_csv(f"{self.root}/unlabeled.csv")
        unlabeled["path"] = "unlabeled" + os.path.sep + unlabeled["rel_path"].str.replace("/", os.path.sep)
        extra = pd.read_csv(f"{self.root}/extra.csv")
        extra["path"] = "extra" + os.path.sep + extra["rel_path"].str.replace("/", os.path.sep)

        df = pd.concat((labeled, unlabeled, extra))
        df["image_id"] = range(len(df))
        df["identity"] = df["label"].apply(lambda x: None if pd.isnull(x) else str(int(x)))
        df["date"] = df["date"].apply(get_date)
        df = df.drop(["rel_path", "label"], axis=1)
        df = df.rename({"Inferred": "inferred"}, axis=1)
        
        return self.finalize_catalogue(df)