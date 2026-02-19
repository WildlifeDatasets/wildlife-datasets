import os

import pandas as pd

from . import utils
from .datasets import WildlifeDataset
from .downloads import DownloadURL

summary = {
    "licenses": "Creative Commons Attribution 4.0 International",
    "licenses_url": "https://creativecommons.org/licenses/by/4.0/legalcode",
    "url": "https://zenodo.org/records/7564529",
    "publication_url": "https://www.mdpi.com/2076-2615/13/5/801",
    "cite": "zuerl2023polarbearvidid",
    "animals": {"polar bear"},
    "animals_simple": "polar bears",
    "real_animals": True,
    "year": 2023,
    "reported_n_total": 138363,
    "reported_n_individuals": 13,
    "wild": False,
    "clear_photos": True,
    "pose": "multiple",
    "unique_pattern": False,
    "from_video": True,
    "cropped": True,
    "span": "short",
    "size": 1501,
}


class PolarBearVidID(DownloadURL, WildlifeDataset):
    summary = summary
    url = "https://zenodo.org/records/7564529/files/PolarBearVidID.zip?download=1"
    archive = "PolarBearVidID.zip"

    def create_catalogue(self) -> pd.DataFrame:
        assert self.root is not None
        metadata = pd.read_csv(os.path.join(self.root, "animal_db.csv"))
        data = utils.find_images(self.root)

        # Finalize the dataframe
        df = pd.DataFrame(
            {
                "image_id": data["file"].apply(lambda x: os.path.splitext(x)[0]),
                "path": data["path"] + os.path.sep + data["file"],
                "video": data["file"].str[7:10].astype(int),
                "id": data["path"].astype(int),
            }
        )
        df = pd.merge(df, metadata, on="id", how="left")
        df.rename({"name": "identity", "sex": "gender"}, axis=1, inplace=True)
        df = df.drop(["id", "zoo", "tracklets"], axis=1)
        return self.finalize_catalogue(df)
