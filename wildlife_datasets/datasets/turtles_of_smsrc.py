import os
import json
import ast

import numpy as np
import pandas as pd
from .downloads import DownloadINaturalist
from .datasets import WildlifeDataset

summary = {
    "licenses": "",
    "licenses_url": "",
    "url": "",
    "publication_url": None,
    "cite": "",
    "animals": {"Green Sea Turtle"},
    "animals_simple": "sea turtles",
    "real_animals": True,
    "year": 2025,
    "reported_n_total": None,
    "reported_n_individuals": None,
    "wild": True,
    "clear_photos": True,
    "pose": "multiple",
    "unique_pattern": True,
    "from_video": False,
    "cropped": False,
    "span": "",
    "size": None,
}


class TurtlesOfSMSRC(DownloadINaturalist, WildlifeDataset):
    summary = summary
    project_id = "turtles-of-smsrc"
    metadata_fields = (
        "species_guess",
        "observed_on",
        "location",
    )

    def create_catalogue(self) -> pd.DataFrame:
        df = pd.read_csv(os.path.join(self.root, "metadata.csv"))
        df["image_id"] = df["observation_id"].astype(str) + "_" + df["photo_id"].astype(str)
        df["identity"] = "unknown"
        df["latitute"] = df["location"].apply(lambda x: ast.literal_eval(x)[0]).astype(float)
        df["longitude"] = df["location"].apply(lambda x: ast.literal_eval(x)[1]).astype(float)
        df = df.drop(["location", "photo_id", "photo_url"], axis=1)
        df = df.rename({
            "observation_id": "encounter_id",
            "species_guess": "species",
            "file_name": "path",
            "observed_on": "date"
            }, axis=1)

        return self.finalize_catalogue(df)
