import ast

import pandas as pd

from .downloads import DownloadINaturalist
from .general import Dataset_Metadata

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
    "span": "1 year",
    "size": None,
}


class TurtlesOfSMSRC(DownloadINaturalist, Dataset_Metadata):
    summary = summary
    project_id = "turtles-of-smsrc"
    metadata_fields = (
        "species_guess",
        "observed_on",
        "location",
    )

    def create_catalogue(self, *, replace_identity: list[tuple] | None = None, **kwargs) -> pd.DataFrame:
        self.replace_identity = [] if replace_identity is None else replace_identity
        return super().create_catalogue(**kwargs)

    def modify_metadata(self, metadata: pd.DataFrame) -> pd.DataFrame:
        metadata["image_id"] = metadata["observation_id"].astype(str) + "_" + metadata["photo_id"].astype(str)
        metadata["identity"] = metadata["observation_id"]
        mask = ~metadata["location"].isnull()
        metadata["latitute"] = metadata.loc[mask, "location"].apply(lambda x: ast.literal_eval(x)[0]).astype(float)
        metadata["longitude"] = metadata.loc[mask, "location"].apply(lambda x: ast.literal_eval(x)[1]).astype(float)
        metadata = metadata.drop(["location", "photo_id", "photo_url"], axis=1)
        metadata = metadata.rename(
            {"observation_id": "encounter_id", "species_guess": "species", "file_name": "path", "observed_on": "date"},
            axis=1,
        )
        return metadata

    def fix_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fix_labels_replace_identity(df, self.replace_identity)
